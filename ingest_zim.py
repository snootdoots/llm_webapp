#!/usr/bin/env python3
"""
Streaming, resumable ZIM → FAISS indexer for offline Wikipedia.

Key features
- Streams the pipeline (parse → split → embed → index) to keep memory flat
- Uses FAISS HNSW (no training phase) and a SQLite docstore on disk
- Periodic checkpointing and index/docstore saves (crash-safe resume)
- Concurrency for embedding calls to Ollama to improve throughput
- Skips non-HTML entries and filters to main content area when possible
"""
import os, sys, gc, json, time, sqlite3, threading, signal
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np, requests, faiss
from bs4 import BeautifulSoup
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from libzim.reader import Archive

# ---------------------------
# Config
# ---------------------------
ZIM_PATH = os.environ.get("ZIM_PATH", "wikipedia_en_top_nopic_2025-08.zim")
INDEX_PATH = os.environ.get("INDEX_PATH", "faiss_wiki_hnsw.index")
DOCSTORE_PATH = os.environ.get("DOCSTORE_PATH", "docstore.sqlite")
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "progress.json")

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))

DOCS_PER_PARSE_BATCH = int(os.environ.get("DOCS_PER_PARSE_BATCH", "50"))
CHUNKS_PER_EMB_BATCH = int(os.environ.get("CHUNKS_PER_EMB_BATCH", "64"))
EMBED_WORKERS = int(os.environ.get("EMBED_WORKERS", "4"))
SAVE_EVERY_CHUNKS = int(os.environ.get("SAVE_EVERY_CHUNKS", "10000"))

CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1500"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))

HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64

MIN_TEXT_LEN = 500
MAX_CHUNKS_PER_DOC = 200
MAX_CHUNKS_PER_BATCH = 2000

# ---------------------------
# Checkpointing utilities
# ---------------------------
_ckpt_lock = threading.Lock()

def load_checkpoint(path: str):
    if not os.path.exists(path):
        return {"last_entry_id_seen": -1, "last_entry_id_persisted": -1, "next_id_persisted": 0}
    with open(path, "r", encoding="utf-8") as f:
        ck = json.load(f)
    ck.setdefault("last_entry_id_seen", ck.get("last_entry_id", -1))
    ck.setdefault("last_entry_id_persisted", ck.get("last_entry_id", -1))
    ck.setdefault("next_id_persisted", ck.get("vectors_added", 0))
    return ck

def save_checkpoint(path: str, last_entry_id_seen: int, last_entry_id_persisted: int, next_id_persisted: int):
    with _ckpt_lock:
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({
                "last_entry_id_seen": int(last_entry_id_seen),
                "last_entry_id_persisted": int(last_entry_id_persisted),
                "next_id_persisted": int(next_id_persisted),
            }, f)
        os.replace(tmp, path)

# ---------------------------
# SQLite + FAISS utilities
# ---------------------------
def open_docstore(db_path: str):
    init = not os.path.exists(db_path)
    conn = sqlite3.connect(db_path)
    if init:
        with conn:
            conn.execute("CREATE TABLE chunks (id INTEGER PRIMARY KEY, text TEXT, meta TEXT);")
    return conn

def next_doc_id(conn: sqlite3.Connection) -> int:
    cur = conn.execute("SELECT COALESCE(MAX(id), -1) FROM chunks")
    row = cur.fetchone()
    return int(row[0]) + 1 if row and row[0] is not None else 0

def insert_chunks(conn, rows):
    with conn:
        conn.executemany("INSERT INTO chunks(id,text,meta) VALUES(?,?,?)", rows)

def open_or_create_hnsw(dimension: int, index_path: str):
    if os.path.exists(index_path):
        index = faiss.read_index(index_path)
        if not isinstance(index, faiss.IndexIDMap):
            index = faiss.IndexIDMap(index)
        try:
            inner = index.index if hasattr(index, "index") else index
            inner.hnsw.efSearch = HNSW_EF_SEARCH
        except Exception:
            pass
        return index
    base = faiss.IndexHNSWFlat(dimension, HNSW_M)
    base.hnsw.efConstruction = HNSW_EF_CONSTRUCTION
    base.hnsw.efSearch = HNSW_EF_SEARCH
    return faiss.IndexIDMap(base)

def save_faiss(index, path):
    tmp = path + ".tmp"
    faiss.write_index(index, tmp)
    os.replace(tmp, path)

def persist_all(index, conn, last_entry_id_persisted, next_id_persisted, last_entry_id_seen):
    save_faiss(index, INDEX_PATH)
    conn.commit()
    save_checkpoint(CHECKPOINT_PATH, last_entry_id_seen, last_entry_id_persisted, next_id_persisted)
    print(f"[checkpoint] persisted entry={last_entry_id_persisted}, next_id={next_id_persisted}")

# ---------------------------
# Embeddings (Ollama HTTP or HF in-process)
# ---------------------------
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "ollama").lower()  # 'ollama' or 'hf'
HF_BATCH_SIZE = int(os.environ.get("HF_BATCH_SIZE", "256"))

# Optional deps for HF backend
try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _ST_AVAILABLE = True
except Exception:
    _ST_AVAILABLE = False

try:
    import torch  # type: ignore
    _TORCH_AVAILABLE = True
    # Let torch use all threads unless user overrides
    if "TORCH_NUM_THREADS" not in os.environ:
        torch.set_num_threads(os.cpu_count() or 4)
except Exception:
    _TORCH_AVAILABLE = False


def _embed_one_ollama(text: str, session: requests.Session):
    # Robust single-text embedding with retries
    for attempt in range(5):
        try:
            resp = session.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
            vec = data.get("embedding") or (data.get("data", [{}])[0].get("embedding") if isinstance(data.get("data"), list) else None)
            if vec is None:
                raise RuntimeError("No embedding in response")
            arr = np.asarray(vec, dtype="float32")
            if arr.shape[0] != EMBED_DIM:
                raise RuntimeError(f"Unexpected embedding dim {arr.shape[0]} != {EMBED_DIM}. Set EMBED_DIM correctly or change model.")
            return arr
        except Exception:
            time.sleep(0.5 * (2 ** attempt))
    raise RuntimeError("Failed to embed after retries")


def embed_texts(texts):
    """Embed a list of texts → np.ndarray [n, EMBED_DIM].
    Backends:
      - EMBED_BACKEND=hf : SentenceTransformers in-process (CPU/MPS/CUDA)
      - EMBED_BACKEND=ollama : HTTP to Ollama server (threaded requests)
    """
    if not texts:
        return np.zeros((0, EMBED_DIM), dtype="float32")

    if EMBED_BACKEND == "hf":
        if not _ST_AVAILABLE:
            raise RuntimeError("SentenceTransformers not installed. pip install sentence-transformers")
        device = "mps" if _TORCH_AVAILABLE and hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else ("cuda" if _TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        model = getattr(embed_texts, "_hf_model", None)
        if model is None or getattr(embed_texts, "_hf_model_name", "") != EMBED_MODEL:
            model = SentenceTransformer(EMBED_MODEL, device=device)
            embed_texts._hf_model = model
            embed_texts._hf_model_name = EMBED_MODEL
        vecs = model.encode(texts, batch_size=HF_BATCH_SIZE, show_progress_bar=False, normalize_embeddings=False, convert_to_numpy=True, device=device)
        vecs = np.asarray(vecs, dtype="float32")
        if vecs.ndim == 1:
            vecs = vecs.reshape(1, -1)
        if vecs.shape[1] != EMBED_DIM:
            raise RuntimeError(f"EMBED_DIM={EMBED_DIM} but model produced dim={vecs.shape[1]}. Set EMBED_DIM accordingly.")
        return vecs

    # Default: Ollama HTTP threaded
    with requests.Session() as s:
        out = np.zeros((len(texts), EMBED_DIM), dtype="float32")
        with ThreadPoolExecutor(max_workers=EMBED_WORKERS) as pool:
            futs = {pool.submit(_embed_one_ollama, t, s): i for i, t in enumerate(texts)}
            for fut in as_completed(futs):
                out[futs[fut]] = fut.result()
        return out

# ---------------------------
# HTML extraction & filters
# ---------------------------
def extract_main_text(html_bytes: bytes) -> str:
    try:
        soup = BeautifulSoup(html_bytes, "lxml")
    except Exception:
        soup = BeautifulSoup(html_bytes, "html.parser")
    for tag in soup.select("script, style, nav, header, footer"):
        try: tag.decompose()
        except Exception: pass
    main = soup.select_one(".mw-parser-output") or getattr(soup, "body", None) or soup
    text = main.get_text("\n", strip=True) if main else soup.get_text("\n", strip=True)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)

def is_valid_article(entry, mime, text):
    if not mime.startswith("text/html"): return False
    if not text or len(text) < MIN_TEXT_LEN: return False
    title = getattr(entry, "title", "") or ""
    if title.startswith(("Category:","File:","Help:","Template:","Talk:")): return False
    if title.startswith(("List of ","Timeline of ","Index of ")): return False
    return True

# ---------------------------
# ZIM iteration helper
# ---------------------------

def iter_entries(zim: Archive, start_id: int):
    """Yield (i, entry) starting from start_id using public iterator when possible."""
    yielded_any = False
    # Try public iterator first
    try:
        it = zim.iter_articles()
        for i, entry in enumerate(it):
            if i < start_id:
                continue
            yield i, entry
            yielded_any = True
        if yielded_any:
            return
    except Exception:
        pass
    # Fallback to index-based access
    total = getattr(zim, "entry_count", 0)
    for i in range(max(0, start_id), total):
        try:
            entry = zim._get_entry_by_id(i)
            yield i, entry
        except Exception:
            continue

# ---------------------------
# Pipeline
# ---------------------------
def main():
    zim = Archive(ZIM_PATH)
    total = zim.entry_count
    print(f"Opening ZIM: {ZIM_PATH}, entries={total}")

    index = open_or_create_hnsw(EMBED_DIM, INDEX_PATH)
    conn = open_docstore(DOCSTORE_PATH)
    ck = load_checkpoint(CHECKPOINT_PATH)
    last_entry_id_seen = ck["last_entry_id_seen"]
    last_entry_id_persisted = ck["last_entry_id_persisted"]
    next_id_persisted = ck["next_id_persisted"]

    ntotal = int(getattr(index, "ntotal", 0))
    if ntotal < next_id_persisted:
        next_id_persisted = ntotal
    elif ntotal > next_id_persisted:
        next_id_persisted = ntotal
    with conn: conn.execute("DELETE FROM chunks WHERE id >= ?", (next_id_persisted,))

    next_id = next_id_persisted
    vectors_added = next_id_persisted
    last_entry_id = last_entry_id_persisted

    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    parse_buffer = []
    pbar = tqdm(total=total-(last_entry_id+1), desc="Entries", unit="entry")

    stop = {"flag": False}
    def handle_sig(sig, frame):
        stop["flag"] = True
    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    for i, entry in iter_entries(zim, start_id=last_entry_id+1):
        try:
            try: item = entry.get_item()
            except Exception: continue
            mime = item.mimetype or ""
            if not mime.startswith("text/html"): continue
            html_bytes = item.content
            if hasattr(html_bytes, "tobytes"): html_bytes = html_bytes.tobytes()
            try: text = extract_main_text(html_bytes)
            except Exception: continue
            if not is_valid_article(entry, mime, text): continue
            meta = {"title": getattr(entry, "title", None), "entry_id": i}
            parse_buffer.append((text, meta))
            if len(parse_buffer) >= DOCS_PER_PARSE_BATCH:
                next_id, vectors_added, last_entry_id = process_batch(parse_buffer, splitter, conn, index, next_id, vectors_added, i)
                parse_buffer.clear()
                last_entry_id_persisted = i
                next_id_persisted = next_id
                persist_all(index, conn, last_entry_id_persisted, next_id_persisted, i)
            if stop["flag"]:
                persist_all(index, conn, last_entry_id_persisted, next_id_persisted, i)
                print("Interrupted, state persisted. Bye.")
                sys.exit(0)
        finally:
            pbar.update(1)
            last_entry_id_seen = i
            save_checkpoint(CHECKPOINT_PATH, last_entry_id_seen, last_entry_id_persisted, next_id_persisted)

    if parse_buffer:
        next_id, vectors_added, last_entry_id = process_batch(parse_buffer, splitter, conn, index, next_id, vectors_added, last_entry_id)
        parse_buffer.clear()
        last_entry_id_persisted = last_entry_id
        next_id_persisted = next_id
        persist_all(index, conn, last_entry_id_persisted, next_id_persisted, last_entry_id_seen)

    pbar.close()
    print(f"Done. Persisted vectors: {next_id_persisted}")

def process_batch(parse_buffer, splitter, conn, index, next_id, vectors_added, last_entry_id):
    docs = []
    for text, meta in parse_buffer:
        c=0
        for chunk in splitter.split_text(text):
            docs.append((chunk, meta))
            c+=1
            if c>=MAX_CHUNKS_PER_DOC: break
    if len(docs)>MAX_CHUNKS_PER_BATCH:
        docs = docs[:MAX_CHUNKS_PER_BATCH]
    for start in range(0, len(docs), CHUNKS_PER_EMB_BATCH):
        sub = docs[start:start+CHUNKS_PER_EMB_BATCH]
        texts = [t for t,_ in sub]
        metas = [m for _,m in sub]
        vecs = embed_texts(texts)
        n = vecs.shape[0]
        ids = np.arange(next_id, next_id+n, dtype="int64")
        rows = [(int(ids[k]), texts[k], json.dumps(metas[k])) for k in range(n)]
        insert_chunks(conn, rows)
        index.add_with_ids(vecs, ids)
        next_id += n
        vectors_added += n
    return next_id, vectors_added, last_entry_id

if __name__=="__main__":
    main()
