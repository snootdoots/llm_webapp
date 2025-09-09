#!/usr/bin/env python3
"""
Repair & resume helper for FAISS index + SQLite docstore + checkpoint.

- Creates timestamped backups of the FAISS index, SQLite docstore, and checkpoint.
- Reads FAISS d (dimension) and ntotal (persisted vectors).
- Reads SQLite row counts, max id, and last ZIM entry_id from meta JSON.
- Reconciles: if SQLite is ahead of FAISS, trims extra rows (optional dry-run).
- Rebuilds a coherent progress.json that reflects persisted state only.
- Optional dimension guard: refuse if FAISS dim != --expect-dim (to prevent mix-ups).

Usage:
  python repair_resume.py \
    --index faiss_wiki_hnsw.index \
    --docstore docstore.sqlite \
    --checkpoint progress.json \
    [--expect-dim 384] \
    [--dry-run]

After success, re-run your ingester with matching EMBED_BACKEND/MODEL/EMBED_DIM.
"""

import argparse, datetime, json, os, shutil, sqlite3, sys
from typing import Tuple, Optional

def ts():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def backup(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    dest = f"{path}.bak_{ts()}"
    shutil.copy2(path, dest)
    return dest

def read_faiss(index_path: str) -> Tuple[int, int]:
    import faiss
    idx = faiss.read_index(index_path)
    try:
        d = int(idx.d)
    except Exception:
        d = int(getattr(idx, "d", 0))
    ntotal = int(getattr(idx, "ntotal", 0))
    return d, ntotal

def read_sqlite(docstore_path: str) -> Tuple[int, int, int, Optional[int]]:
    con = sqlite3.connect(docstore_path)
    try:
        # quick check
        con.execute("PRAGMA quick_check;").fetchone()
        row = con.execute("SELECT COUNT(*), COALESCE(MIN(id), -1), COALESCE(MAX(id), -1) FROM chunks").fetchone()
        rows, min_id, max_id = (int(row[0]), int(row[1]), int(row[2]))
        # last fully saved ZIM entry id (from meta JSON)
        row2 = con.execute(
            "SELECT MAX(CAST(json_extract(meta,'$.entry_id') AS INTEGER)) FROM chunks"
        ).fetchone()
        max_entry_id = row2[0] if row2 and row2[0] is not None else None
    finally:
        con.close()
    return rows, min_id, max_id, max_entry_id

def trim_sqlite(docstore_path: str, keep_n: int) -> None:
    con = sqlite3.connect(docstore_path)
    try:
        with con:
            con.execute("DELETE FROM chunks WHERE id >= ?", (keep_n,))
        # optional vacuum to reclaim disk space (can be slow on huge DBs)
        # with con:
        #     con.execute("VACUUM;")
    finally:
        con.close()

def write_checkpoint(checkpoint_path: str, last_entry_id_seen: int, last_entry_id_persisted: int, next_id_persisted: int) -> None:
    tmp = checkpoint_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(
            {
                "last_entry_id_seen": int(last_entry_id_seen),
                "last_entry_id_persisted": int(last_entry_id_persisted),
                "next_id_persisted": int(next_id_persisted),
            },
            f,
        )
    os.replace(tmp, checkpoint_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="Path to FAISS index (e.g., faiss.index)")
    ap.add_argument("--docstore", required=True, help="Path to SQLite docstore (e.g., docstore.sqlite)")
    ap.add_argument("--checkpoint", required=True, help="Path to progress.json")
    ap.add_argument("--expect-dim", type=int, default=None, help="Expected FAISS dimension (e.g., 384 or 768)")
    ap.add_argument("--dry-run", action="store_true", help="Do not modify SQLite/checkpoint; just report")
    args = ap.parse_args()

    # 0) Basic existence checks
    for p in [args.index, args.docstore]:
        if not os.path.exists(p):
            print(f"[FATAL] Missing file: {p}", file=sys.stderr)
            sys.exit(1)

    # 1) Backups
    b1 = backup(args.index)
    b2 = backup(args.docstore)
    b3 = backup(args.checkpoint) if os.path.exists(args.checkpoint) else None
    print("[backup] FAISS →", b1)
    print("[backup] SQLite →", b2)
    print("[backup] Checkpoint →", b3 or "(none)")

    # 2) Read FAISS
    try:
        faiss_d, faiss_ntotal = read_faiss(args.index)
    except Exception as e:
        print(f"[FATAL] Could not read FAISS index: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[faiss] d={faiss_d}, ntotal={faiss_ntotal}")

    if args.expect_dim is not None:  # handle wrong hyphen? Fix below
        pass

    # Fix argparse attribute name (Python identifiers cannot contain '-')
    expect_dim = args.expect_dim if hasattr(args, "expect_dim") else None
    if expect_dim is None and hasattr(args, "expect-dim"):
        # defensive: if someone passed --expect-dim, argparse stored expect_dim
        expect_dim = getattr(args, "expect_dim")
    if expect_dim is not None and faiss_d != expect_dim:
        print(f"[GUARD] FAISS d={faiss_d} != --expect-dim={expect_dim}. Refusing to proceed.", file=sys.stderr)
        sys.exit(2)

    # 3) Read SQLite
    try:
        rows, min_id, max_id, max_entry_id_any = read_sqlite(args.docstore)
    except Exception as e:
        print(f"[FATAL] Could not read SQLite: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"[sqlite] rows={rows}, min_id={min_id}, max_id={max_id}, last_entry_id(any)={max_entry_id_any}")

    # If DB is empty
    if rows == 0 or max_id < 0:
        n_sqlite = 0
    else:
        n_sqlite = max_id + 1

    # 4) Compute persisted vector count N and last_entry_id among persisted rows only
    N = min(faiss_ntotal, n_sqlite)

    # If we need last_entry_id among only persisted IDs [0..N-1]
    last_entry_id_persisted = -1
    if N > 0:
        con = sqlite3.connect(args.docstore)
        try:
            row = con.execute(
                "SELECT MAX(CAST(json_extract(meta,'$.entry_id') AS INTEGER)) FROM chunks WHERE id < ?",
                (N,),
            ).fetchone()
            last_entry_id_persisted = row[0] if row and row[0] is not None else -1
        finally:
            con.close()

    print(f"[reconcile] N_faiss={faiss_ntotal}, N_sqlite={n_sqlite} → N_persisted={N}, last_entry_id_persisted={last_entry_id_persisted}")

    # 5) If SQLite is ahead, trim (unless dry-run)
    if n_sqlite > faiss_ntotal:
        print(f"[action] SQLite ahead of FAISS by {n_sqlite - faiss_ntotal} rows.")
        if args.dry_run:
            print("[dry-run] Would trim SQLite rows with id >= ", faiss_ntotal)
        else:
            trim_sqlite(args.docstore, faiss_ntotal)
            print("[done] Trimmed SQLite to match FAISS ntotal")
            N = faiss_ntotal  # after trim, N aligns to FAISS
    elif faiss_ntotal > n_sqlite:
        # Harmless case: extra vectors in FAISS without docstore rows
        print(f"[notice] FAISS has {faiss_ntotal - n_sqlite} more vectors than SQLite (dangling). Proceeding with N={N}.")

    # 6) Write coherent checkpoint
    last_entry_id_seen = last_entry_id_persisted  # conservative
    if args.dry_run:
        print("[dry-run] Would write checkpoint:")
        print(json.dumps({
            "last_entry_id_seen": last_entry_id_seen,
            "last_entry_id_persisted": last_entry_id_persisted,
            "next_id_persisted": N
        }, indent=2))
    else:
        write_checkpoint(args.checkpoint, last_entry_id_seen, last_entry_id_persisted, N)
        print(f"[checkpoint] Wrote {args.checkpoint}: seen={last_entry_id_seen}, persisted={last_entry_id_persisted}, next_id={N}")

    print("\n[ok] Repair complete. Re-run your ingester with matching EMBED_DIM/model.\n"
          "     Example (FAISS d == 384):\n"
          "       EMBED_BACKEND=hf EMBED_MODEL=snowflake/snowflake-arctic-embed-xs EMBED_DIM=384 python ingest.py\n"
          "     Example (FAISS d == 768):\n"
          "       EMBED_BACKEND=ollama EMBED_MODEL=nomic-embed-text EMBED_DIM=768 python ingest.py")

if __name__ == "__main__":
    main()
