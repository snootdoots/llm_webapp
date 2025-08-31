from flask import Flask, request, jsonify
from flask_cors import CORS
import ollama
import subprocess
import socket
import time
import threading
import os
from database import db

app = Flask(__name__)
CORS(app)

# Configuration
OLLAMA_PORT = 11434
AVAILABLE_MODELS = ["gemma:2b", "deepseek-coder:6.7b", "llama2:7b", "mistral:7b"]

def is_port_in_use(port):
    """Check if a given port is already in use."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0

def ensure_ollama_running():
    """Start Ollama server if not already running."""
    if is_port_in_use(OLLAMA_PORT):
        return True
    else:
        try:
            # Check if ollama is installed
            subprocess.run(["ollama", "--version"], check=True, capture_output=True)
            
            process = subprocess.Popen(
                ["ollama", "serve"], 
                stdout=subprocess.DEVNULL, 
                stderr=subprocess.DEVNULL
            )
            time.sleep(3)  # Wait for server to boot
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

@app.route('/api/status', methods=['GET'])
def get_status():
    """Check if Ollama server is running."""
    is_running = is_port_in_use(OLLAMA_PORT)
    return jsonify({
        'running': is_running,
        'port': OLLAMA_PORT
    })

@app.route('/api/start', methods=['POST'])
def start_server():
    """Start the Ollama server."""
    success = ensure_ollama_running()
    return jsonify({
        'success': success,
        'message': 'Server started successfully' if success else 'Failed to start server'
    })

@app.route('/api/chat', methods=['POST'])
def chat():
    """Send a message to the selected LLM and get response."""
    data = request.json
    model = data.get('model', 'gemma:2b')
    prompt = data.get('prompt', '')
    conversation_id = data.get('conversation_id')
    
    if not prompt:
        return jsonify({'error': 'No prompt provided'}), 400
    
    try:
        # Ensure server is running
        if not is_port_in_use(OLLAMA_PORT):
            ensure_ollama_running()
        
        # Create new conversation if none exists
        if not conversation_id:
            title = prompt[:50] + "..." if len(prompt) > 50 else prompt
            conversation_id = db.create_conversation(title, model)
        
        # Add user message to database
        db.add_message(conversation_id, 'user', prompt)
        
        # Send request to Ollama
        response = ollama.generate(model=model, prompt=prompt)
        ai_response = response['response'].replace('*', '')
        
        # Add AI response to database
        db.add_message(conversation_id, 'assistant', ai_response)
        
        return jsonify({
            'response': ai_response,
            'model': model,
            'conversation_id': conversation_id
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def get_models():
    """Get available models."""
    try:
        # Try to get actual installed models from Ollama
        if is_port_in_use(OLLAMA_PORT):
            client = ollama.Client()
            installed_models = client.list()
            model_names = [model['name'] for model in installed_models['models']]
            return jsonify({
                'models': model_names if model_names else AVAILABLE_MODELS
            })
        else:
            return jsonify({
                'models': AVAILABLE_MODELS
            })
    except Exception:
        return jsonify({
            'models': AVAILABLE_MODELS
        })

@app.route('/api/conversations', methods=['GET'])
def get_conversations():
    """Get all conversations."""
    try:
        conversations = db.get_conversations()
        return jsonify({'conversations': conversations})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversations/<int:conversation_id>', methods=['GET'])
def get_conversation(conversation_id):
    """Get messages for a specific conversation."""
    try:
        messages = db.get_conversation_messages(conversation_id)
        return jsonify({'messages': messages})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversations/<int:conversation_id>', methods=['DELETE'])
def delete_conversation(conversation_id):
    """Delete a conversation."""
    try:
        db.delete_conversation(conversation_id)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/conversations/<int:conversation_id>/title', methods=['PUT'])
def update_conversation_title(conversation_id):
    """Update conversation title."""
    try:
        data = request.json
        title = data.get('title', '')
        if title:
            db.update_conversation_title(conversation_id, title)
            return jsonify({'success': True})
        else:
            return jsonify({'error': 'Title is required'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Try to start Ollama server on startup
    ensure_ollama_running()
    
    print("🤖 LLM Chat Backend API")
    print("Server running on http://localhost:5001")
    print("Available endpoints:")
    print("  GET  /api/status  - Check server status")
    print("  POST /api/start   - Start Ollama server")
    print("  POST /api/chat    - Send chat message")
    print("  GET  /api/models  - Get available models")
    
    app.run(debug=True, port=5001)
