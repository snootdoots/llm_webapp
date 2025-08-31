# LLM Chat React App

A modern React web application for chatting with various LLM models powered by Ollama.

## Features

- 🤖 Chat with multiple LLM models (Gemma 2B, DeepSeek Coder, Llama 2, Mistral)
- 💬 Real-time chat interface with typing indicators
- 🎨 Modern, responsive UI with glassmorphism design
- ⚙️ Model selection and server management
- 📱 Mobile-responsive design
- 🔄 Chat history management

## Prerequisites

- Node.js (v16 or higher)
- Python 3.8 or higher
- Ollama installed on your system ([Install Ollama](https://ollama.ai/))

## Installation

### Frontend (React)

1. Navigate to the React directory:
   ```bash
   cd react
   ```

2. Install dependencies:
   ```bash
   npm install --legacy-peer-deps
   ```

### Backend (Python Flask)

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Application

### 1. Start the Backend

In the `react` directory, run:
```bash
python backend.py
```

The backend will start on `http://localhost:5000` and automatically attempt to start the Ollama server.

### 2. Start the Frontend

In a new terminal, in the `react` directory, run:
```bash
npm run dev
```

The React app will start on `http://localhost:5173`.

### 3. Access the Application

Open your browser and navigate to `http://localhost:5173`.

## Usage

1. **Model Selection**: Click the settings icon to open the sidebar and select your preferred LLM model
2. **Start Server**: If Ollama isn't running, click "Start Ollama Server" in the sidebar
3. **Chat**: Type your message and press Enter or click the send button
4. **Clear Chat**: Use the "Clear Chat" button to start a new conversation

## API Endpoints

The backend provides the following endpoints:

- `GET /api/status` - Check if Ollama server is running
- `POST /api/start` - Start the Ollama server
- `POST /api/chat` - Send a chat message to the selected model
- `GET /api/models` - Get available models

## Available Models

- **Gemma 2B**: Google's lightweight language model
- **DeepSeek Coder 6.7B**: Specialized for coding tasks
- **Llama 2 7B**: Meta's open-source language model
- **Mistral 7B**: High-performance 7B parameter model

## Troubleshooting

### Ollama Not Found
If you get an error about Ollama not being found:
1. Install Ollama from [https://ollama.ai/](https://ollama.ai/)
2. Pull the models you want to use: `ollama pull gemma:2b`

### Server Connection Issues
- Make sure the backend is running on port 5000
- Check that Ollama is installed and accessible
- Verify that the models are downloaded: `ollama list`

### CORS Issues
The backend includes CORS headers, but if you encounter issues, make sure both frontend and backend are running on their respective ports.

## Development

### Frontend Structure
- `src/App.tsx` - Main application component
- `src/App.css` - Styling for the chat interface
- `package.json` - Dependencies and scripts

### Backend Structure
- `backend.py` - Flask API server
- `requirements.txt` - Python dependencies

## License

This project is for educational purposes.
