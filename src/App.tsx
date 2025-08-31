import { useState, useRef, useEffect } from 'react'
import { Send, Bot, User, RefreshCw, Settings, CheckCircle, AlertCircle } from 'lucide-react'
import './App.css'

interface Message {
  role: 'user' | 'assistant'
  content: string
  timestamp: Date
}

interface Model {
  id: string
  name: string
  description: string
}

interface Conversation {
  id: number
  title: string
  model: string
  created_at: string
  updated_at: string
  message_count: number
  last_message_time: string
}

const AVAILABLE_MODELS: Model[] = [
  { id: "gemma:2b", name: "Gemma 2B", description: "Google's lightweight language model" },
  { id: "deepseek-coder:6.7b (unfinished)", name: "DeepSeek Coder 6.7B", description: "Specialized for coding tasks" },
  { id: "llama2:7b (unfinished)", name: "Llama 2 7B", description: "Meta's open-source language model" },
  { id: "mistral:7b (unfinished)", name: "Mistral 7B", description: "High-performance 7B parameter model" }
]

function App() {
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState('')
  const [selectedModel, setSelectedModel] = useState<Model>(AVAILABLE_MODELS[0])
  const [isLoading, setIsLoading] = useState(false)
  const [isServerRunning, setIsServerRunning] = useState(false)
  const [showSidebar, setShowSidebar] = useState(false)
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [currentConversationId, setCurrentConversationId] = useState<number | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Check if Ollama server is running and load conversations
  useEffect(() => {
    checkServerStatus()
    loadConversations()
  }, [])

  const checkServerStatus = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/status')
      const data = await response.json()
      setIsServerRunning(data.running)
    } catch (error) {
      console.log('Server status check failed:', error)
      setIsServerRunning(false)
    }
  }

  const startOllamaServer = async () => {
    setIsLoading(true)
    try {
      const response = await fetch('http://localhost:5001/api/start', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      })
      const data = await response.json()
      setIsServerRunning(data.success)
    } catch (error) {
      console.error('Failed to start server:', error)
    } finally {
      setIsLoading(false)
    }
  }

  const loadConversations = async () => {
    try {
      const response = await fetch('http://localhost:5001/api/conversations')
      const data = await response.json()
      setConversations(data.conversations || [])
    } catch (error) {
      console.error('Failed to load conversations:', error)
    }
  }

  const loadConversation = async (conversationId: number) => {
    try {
      console.log('Loading conversation:', conversationId)
      const response = await fetch(`http://localhost:5001/api/conversations/${conversationId}`)
      const data = await response.json()
      console.log('Conversation data:', data)
      const messages = data.messages || []
      
      // Convert database messages to app format
      const appMessages: Message[] = messages.map((msg: any) => ({
        role: msg.role as 'user' | 'assistant',
        content: msg.content,
        timestamp: new Date(msg.timestamp)
      }))
      
      console.log('App messages:', appMessages)
      setMessages(appMessages)
      setCurrentConversationId(conversationId)
    } catch (error) {
      console.error('Failed to load conversation:', error)
    }
  }

  const deleteConversation = async (conversationId: number) => {
    try {
      await fetch(`http://localhost:5001/api/conversations/${conversationId}`, {
        method: 'DELETE',
      })
      
      // Remove from local state
      setConversations(prev => prev.filter(c => c.id !== conversationId))
      
      // Clear current conversation if it was deleted
      if (currentConversationId === conversationId) {
        setMessages([])
        setCurrentConversationId(null)
      }
    } catch (error) {
      console.error('Failed to delete conversation:', error)
    }
  }

  const sendMessage = async () => {
    if (!inputValue.trim() || isLoading) return

    const userMessage: Message = {
      role: 'user',
      content: inputValue,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue('')
    setIsLoading(true)

    try {
      // Call the backend API to get response from Ollama
      const response = await fetch('http://localhost:5001/api/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: selectedModel.id,
          prompt: inputValue,
          conversation_id: currentConversationId
        }),
      })
      
      const data = await response.json()
      
      if (data.error) {
        throw new Error(data.error)
      }
      
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.response,
        timestamp: new Date()
      }

      setMessages(prev => [...prev, assistantMessage])
      
      // Update conversation ID if this was a new conversation
      if (data.conversation_id && !currentConversationId) {
        setCurrentConversationId(data.conversation_id)
        // Reload conversations to show the new one
        loadConversations()
      }
    } catch (error) {
      const errorMessage: Message = {
        role: 'assistant',
        content: 'Sorry, I encountered an error. Please try again.',
        timestamp: new Date()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const clearChat = () => {
    setMessages([])
    setCurrentConversationId(null)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="header-left">
            <button 
              className="sidebar-toggle"
              onClick={() => setShowSidebar(!showSidebar)}
            >
              <Settings size={20} />
            </button>
            <h1>🤖 LLM Chat</h1>
          </div>
          <div className="header-right">
            <div className={`server-status ${isServerRunning ? 'running' : 'stopped'}`}>
              {isServerRunning ? (
                <>
                  <CheckCircle size={16} />
                  <span>Ollama Running</span>
                </>
              ) : (
                <>
                  <AlertCircle size={16} />
                  <span>Server Stopped</span>
                </>
              )}
            </div>
          </div>
        </div>
      </header>

      <div className="main-container">
        {/* Chat History Section */}
        <div className="chat-history-section">
          <h3>Chat History</h3>
          <div className="conversation-list">
            {conversations.length === 0 ? (
              <p className="no-conversations">No conversations yet</p>
            ) : (
              conversations.map((conversation) => (
                <div
                  key={conversation.id}
                  className={`conversation-item ${currentConversationId === conversation.id ? 'active' : ''}`}
                >
                  <button
                    className="conversation-button"
                    onClick={() => loadConversation(conversation.id)}
                  >
                    <div className="conversation-info">
                      <strong>{conversation.title}</strong>
                      <small>{conversation.model} • {conversation.message_count} messages</small>
                    </div>
                  </button>
                  <button
                    className="delete-conversation"
                    onClick={() => deleteConversation(conversation.id)}
                    title="Delete conversation"
                  >
                    ×
                  </button>
                </div>
              ))
            )}
          </div>
          
          {/* Model Selection at Bottom */}
          <div className="model-selection-bottom">
            <h3>Model Selection</h3>
            <div className="model-selector">
              <select
                className="model-dropdown"
                value={selectedModel.id}
                onChange={e => {
                  const model = AVAILABLE_MODELS.find(m => m.id === e.target.value)
                  if (model) setSelectedModel(model)
                }}
              >
                {AVAILABLE_MODELS.map(model => (
                  <option key={model.id} value={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>

        {/* Chat Area */}
        <main className="chat-area">
          <div className="messages-container">
            {messages.length === 0 ? (
              <div className="empty-state">
                <Bot size={48} />
                <h2>Welcome to LLM Chat!</h2>
                <p>Start a conversation with {selectedModel.name}</p>
              </div>
            ) : (
              messages.map((message, index) => (
                <div key={index} className={`message ${message.role}`}>
                  <div className="message-avatar">
                    {message.role === 'user' ? <User size={20} /> : <Bot size={20} />}
                  </div>
                  <div className="message-content">
                    <div className="message-text">{message.content}</div>
                    <div className="message-timestamp">
                      {message.timestamp.toLocaleTimeString()}
                    </div>
                  </div>
                </div>
              ))
            )}
            
            {isLoading && (
              <div className="message assistant">
                <div className="message-avatar">
                  <Bot size={20} />
                </div>
                <div className="message-content">
                  <div className="message-text">
                    <div className="typing-indicator">
                      <span></span>
                      <span></span>
                      <span></span>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="input-area">
            <div className="input-container">
              <textarea
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={`Ask ${selectedModel.name} anything...`}
                disabled={isLoading || !isServerRunning}
                rows={1}
                className="message-input"
              />
              <button
                onClick={sendMessage}
                disabled={!inputValue.trim() || isLoading || !isServerRunning}
                className="send-button"
              >
                <Send size={20} />
              </button>
            </div>
            <div className="input-footer">
              <small>
                Powered by Ollama • {selectedModel.name} • {new Date().toLocaleDateString()}
              </small>
            </div>
          </div>
        </main>
      </div>
    </div>
  )
}

export default App
