import React, { useState, useEffect } from 'react';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import './App.css'; // 先创建这个CSS文件
import rehypeHighlight from 'rehype-highlight';
import 'highlight.js/styles/github.css'; // 或者选择其他主题

function App() {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError('');
    setResponse('');

    try {
      const result = await axios.post('/api/ask', {
        question: question.trim()
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
        withCredentials: false
      });
      
      setResponse(result.data.answer);
    } catch (err) {
      console.error('API Error:', err);
      setError(err.response?.data?.detail || err.message || 'An error occurred while processing your question');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <h1>RAG System - Question & Answer</h1>
      <p>Ask any question and get intelligent answers powered by RAG technology</p>
      
      <div className="chat-container">
        <form onSubmit={handleSubmit} className="input-group">
          <input
            type="text"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder="Enter your question here..."
            disabled={loading}
          />
          <button type="submit" disabled={loading}>
            {loading ? 'Thinking...' : 'Ask'}
          </button>
        </form>

        {error && (
          <div className="error">
            Error: {error}
          </div>
        )}

        <div className="response-container">
          {loading ? (
            <div className="loading">
              <div className="spinner"></div>
              <span>Processing your question...</span>
            </div>
          ) : response ? (
            <div className="markdown-response">
              <h3>Answer:</h3>
              <ReactMarkdown rehypePlugins={[rehypeHighlight]}>
                {response}
              </ReactMarkdown>
            </div>
          ) : (
            <p>Your answer will appear here...</p>
          )}
        </div>
      </div>
    </div>
  );
}

export default App;