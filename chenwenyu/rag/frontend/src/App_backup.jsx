import React, { useState, useEffect } from 'react';
import axios from 'axios';

function App() {
  const [question, setQuestion] = useState('');
  const [response, setResponse] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    console.log('App component mounted');
    // 自动测试API连接
    testAPIConnection();
  }, []);

  const testAPIConnection = async () => {
    console.log('Testing API connection...');
    try {
      const result = await axios.get('/api/health');
      console.log('API connection successful:', result.data);
    } catch (err) {
      console.error('API connection failed:', err);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError('');
    setResponse('');

    try {
      console.log('Sending question:', question);
      const result = await axios.post('/api/ask', {
        question: question.trim()
      }, {
        headers: {
          'Content-Type': 'application/json',
        },
        withCredentials: false
      });
      
      console.log('Received response:', result.data);
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
            <div>
              <h3>Answer:</h3>
              <p>{response}</p>
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