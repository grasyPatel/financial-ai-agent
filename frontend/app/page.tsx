'use client';

import { useState } from 'react';

export default function Home() {
  const [messages, setMessages] = useState<Array<{id: number, text: string, sender: 'user' | 'bot'}>>([
    { id: 1, text: "Hello! I'm your Deep Finance Research Assistant. Ask me anything about financial markets, companies, or economic trends!", sender: 'bot' }
  ]);
  const [input, setInput] = useState('');

  const sendMessage = () => {
    if (!input.trim()) return;
    
    const newMessage = { id: Date.now(), text: input, sender: 'user' as const };
    setMessages(prev => [...prev, newMessage]);
    
    // Simulate bot response (we'll connect this to real AI later)
    setTimeout(() => {
      const botResponse = { 
        id: Date.now() + 1, 
        text: `I received your message: "${input}". The AI backend will be connected in Phase 4!`, 
        sender: 'bot' as const 
      };
      setMessages(prev => [...prev, botResponse]);
    }, 1000);
    
    setInput('');
  };

  return (
    <div className="flex flex-col h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-blue-600 text-white p-4 shadow-lg">
        <h1 className="text-2xl font-bold">Deep Finance Research Chatbot</h1>
        <p className="text-blue-100">Phase 1: Basic Structure ✅</p>
      </header>

      {/* Chat Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div
              className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
                message.sender === 'user'
                  ? 'bg-blue-500 text-white'
                  : 'bg-white text-gray-800 border'
              }`}
            >
              {message.text}
            </div>
          </div>
        ))}
      </div>

      {/* Input Area */}
      <div className="border-t bg-white p-4">
        <div className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Ask about finance: 'Is HDFC Bank undervalued?'"
            className="flex-1 border rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={sendMessage}
            className="bg-blue-500 text-white px-6 py-2 rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
          >
            Send
          </button>
        </div>
        <p className="text-sm text-gray-500 mt-2">
          🚧 Backend connection coming in Phase 4. Currently showing demo responses.
        </p>
      </div>
    </div>
  );
}
