'use client';

import React, { useState, useRef, useEffect, useCallback } from 'react';
import {
  Box,
  Container,
  TextField,
  Button,
  Typography,
  Paper,
  Card,
  CardContent,
  CircularProgress,
  Chip,

  Alert,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  Avatar,
  Drawer,
  IconButton,
  ListItemButton,
  ListItemIcon,
} from '@mui/material';
import {
  Send,
  TrendingUp,
  Analytics,
  SmartToy,
  ExpandMore,
  AccountBalance,
  ShowChart,
  NewReleases,
  History,
  Add,
  Chat,
} from '@mui/icons-material';
import Navigation from '../../components/Navigation';
import ProtectedRoute from '../../components/ProtectedRoute';
import { useAuth } from '../../contexts/AuthContext';
import { useRouter } from 'next/navigation';

interface ChatMessage {
  id: string;
  message: string;
  sender: 'user' | 'ai';
  timestamp: Date;
  typing?: boolean;
}

interface Thread {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  lastMessage?: {
    content: string;
    createdAt: string;
    role: string;
  };
  messageCount: number;
}

interface BackendMessage {
  id: string;
  content: string;
  role: 'USER' | 'ASSISTANT' | 'SYSTEM';
  createdAt: string;
  reasoning?: string;
}

interface ResearchInsight {
  title: string;
  content: string;
  confidence: number;
  category: string;
  supporting_data: unknown[];
  sources: string[];
}

interface FinancialData {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  market_cap?: number;
  pe_ratio?: number;
  dividend_yield?: number;
  timestamp: string;
}

interface ResearchResponse {
  query: string;
  answer: string;
  insights: ResearchInsight[];
  financial_data: FinancialData[];
  sources: string[];
  timestamp: string;
  processing_time_ms: number;
  confidence_score: number;
}

const QUICK_ACTIONS = [
  { label: "Analyze AAPL", icon: <ShowChart />, query: "Analyze Apple stock" },
  { label: "Market Overview", icon: <Analytics />, query: "What's the current market outlook?" },
  { label: "Tech Stocks", icon: <TrendingUp />, query: "How are technology stocks performing?" },
  { label: "AI Sector", icon: <SmartToy />, query: "What's happening in the AI sector?" },
];

export default function ChatPage() {
  // Authentication
  const { user, token, isAuthenticated, isLoading: authLoading } = useAuth();
  const router = useRouter();

  // State management
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [streamingMessage, setStreamingMessage] = useState('');
  const [thinkingTrace, setThinkingTrace] = useState<string[]>([]);
  const [showThinking, setShowThinking] = useState(false);
  const [researchData, setResearchData] = useState<ResearchResponse | null>(null);
  const [threads, setThreads] = useState<Thread[]>([]);
  const [currentThread, setCurrentThread] = useState<Thread | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  // Redirect to login if not authenticated
  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      router.push('/login');
    }
  }, [authLoading, isAuthenticated, router]);

  // API functions
  const apiCall = useCallback(async (url: string, options: RequestInit = {}) => {
    if (!token) {
      throw new Error('No authentication token found');
    }

    const response = await fetch(`http://localhost:8000${url}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${token}`,
        ...options.headers,
      },
    });

    if (!response.ok) {
      throw new Error(`API call failed: ${response.statusText}`);
    }

    return response.json();
  }, [token]);

  const loadThreads = useCallback(async () => {
    try {
      const threadsData = await apiCall('/chat/threads');
      setThreads(threadsData);
    } catch (error) {
      // Error loading threads - handle silently
    }
  }, [apiCall]);

  const selectThread = useCallback(async (thread: Thread) => {
    setCurrentThread(thread);
    setLoadingHistory(true);
    try {
      const threadData = await apiCall(`/chat/threads/${thread.id}`);
      const backendMessages: BackendMessage[] = threadData.messages;
      
      // Convert backend messages to frontend format
      const convertedMessages: ChatMessage[] = backendMessages.map((msg) => ({
        id: msg.id,
        message: msg.content,
        sender: msg.role === 'USER' ? 'user' : 'ai',
        timestamp: new Date(msg.createdAt),
      }));

      setMessages(convertedMessages);
      setDrawerOpen(false);
    } catch (error) {
      // Error loading thread messages - handle silently
    } finally {
      setLoadingHistory(false);
    }
  }, [apiCall]);

  const createNewThread = useCallback(async () => {
    try {
      const newThread = await apiCall('/chat/threads', {
        method: 'POST',
        body: JSON.stringify({ title: 'New Chat' }),
      });
      
      setThreads(prev => [newThread, ...prev]);
      setCurrentThread(newThread);
      setMessages([{
        id: '1',
        message: "Hello! I'm your AI Financial Research Assistant. I can provide real-time stock analysis, market insights, and comprehensive financial research. What would you like to explore today?",
        sender: 'ai',
        timestamp: new Date(),
      }]);
      setDrawerOpen(false);
    } catch (error) {
      // Error creating new thread - handle silently
    }
  }, [apiCall]);

  const saveMessage = useCallback(async (threadId: string, content: string, role: 'USER' | 'ASSISTANT') => {
    try {
      await apiCall(`/chat/threads/${threadId}/messages`, {
        method: 'POST',
        body: JSON.stringify({
          content,
          role,
        }),
      });
    } catch (error) {
      // Error saving message - handle silently
    }
  }, [apiCall]);

  // Load threads when authenticated
  useEffect(() => {
    if (isAuthenticated && token) {
      loadThreads();
    }
  }, [isAuthenticated, token, loadThreads]);

  // Auto-select first thread if no current thread
  useEffect(() => {
    if (!currentThread && threads.length > 0) {
      selectThread(threads[0]);
    }
  }, [currentThread, threads, selectThread]);

  const handleSendMessage = async (messageText?: string) => {
    const messageToSend = messageText || input.trim();
    if (!messageToSend) return;

    // Create new thread if none exists
    let threadToUse = currentThread;
    if (!threadToUse) {
      try {
        threadToUse = await apiCall('/chat/threads', {
          method: 'POST',
          body: JSON.stringify({ title: messageToSend.length > 50 ? messageToSend.substring(0, 50) + '...' : messageToSend }),
        });
        setCurrentThread(threadToUse);
        setThreads(prev => [threadToUse!, ...prev]);
      } catch (error) {
        // Error creating thread - handle silently
        return;
      }
    }

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      message: messageToSend,
      sender: 'user',
      timestamp: new Date(),
    };

    const typingMessage: ChatMessage = {
      id: (Date.now() + 1).toString(),
      message: 'Analyzing your request...',
      sender: 'ai',
      timestamp: new Date(),
      typing: true,
    };

    setMessages(prev => [...prev, userMessage, typingMessage]);
    setInput('');
    setIsLoading(true);
    setStreamingMessage(''); // Reset streaming message
    setThinkingTrace([]); // Reset thinking trace for new message

    // Save user message to backend
    try {
      if (threadToUse) {
        await saveMessage(threadToUse.id, messageToSend, 'USER');
      }
    } catch (error) {
      console.error('Error saving user message:', error);
    }

    try {
      // Try streaming research first, fallback to regular research
      try {
        const streamingResponse = await fetch('http://localhost:9000/research/stream', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query: messageToSend,
            session_id: `session_${Date.now()}`,
            include_thinking: true
          }),
        });

        if (streamingResponse.ok) {
          // Handle streaming response
          const reader = streamingResponse.body?.getReader();
          const decoder = new TextDecoder();
          let finalData: ResearchResponse | null = null;

          if (reader) {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;

              const chunk = decoder.decode(value);
              const lines = chunk.split('\n');

              for (const line of lines) {
                if (line.startsWith('data: ')) {
                  try {
                    const data = JSON.parse(line.slice(6));
                    
                    if (data.type === 'thinking') {
                      // Add to thinking trace instead of updating main message
                      setThinkingTrace(prev => [...prev, data.content]);
                      // Update typing indicator
                      setMessages(prev => prev.map(msg => {
                        if (msg.typing) {
                          return { ...msg, message: '🤔 Thinking...' };
                        }
                        return msg;
                      }));
                    } else if (data.type === 'streaming_response') {
                      // Handle token-level streaming for the final response
                      setStreamingMessage(prev => prev + (data.content || ''));
                      setMessages(prev => prev.map(msg => {
                        if (msg.typing) {
                          return { ...msg, message: streamingMessage + (data.content || '') };
                        }
                        return msg;
                      }));
                    } else if (data.type === 'final_report') {
                      // Convert streaming format to ResearchResponse format
                      finalData = {
                        query: messageToSend,
                        answer: data.content,
                        insights: [],
                        financial_data: data.financial_data || [],
                        sources: data.citations || [],
                        timestamp: data.timestamp,
                        processing_time_ms: 0,
                        confidence_score: 0.85
                      };
                    }
                  } catch (e) {
                    // Error parsing streaming data - continue processing
                  }
                }
              }
            }
          }

          if (finalData) {
            setResearchData(finalData);
            // Remove typing message and add AI response
            setMessages(prev => {
              const filtered = prev.filter(msg => !msg.typing);
              return [...filtered, {
                id: (Date.now() + 2).toString(),
                message: finalData!.answer,
                sender: 'ai',
                timestamp: new Date(),
              }];
            });

            // Save AI response to backend
            try {
              if (threadToUse) {
                await saveMessage(threadToUse.id, finalData.answer, 'ASSISTANT');
              }
            } catch (error) {
              // Error saving AI message - handle silently
            }
          } else {
            throw new Error('No final data received from stream');
          }
        } else {
          throw new Error('Streaming not available');
        }
      } catch (streamError) {
        // Streaming failed, falling back to regular research
        
        // Fallback to regular research endpoint
        const response = await fetch('http://localhost:9000/research', {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            query: messageToSend,
            analysis_depth: 'standard',
            include_charts: false,
            timeframe: '1y'
          }),
        });

        if (!response.ok) {
          throw new Error('Failed to get AI response');
        }

        const data: ResearchResponse = await response.json();
        setResearchData(data);

        // Remove typing message and add AI response
        setMessages(prev => {
          const filtered = prev.filter(msg => !msg.typing);
          return [...filtered, {
            id: (Date.now() + 2).toString(),
            message: data.answer,
            sender: 'ai',
            timestamp: new Date(),
          }];
        });

        // Save AI response to backend
        try {
          if (threadToUse) {
            await saveMessage(threadToUse.id, data.answer, 'ASSISTANT');
          }
        } catch (error) {
          // Error saving AI message - handle silently
        }
      }

    } catch (error) {
      // Error calling AI service - show user-friendly message
      
      // Remove typing message and add error response
      setMessages(prev => {
        const filtered = prev.filter(msg => !msg.typing);
        return [...filtered, {
          id: (Date.now() + 2).toString(),
          message: "I apologize, but I'm having trouble connecting to the research service. Please ensure the AI service is running on port 9000 and try again.",
          sender: 'ai',
          timestamp: new Date(),
        }];
      });
    }

    setIsLoading(false);
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatPrice = (price: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(price);
  };

  const formatNumber = (num: number) => {
    return new Intl.NumberFormat('en-US').format(num);
  };

  // Show loading while checking authentication
  if (authLoading) {
    return (
      <Box
        display="flex"
        justifyContent="center"
        alignItems="center"
        minHeight="100vh"
      >
        <CircularProgress />
      </Box>
    );
  }

  // Don't render if not authenticated (will redirect)
  if (!isAuthenticated) {
    return null;
  }

  return (
    <ProtectedRoute requireAuth={true}>
      <Navigation />
      
      {/* Chat History Drawer */}
      <Drawer
        anchor="left"
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        sx={{
          width: 320,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: 320,
            boxSizing: 'border-box',
            mt: '64px', // Account for navigation height
          },
        }}
      >
        <Box sx={{ p: 2 }}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">Chat History</Typography>
            <Button
              variant="contained"
              size="small"
              startIcon={<Add />}
              onClick={createNewThread}
            >
              New Chat
            </Button>
          </Box>

          <List>
            {threads.map((thread) => (
              <ListItemButton
                key={thread.id}
                onClick={() => selectThread(thread)}
                selected={currentThread?.id === thread.id}
                sx={{ mb: 1, borderRadius: 1 }}
              >
                <ListItemIcon>
                  <Chat />
                </ListItemIcon>
                <ListItemText
                  primary={thread.title}
                  secondary={
                    thread.lastMessage
                      ? `${thread.lastMessage.content.substring(0, 50)}${thread.lastMessage.content.length > 50 ? '...' : ''}`
                      : 'No messages'
                  }
                  primaryTypographyProps={{
                    variant: 'body2',
                    fontWeight: currentThread?.id === thread.id ? 'bold' : 'normal',
                  }}
                  secondaryTypographyProps={{
                    variant: 'caption',
                  }}
                />
              </ListItemButton>
            ))}
          </List>

          {threads.length === 0 && (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography variant="body2" color="text.secondary">
                No chat history yet
              </Typography>
              <Button
                variant="outlined"
                size="small"
                startIcon={<Add />}
                onClick={createNewThread}
                sx={{ mt: 1 }}
              >
                Start First Chat
              </Button>
            </Box>
          )}
        </Box>
      </Drawer>

      <Container maxWidth="xl" sx={{ mt: 2, mb: 2 }}>
        <Box sx={{ display: 'flex', gap: 2, height: 'calc(100vh - 120px)' }}>
          
          {/* Main Chat Interface */}
          <Box sx={{ flex: '1 1 70%', minWidth: 0 }}>
            <Paper 
              elevation={2} 
              sx={{ 
                height: '100%', 
                display: 'flex', 
                flexDirection: 'column',
                borderRadius: 2,
              }}
            >
              {/* Chat Header */}
              <Box 
                sx={{ 
                  p: 2, 
                  borderBottom: '1px solid #e0e0e0',
                  bgcolor: 'primary.main',
                  color: 'white',
                  borderRadius: '8px 8px 0 0',
                }}
              >
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Box>
                    <Typography variant="h6" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <SmartToy />
                      AI Financial Research Assistant
                    </Typography>
                    <Typography variant="body2" sx={{ opacity: 0.9 }}>
                      {currentThread ? currentThread.title : 'Real-time analysis • Market insights • Investment research'}
                    </Typography>
                  </Box>
                  <IconButton
                    color="inherit"
                    onClick={() => setDrawerOpen(true)}
                    sx={{ opacity: 0.9 }}
                  >
                    <History />
                  </IconButton>
                </Box>
              </Box>

              {/* Quick Actions */}
              <Box sx={{ p: 2, borderBottom: '1px solid #e0e0e0' }}>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                  Quick Actions:
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                  {QUICK_ACTIONS.map((action, index) => (
                    <Chip
                      key={index}
                      icon={action.icon}
                      label={action.label}
                      variant="outlined"
                      clickable
                      size="small"
                      onClick={() => handleSendMessage(action.query)}
                      sx={{ 
                        '&:hover': { 
                          backgroundColor: 'primary.light', 
                          color: 'white' 
                        } 
                      }}
                    />
                  ))}
                </Box>
              </Box>

              {/* Messages Area */}
              <Box sx={{ flexGrow: 1, overflowY: 'auto', p: 2 }}>
                {loadingHistory ? (
                  <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100%' }}>
                    <CircularProgress />
                    <Typography sx={{ ml: 2 }}>Loading chat history...</Typography>
                  </Box>
                ) : messages.length === 0 ? (
                  <Box sx={{ display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center', height: '100%', color: 'text.secondary' }}>
                    <SmartToy sx={{ fontSize: 48, mb: 2, opacity: 0.3 }} />
                    <Typography variant="h6" sx={{ mb: 1 }}>
                      Welcome to AI Financial Research
                    </Typography>
                    <Typography variant="body1" align="center" sx={{ mb: 2 }}>
                      I can provide real-time stock analysis, market insights, and comprehensive financial research.
                    </Typography>
                    <Button
                      variant="outlined"
                      startIcon={<Add />}
                      onClick={createNewThread}
                    >
                      Start New Chat
                    </Button>
                  </Box>
                ) : (
                  messages.map((message) => (
                  <Box
                    key={message.id}
                    sx={{
                      display: 'flex',
                      justifyContent: message.sender === 'user' ? 'flex-end' : 'flex-start',
                      mb: 2,
                    }}
                  >
                    <Box sx={{ display: 'flex', alignItems: 'flex-start', maxWidth: '80%' }}>
                      {message.sender === 'ai' && (
                        <Avatar sx={{ mr: 1, bgcolor: 'primary.main', width: 32, height: 32 }}>
                          <SmartToy sx={{ fontSize: 20 }} />
                        </Avatar>
                      )}
                      
                      <Card
                        elevation={1}
                        sx={{
                          bgcolor: message.sender === 'user' ? 'primary.main' : 'grey.50',
                          color: message.sender === 'user' ? 'white' : 'text.primary',
                        }}
                      >
                        <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                          {message.typing ? (
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                              <CircularProgress size={16} />
                              <Typography variant="body2">
                                {message.message}
                              </Typography>
                            </Box>
                          ) : (
                            <Typography 
                              variant="body1" 
                              sx={{ 
                                whiteSpace: 'pre-wrap',
                                fontSize: '0.95rem',
                                lineHeight: 1.5,
                              }}
                            >
                              {message.message}
                            </Typography>
                          )}
                          
                          <Typography 
                            variant="caption" 
                            sx={{ 
                              display: 'block', 
                              mt: 1, 
                              opacity: 0.7,
                              fontSize: '0.75rem',
                            }}
                          >
                            {message.timestamp.toLocaleTimeString()}
                          </Typography>
                        </CardContent>
                      </Card>

                      {message.sender === 'user' && (
                        <Avatar sx={{ ml: 1, bgcolor: 'secondary.main', width: 32, height: 32 }}>
                          U
                        </Avatar>
                      )}
                    </Box>
                  </Box>
                )))}
                <div ref={messagesEndRef} />
              </Box>

              {/* Input Area */}
              <Box sx={{ p: 2, borderTop: '1px solid #e0e0e0' }}>
                <Box sx={{ display: 'flex', gap: 1 }}>
                  <TextField
                    fullWidth
                    multiline
                    maxRows={3}
                    placeholder="Ask about stocks, market trends, or financial analysis..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyPress={handleKeyPress}
                    disabled={isLoading}
                    variant="outlined"
                    size="small"
                  />
                  <Button
                    variant="contained"
                    endIcon={<Send />}
                    onClick={() => handleSendMessage()}
                    disabled={!input.trim() || isLoading}
                    sx={{ minWidth: 120 }}
                  >
                    {isLoading ? <CircularProgress size={20} /> : 'Send'}
                  </Button>
                </Box>
              </Box>
            </Paper>
          </Box>

          {/* Research Panel */}
          <Box sx={{ flex: '0 0 30%', minWidth: '300px' }}>
            <Paper elevation={2} sx={{ height: '100%', p: 2, borderRadius: 2 }}>
              <Typography variant="h6" sx={{ mb: 2, display: 'flex', alignItems: 'center', gap: 1 }}>
                <Analytics />
                Research Insights
              </Typography>

              {/* Thinking Trace Panel */}
              {thinkingTrace.length > 0 && (
                <Accordion sx={{ mb: 2 }}>
                  <AccordionSummary 
                    expandIcon={<ExpandMore />}
                    onClick={() => setShowThinking(!showThinking)}
                  >
                    <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                      <SmartToy />
                      AI Thinking Process ({thinkingTrace.length} steps)
                    </Typography>
                  </AccordionSummary>
                  <AccordionDetails>
                    <Box sx={{ maxHeight: 300, overflowY: 'auto' }}>
                      {thinkingTrace.map((thought, index) => (
                        <Alert 
                          key={index}
                          severity="info" 
                          variant="outlined"
                          sx={{ mb: 1, fontSize: '0.85rem' }}
                        >
                          <Typography variant="body2" sx={{ fontSize: '0.8rem', fontFamily: 'monospace' }}>
                            Step {index + 1}: {thought}
                          </Typography>
                        </Alert>
                      ))}
                    </Box>
                  </AccordionDetails>
                </Accordion>
              )}

              {researchData ? (
                <Box sx={{ height: 'calc(100% - 60px)', overflowY: 'auto' }}>
                  {/* Financial Data */}
                  {researchData.financial_data.length > 0 && (
                    <Accordion defaultExpanded sx={{ mb: 2 }}>
                      <AccordionSummary expandIcon={<ExpandMore />}>
                        <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <ShowChart />
                          Market Data
                        </Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        {researchData.financial_data.map((stock, index) => (
                          <Card key={index} variant="outlined" sx={{ mb: 1 }}>
                            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                <Typography variant="h6">{stock.symbol}</Typography>
                                <Typography 
                                  variant="h6" 
                                  color={stock.change >= 0 ? 'success.main' : 'error.main'}
                                >
                                  {formatPrice(stock.price)}
                                </Typography>
                              </Box>
                              
                              <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                                <Typography variant="body2">
                                  Change: {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(2)}
                                </Typography>
                                <Typography 
                                  variant="body2"
                                  color={stock.change_percent >= 0 ? 'success.main' : 'error.main'}
                                >
                                  {stock.change_percent >= 0 ? '+' : ''}{stock.change_percent.toFixed(1)}%
                                </Typography>
                              </Box>
                              
                              <Typography variant="body2" color="text.secondary">
                                Volume: {formatNumber(stock.volume)}
                              </Typography>
                              
                              {stock.pe_ratio && (
                                <Typography variant="body2" color="text.secondary">
                                  P/E: {stock.pe_ratio.toFixed(1)}
                                </Typography>
                              )}
                            </CardContent>
                          </Card>
                        ))}
                      </AccordionDetails>
                    </Accordion>
                  )}

                  {/* Research Insights */}
                  {researchData.insights.length > 0 && (
                    <Accordion defaultExpanded sx={{ mb: 2 }}>
                      <AccordionSummary expandIcon={<ExpandMore />}>
                        <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <TrendingUp />
                          Key Insights
                        </Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        {researchData.insights.map((insight, index) => (
                          <Alert 
                            key={index} 
                            severity={insight.confidence > 0.8 ? 'success' : insight.confidence > 0.6 ? 'warning' : 'info'}
                            sx={{ mb: 1 }}
                          >
                            <Typography variant="subtitle2" sx={{ fontWeight: 'bold' }}>
                              {insight.title}
                            </Typography>
                            <Typography variant="body2">
                              {insight.content}
                            </Typography>
                            <Typography variant="caption" sx={{ display: 'block', mt: 0.5 }}>
                              Confidence: {(insight.confidence * 100).toFixed(0)}% • {insight.category}
                            </Typography>
                          </Alert>
                        ))}
                      </AccordionDetails>
                    </Accordion>
                  )}

                  {/* Sources */}
                  {researchData.sources.length > 0 && (
                    <Accordion sx={{ mb: 2 }}>
                      <AccordionSummary expandIcon={<ExpandMore />}>
                        <Typography variant="subtitle1" sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <NewReleases />
                          Data Sources
                        </Typography>
                      </AccordionSummary>
                      <AccordionDetails>
                        <List dense>
                          {researchData.sources.map((source, index) => (
                            <ListItem key={index}>
                              <ListItemText primary={source} />
                            </ListItem>
                          ))}
                        </List>
                      </AccordionDetails>
                    </Accordion>
                  )}

                  {/* Analysis Metadata */}
                  <Card variant="outlined">
                    <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                      <Typography variant="subtitle2" sx={{ mb: 1 }}>
                        Analysis Details
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Processing Time: {researchData.processing_time_ms}ms
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Confidence Score: {(researchData.confidence_score * 100).toFixed(0)}%
                      </Typography>
                      <Typography variant="body2" color="text.secondary">
                        Generated: {new Date(researchData.timestamp).toLocaleString()}
                      </Typography>
                    </CardContent>
                  </Card>
                </Box>
              ) : (
                <Box 
                  sx={{ 
                    height: 'calc(100% - 60px)', 
                    display: 'flex', 
                    flexDirection: 'column',
                    justifyContent: 'center', 
                    alignItems: 'center',
                    color: 'text.secondary',
                  }}
                >
                  <AccountBalance sx={{ fontSize: 48, mb: 2, opacity: 0.3 }} />
                  <Typography variant="body1" align="center">
                    Start a conversation to see detailed research insights, market data, and analysis results here.
                  </Typography>
                </Box>
              )}
            </Paper>
          </Box>
        </Box>
      </Container>
    </ProtectedRoute>
  );
}