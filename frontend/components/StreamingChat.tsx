'use client';

import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  TextField,
  Button,
  Typography,
  Card,
  CardContent,
  Chip,
  LinearProgress,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  List,
  ListItem,
  ListItemText,
  IconButton,
  Tooltip,
  Alert,
  Divider
} from '@mui/material';
import {
  Send as SendIcon,
  ExpandMore as ExpandMoreIcon,
  Link as LinkIcon,
  Download as DownloadIcon,
  Visibility as ThinkingIcon,
  Assessment as ReportIcon,
  Timeline as StreamIcon
} from '@mui/icons-material';
import ReactMarkdown from 'react-markdown';

interface Message {
  id: string;
  type: 'user' | 'assistant' | 'thinking' | 'error';
  content: string;
  timestamp: Date;
  sources?: Source[];
  financialData?: FinancialData[];
  citations?: string[];
  thinkingSteps?: string[];
  isStreaming?: boolean;
}

interface Source {
  url: string;
  title: string;
  snippet: string;
  domain: string;
  relevance_score?: number;
}

interface FinancialData {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  market_cap?: number;
  pe_ratio?: number;
  company_name?: string;
}

const StreamingChat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showThinking, setShowThinking] = useState(true);
  const [sessionId] = useState(() => `session_${Date.now()}`);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: `user_${Date.now()}`,
      type: 'user',
      content: input,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      // Create assistant message placeholder
      const assistantMessageId = `assistant_${Date.now()}`;
      const assistantMessage: Message = {
        id: assistantMessageId,
        type: 'assistant',
        content: '',
        timestamp: new Date(),
        thinkingSteps: [],
        isStreaming: true
      };

      setMessages(prev => [...prev, assistantMessage]);

      // Start streaming request
      const response = await fetch('http://localhost:9000/research/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: input,
          session_id: sessionId,
          include_thinking: showThinking
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();

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
                
                setMessages(prev => prev.map(msg => {
                  if (msg.id === assistantMessageId) {
                    const updatedMessage = { ...msg };
                    
                    if (data.type === 'thinking') {
                      updatedMessage.thinkingSteps = [
                        ...(updatedMessage.thinkingSteps || []),
                        data.content
                      ];
                    } else if (data.type === 'final_report') {
                      updatedMessage.content = data.content;
                      updatedMessage.sources = data.sources;
                      updatedMessage.financialData = data.financial_data;
                      updatedMessage.citations = data.citations;
                      updatedMessage.isStreaming = false;
                    } else if (data.type === 'error') {
                      updatedMessage.type = 'error';
                      updatedMessage.content = data.content;
                      updatedMessage.isStreaming = false;
                    }
                    
                    return updatedMessage;
                  }
                  return msg;
                }));
              } catch (e) {
                console.error('Error parsing streaming data:', e);
              }
            }
          }
        }
      }
    } catch (error) {
      console.error('Streaming error:', error);
      setMessages(prev => prev.map(msg => {
        if (msg.type === 'assistant' && msg.isStreaming) {
          return {
            ...msg,
            type: 'error' as const,
            content: `Error: ${error instanceof Error ? error.message : 'Unknown error occurred'}`,
            isStreaming: false
          };
        }
        return msg;
      }));
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const exportReport = (message: Message) => {
    const report = `# Financial Research Report

**Query:** ${messages.find(m => m.type === 'user' && m.timestamp < message.timestamp)?.content || 'N/A'}
**Generated:** ${message.timestamp.toLocaleString()}

## Analysis

${message.content}

## Sources

${message.citations?.map((citation, i) => `${i + 1}. ${citation}`).join('\n') || 'No sources available'}

## Financial Data

${message.financialData?.map(data => 
  `- **${data.symbol}** (${data.company_name || data.symbol}): $${data.price} (${data.change_percent > 0 ? '+' : ''}${data.change_percent.toFixed(2)}%)`
).join('\n') || 'No financial data available'}
`;

    const blob = new Blob([report], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `financial-report-${Date.now()}.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const renderMessage = (message: Message) => {
    if (message.type === 'user') {
      return (
        <Box key={message.id} sx={{ display: 'flex', justifyContent: 'flex-end', mb: 2 }}>
          <Paper sx={{ p: 2, maxWidth: '70%', bgcolor: 'primary.main', color: 'white' }}>
            <Typography>{message.content}</Typography>
          </Paper>
        </Box>
      );
    }

    return (
      <Box key={message.id} sx={{ display: 'flex', justifyContent: 'flex-start', mb: 2 }}>
        <Paper sx={{ p: 2, maxWidth: '90%', width: '100%' }}>
          {/* Thinking Steps */}
          {message.thinkingSteps && message.thinkingSteps.length > 0 && (
            <Accordion sx={{ mb: 2 }}>
              <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                  <ThinkingIcon fontSize="small" />
                  <Typography variant="body2">
                    Show Thinking Process ({message.thinkingSteps.length} steps)
                  </Typography>
                  {message.isStreaming && <LinearProgress sx={{ flexGrow: 1, ml: 2 }} />}
                </Box>
              </AccordionSummary>
              <AccordionDetails>
                <List dense>
                  {message.thinkingSteps.map((step, i) => (
                    <ListItem key={i}>
                      <ListItemText 
                        primary={step}
                        primaryTypographyProps={{ variant: 'body2' }}
                      />
                    </ListItem>
                  ))}
                </List>
              </AccordionDetails>
            </Accordion>
          )}

          {/* Main Content */}
          {message.type === 'error' ? (
            <Alert severity="error">
              {message.content}
            </Alert>
          ) : (
            <Box>
              {message.isStreaming && !message.content ? (
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                  <StreamIcon />
                  <Typography>Researching and analyzing...</Typography>
                  <LinearProgress sx={{ flexGrow: 1 }} />
                </Box>
              ) : (
                <ReactMarkdown>{message.content}</ReactMarkdown>
              )}
            </Box>
          )}

          {/* Financial Data */}
          {message.financialData && message.financialData.length > 0 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="h6" gutterBottom>📊 Financial Data</Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                {message.financialData.map(data => (
                  <Card key={data.symbol} variant="outlined" sx={{ minWidth: 200 }}>
                    <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                      <Typography variant="h6">{data.symbol}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        {data.company_name}
                      </Typography>
                      <Typography variant="h5">
                        ${data.price}
                      </Typography>
                      <Chip
                        label={`${data.change_percent > 0 ? '+' : ''}${data.change_percent.toFixed(2)}%`}
                        color={data.change_percent > 0 ? 'success' : 'error'}
                        size="small"
                      />
                    </CardContent>
                  </Card>
                ))}
              </Box>
            </Box>
          )}

          {/* Sources */}
          {message.sources && message.sources.length > 0 && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="h6" gutterBottom>📚 Sources</Typography>
              <List dense>
                {message.sources.map((source, i) => (
                  <ListItem key={i} sx={{ pl: 0 }}>
                    <ListItemText
                      primary={
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                          <Typography variant="body2" sx={{ fontWeight: 'bold' }}>
                            {source.title}
                          </Typography>
                          <Tooltip title="Open source">
                            <IconButton size="small" onClick={() => window.open(source.url, '_blank')}>
                              <LinkIcon fontSize="small" />
                            </IconButton>
                          </Tooltip>
                        </Box>
                      }
                      secondary={
                        <Box>
                          <Typography variant="body2" color="text.secondary">
                            {source.snippet}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            {source.domain}
                          </Typography>
                        </Box>
                      }
                    />
                  </ListItem>
                ))}
              </List>
            </Box>
          )}

          {/* Export Button */}
          {message.content && !message.isStreaming && (
            <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end' }}>
              <Tooltip title="Export Report">
                <IconButton onClick={() => exportReport(message)} size="small">
                  <DownloadIcon />
                </IconButton>
              </Tooltip>
            </Box>
          )}
        </Paper>
      </Box>
    );
  };

  return (
    <Box sx={{ height: '100vh', display: 'flex', flexDirection: 'column' }}>
      {/* Header */}
      <Paper sx={{ p: 2, borderRadius: 0 }}>
        <Typography variant="h5" sx={{ fontWeight: 'bold' }}>
          🔍 Deep Finance Research Chatbot
        </Typography>
        <Typography variant="body2" color="text.secondary">
          Advanced AI-powered financial research with real-time streaming
        </Typography>
      </Paper>

      {/* Messages */}
      <Box sx={{ flexGrow: 1, overflow: 'auto', p: 2 }}>
        {messages.length === 0 && (
          <Box sx={{ textAlign: 'center', mt: 4 }}>
            <ReportIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
            <Typography variant="h6" color="text.secondary" gutterBottom>
              Welcome to Deep Finance Research
            </Typography>
            <Typography variant="body2" color="text.secondary">
              Ask me about stocks, market trends, company analysis, or any financial topic.
              I'll research the web and provide comprehensive analysis with live data.
            </Typography>
            <Box sx={{ mt: 2, display: 'flex', gap: 1, justifyContent: 'center', flexWrap: 'wrap' }}>
              {[
                "Compare Apple vs Tesla AI strategy",
                "HDFC Bank quarterly performance analysis",
                "Indian jewelry sector investment outlook",
                "Netflix streaming market analysis"
              ].map((suggestion, i) => (
                <Chip
                  key={i}
                  label={suggestion}
                  variant="outlined"
                  onClick={() => setInput(suggestion)}
                  sx={{ cursor: 'pointer' }}
                />
              ))}
            </Box>
          </Box>
        )}
        
        {messages.map(renderMessage)}
        <div ref={messagesEndRef} />
      </Box>

      {/* Input */}
      <Paper sx={{ p: 2, borderRadius: 0 }}>
        <Box sx={{ display: 'flex', gap: 2, alignItems: 'flex-end' }}>
          <TextField
            fullWidth
            multiline
            maxRows={4}
            placeholder="Ask me about financial markets, stocks, or investment analysis..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isLoading}
            variant="outlined"
          />
          <Button
            variant="contained"
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            sx={{ minWidth: '100px' }}
            endIcon={<SendIcon />}
          >
            {isLoading ? 'Researching...' : 'Send'}
          </Button>
        </Box>
      </Paper>
    </Box>
  );
};

export default StreamingChat;