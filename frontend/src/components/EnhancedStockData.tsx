'use client';

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
  Alert,
  Divider,
} from '@mui/material';
import { 
  TrendingUp, 
  TrendingDown, 
  DollarSign, 
  BarChart3, 
  Activity,
  Target
} from 'lucide-react';

interface StockData {
  symbol: string;
  price: number;
  change: number;
  change_percent: number;
  volume: number;
  market_cap?: number;
  pe_ratio?: number;
  dividend_yield?: number;
  fifty_two_week_high?: number;
  fifty_two_week_low?: number;
  avg_volume?: number;
  beta?: number;
  sector?: string;
  industry?: string;
  company_name?: string;
}

interface EnhancedStockDataProps {
  symbol: string;
  onDataLoad?: (data: StockData) => void;
}

const EnhancedStockData: React.FC<EnhancedStockDataProps> = ({ symbol, onDataLoad }) => {
  const [stockData, setStockData] = useState<StockData | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchStockData = async (stockSymbol: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`http://localhost:9000/api/stock/${stockSymbol}/real`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch stock data: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      if (result.success) {
        setStockData(result.data);
        onDataLoad?.(result.data);
      } else {
        throw new Error('Failed to get stock data');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch stock data');
      console.error('Stock data error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (symbol) {
      fetchStockData(symbol);
      // Set up periodic refresh every 30 seconds
      const interval = setInterval(() => fetchStockData(symbol), 30000);
      return () => clearInterval(interval);
    }
  }, [symbol]);

  const formatCurrency = (value: number | undefined) => {
    if (value === undefined || value === null) return 'N/A';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatLargeNumber = (value: number | undefined) => {
    if (value === undefined || value === null) return 'N/A';
    
    if (value >= 1e12) return `$${(value / 1e12).toFixed(2)}T`;
    if (value >= 1e9) return `$${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `$${(value / 1e6).toFixed(2)}M`;
    return formatCurrency(value);
  };

  const formatVolume = (value: number | undefined) => {
    if (value === undefined || value === null) return 'N/A';
    
    if (value >= 1e9) return `${(value / 1e9).toFixed(2)}B`;
    if (value >= 1e6) return `${(value / 1e6).toFixed(2)}M`;
    if (value >= 1e3) return `${(value / 1e3).toFixed(2)}K`;
    return value.toLocaleString();
  };

  const formatPercentage = (value: number | undefined) => {
    if (value === undefined || value === null) return 'N/A';
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
            <Activity size={24} />
            <Typography variant="h6">Loading {symbol} Data...</Typography>
          </Box>
          <LinearProgress />
        </CardContent>
      </Card>
    );
  }

  if (error) {
    return (
      <Card>
        <CardContent>
          <Alert severity="error">
            {error}
          </Alert>
        </CardContent>
      </Card>
    );
  }

  if (!stockData) {
    return (
      <Card>
        <CardContent>
          <Typography variant="body2" color="text.secondary">
            No stock data available
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const isPositive = stockData.change >= 0;

  return (
    <Card>
      <CardContent>
        {/* Header */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'start', mb: 3 }}>
          <Box>
            <Typography variant="h5" fontWeight="bold">
              {stockData.symbol}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {stockData.company_name}
            </Typography>
            {stockData.sector && (
              <Chip 
                label={stockData.sector} 
                size="small" 
                variant="outlined" 
                sx={{ mt: 1 }}
              />
            )}
          </Box>
          <Box sx={{ textAlign: 'right' }}>
            <Typography variant="h4" fontWeight="bold">
              {formatCurrency(stockData.price)}
            </Typography>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              {isPositive ? (
                <TrendingUp size={20} color="green" />
              ) : (
                <TrendingDown size={20} color="red" />
              )}
              <Typography 
                variant="h6" 
                color={isPositive ? 'success.main' : 'error.main'}
              >
                {formatCurrency(stockData.change)} ({formatPercentage(stockData.change_percent)})
              </Typography>
            </Box>
          </Box>
        </Box>

        <Divider sx={{ my: 2 }} />

        {/* Key Metrics */}
        <Box sx={{ 
          display: 'grid', 
          gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
          gap: 2 
        }}>
          <Card variant="outlined">
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <BarChart3 size={16} />
                <Typography variant="caption" color="text.secondary">
                  Volume
                </Typography>
              </Box>
              <Typography variant="body1" fontWeight="bold">
                {formatVolume(stockData.volume)}
              </Typography>
              <Typography variant="caption" color="text.secondary">
                Avg: {formatVolume(stockData.avg_volume)}
              </Typography>
            </CardContent>
          </Card>

          <Card variant="outlined">
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <DollarSign size={16} />
                <Typography variant="caption" color="text.secondary">
                  Market Cap
                </Typography>
              </Box>
              <Typography variant="body1" fontWeight="bold">
                {formatLargeNumber(stockData.market_cap)}
              </Typography>
            </CardContent>
          </Card>

          <Card variant="outlined">
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Target size={16} />
                <Typography variant="caption" color="text.secondary">
                  P/E Ratio
                </Typography>
              </Box>
              <Typography variant="body1" fontWeight="bold">
                {stockData.pe_ratio?.toFixed(2) || 'N/A'}
              </Typography>
            </CardContent>
          </Card>

          <Card variant="outlined">
            <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
              <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                <Activity size={16} />
                <Typography variant="caption" color="text.secondary">
                  Beta
                </Typography>
              </Box>
              <Typography variant="body1" fontWeight="bold">
                {stockData.beta?.toFixed(3) || 'N/A'}
              </Typography>
            </CardContent>
          </Card>
        </Box>

        {/* 52-Week Range */}
        {stockData.fifty_two_week_high && stockData.fifty_two_week_low && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="subtitle2" gutterBottom>
              52-Week Range
            </Typography>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
              <Typography variant="body2" color="success.main">
                {formatCurrency(stockData.fifty_two_week_low)}
              </Typography>
              <Typography variant="body2" color="error.main">
                {formatCurrency(stockData.fifty_two_week_high)}
              </Typography>
            </Box>
            <Box sx={{ position: 'relative', height: 8, bgcolor: 'grey.200', borderRadius: 4, mb: 2 }}>
              {(() => {
                const range = stockData.fifty_two_week_high! - stockData.fifty_two_week_low!;
                const position = ((stockData.price - stockData.fifty_two_week_low!) / range) * 100;
                return (
                  <Box
                    sx={{
                      position: 'absolute',
                      left: `${Math.max(0, Math.min(100, position))}%`,
                      top: '50%',
                      transform: 'translate(-50%, -50%)',
                      width: 16,
                      height: 16,
                      bgcolor: 'primary.main',
                      borderRadius: '50%',
                      border: '2px solid white',
                      boxShadow: 1
                    }}
                  />
                );
              })()}
            </Box>
          </Box>
        )}

        {/* Additional Info */}
        {stockData.dividend_yield && stockData.dividend_yield > 0 && (
          <Box sx={{ mt: 2 }}>
            <Chip
              label={`Dividend Yield: ${stockData.dividend_yield.toFixed(2)}%`}
              color="primary"
              variant="outlined"
              size="small"
            />
          </Box>
        )}

        {/* Last Updated */}
        <Typography variant="caption" color="text.secondary" sx={{ display: 'block', mt: 2 }}>
          Last updated: {new Date().toLocaleTimeString()}
        </Typography>
      </CardContent>
    </Card>
  );
};

export default EnhancedStockData;