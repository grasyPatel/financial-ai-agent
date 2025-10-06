'use client';

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Box,
  Chip,
  LinearProgress,
  Alert,
} from '@mui/material';
import { TrendingUp, TrendingDown, BarChart3 } from 'lucide-react';

interface PriceData {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface TechnicalIndicators {
  sma_20?: number;
  sma_50?: number;
  support_level?: number;
  resistance_level?: number;
  rsi?: number;
  macd?: number;
}

interface ChartData {
  symbol: string;
  timeframe: string;
  prices: PriceData[];
  technical_indicators: TechnicalIndicators;
  volume_data: Array<{ date: string; volume: number }>;
}

interface FinancialChartProps {
  symbol: string;
}

const FinancialChart: React.FC<FinancialChartProps> = ({ symbol }) => {
  const [chartData, setChartData] = useState<ChartData | null>(null);
  const [timeframe, setTimeframe] = useState('1y');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchChartData = async (newSymbol: string, newTimeframe: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(
        `http://localhost:9000/api/stock/${newSymbol}/chart?timeframe=${newTimeframe}`
      );
      
      if (!response.ok) {
        throw new Error(`Failed to fetch chart data: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      if (result.success) {
        setChartData(result.data);
      } else {
        throw new Error('Failed to get chart data');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch chart data');
      console.error('Chart data error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (symbol) {
      fetchChartData(symbol, timeframe);
    }
  }, [symbol, timeframe]);

  const getPriceDirection = (prices: PriceData[]) => {
    if (!prices || prices.length < 2) return 'neutral';
    const latest = prices[prices.length - 1];
    const previous = prices[prices.length - 2];
    return latest.close > previous.close ? 'up' : 'down';
  };

  const formatCurrency = (value: number | undefined) => {
    if (value === undefined || value === null) return 'N/A';
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercentage = (value: number | undefined) => {
    if (value === undefined || value === null) return 'N/A';
    return `${value.toFixed(2)}%`;
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
            <BarChart3 size={24} />
            <Typography variant="h6">Loading Chart Data...</Typography>
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

  if (!chartData) {
    return (
      <Card>
        <CardContent>
          <Typography variant="body2" color="text.secondary">
            No chart data available
          </Typography>
        </CardContent>
      </Card>
    );
  }

  const direction = getPriceDirection(chartData.prices);
  const latestPrice = chartData.prices[chartData.prices.length - 1];

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            <BarChart3 size={24} />
            <Typography variant="h6">
              {chartData.symbol} Price Chart
            </Typography>
            <Chip 
              icon={direction === 'up' ? <TrendingUp size={16} /> : <TrendingDown size={16} />}
              label={direction === 'up' ? 'Bullish' : 'Bearish'}
              color={direction === 'up' ? 'success' : 'error'}
              size="small"
            />
          </Box>
          
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Timeframe</InputLabel>
            <Select
              value={timeframe}
              label="Timeframe"
              onChange={(e) => setTimeframe(e.target.value)}
            >
              <MenuItem value="1w">1 Week</MenuItem>
              <MenuItem value="1m">1 Month</MenuItem>
              <MenuItem value="3m">3 Months</MenuItem>
              <MenuItem value="6m">6 Months</MenuItem>
              <MenuItem value="1y">1 Year</MenuItem>
              <MenuItem value="2y">2 Years</MenuItem>
            </Select>
          </FormControl>
        </Box>

        {/* Price Summary */}
        <Box sx={{ mb: 3, p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
          <Typography variant="subtitle2" gutterBottom>
            Latest Price Data
          </Typography>
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(120px, 1fr))', gap: 2 }}>
            <Box>
              <Typography variant="caption" color="text.secondary">Close</Typography>
              <Typography variant="body2" fontWeight="bold">
                {formatCurrency(latestPrice?.close)}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">High</Typography>
              <Typography variant="body2">{formatCurrency(latestPrice?.high)}</Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">Low</Typography>
              <Typography variant="body2">{formatCurrency(latestPrice?.low)}</Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">Open</Typography>
              <Typography variant="body2">{formatCurrency(latestPrice?.open)}</Typography>
            </Box>
          </Box>
        </Box>

        {/* Technical Indicators */}
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle2" gutterBottom>
            Technical Indicators
          </Typography>
          <Box sx={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))', gap: 2 }}>
            <Box>
              <Typography variant="caption" color="text.secondary">SMA (20)</Typography>
              <Typography variant="body2">
                {formatCurrency(chartData.technical_indicators.sma_20)}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">SMA (50)</Typography>
              <Typography variant="body2">
                {formatCurrency(chartData.technical_indicators.sma_50)}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">Support</Typography>
              <Typography variant="body2" color="success.main">
                {formatCurrency(chartData.technical_indicators.support_level)}
              </Typography>
            </Box>
            <Box>
              <Typography variant="caption" color="text.secondary">Resistance</Typography>
              <Typography variant="body2" color="error.main">
                {formatCurrency(chartData.technical_indicators.resistance_level)}
              </Typography>
            </Box>
          </Box>
        </Box>

        {/* Simple Price Chart Representation */}
        <Box sx={{ mb: 2 }}>
          <Typography variant="subtitle2" gutterBottom>
            Price Movement ({chartData.timeframe})
          </Typography>
          <Box sx={{ height: 200, bgcolor: 'grey.50', p: 2, borderRadius: 1 }}>
            <Typography variant="body2" color="text.secondary" align="center" sx={{ mt: 8 }}>
              📊 Chart visualization would go here
            </Typography>
            <Typography variant="caption" color="text.secondary" align="center" display="block" sx={{ mt: 2 }}>
              Data points: {chartData.prices.length} • Latest: {new Date(latestPrice.date).toLocaleDateString()}
            </Typography>
          </Box>
        </Box>

        {/* Chart Stats */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', mt: 2 }}>
          <Typography variant="caption" color="text.secondary">
            Period: {chartData.timeframe}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {chartData.prices.length} data points
          </Typography>
        </Box>
      </CardContent>
    </Card>
  );
};

export default FinancialChart;