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
  TextField,
  Button,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Paper,
  Divider,
} from '@mui/material';
import { TrendingUp, TrendingDown, Wallet, Plus, DollarSign } from 'lucide-react';

interface PortfolioPosition {
  symbol: string;
  shares: number;
  avg_cost: number;
  current_price: number;
  market_value: number;
  unrealized_pnl: number;
  unrealized_pnl_percent: number;
  day_change: number;
  day_change_percent: number;
  weight: number;
}

interface PortfolioSummary {
  total_value: number;
  total_cost: number;
  total_pnl: number;
  total_pnl_percent: number;
  day_change: number;
  day_change_percent: number;
  cash_balance: number;
  positions: PortfolioPosition[];
}

const PortfolioTracker: React.FC = () => {
  const [portfolio, setPortfolio] = useState<PortfolioSummary | null>(null);
  const [symbols, setSymbols] = useState('AAPL,MSFT,GOOGL,TSLA');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchPortfolio = async (symbolList: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(
        `http://localhost:9000/api/portfolio/analysis?symbols=${encodeURIComponent(symbolList)}`
      );
      
      if (!response.ok) {
        throw new Error(`Failed to fetch portfolio: ${response.statusText}`);
      }
      
      const result = await response.json();
      
      if (result.success) {
        setPortfolio(result.data);
      } else {
        throw new Error('Failed to get portfolio data');
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch portfolio');
      console.error('Portfolio error:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchPortfolio(symbols);
  }, []);

  const handleAnalyze = () => {
    fetchPortfolio(symbols);
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
    }).format(value);
  };

  const formatPercentage = (value: number) => {
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value.toFixed(2)}%`;
  };

  const getColorForPercentage = (value: number) => {
    return value >= 0 ? 'success.main' : 'error.main';
  };

  if (loading) {
    return (
      <Card>
        <CardContent>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
            <Wallet size={24} />
            <Typography variant="h6">Analyzing Portfolio...</Typography>
          </Box>
          <LinearProgress />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 3 }}>
          <Wallet size={24} />
          <Typography variant="h6">Portfolio Analysis</Typography>
        </Box>

        {/* Portfolio Input */}
        <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
          <TextField
            fullWidth
            size="small"
            label="Stock symbols (comma-separated)"
            value={symbols}
            onChange={(e) => setSymbols(e.target.value)}
            placeholder="AAPL,MSFT,GOOGL,TSLA"
            helperText="Enter up to 10 stock symbols separated by commas"
          />
          <Button 
            variant="contained" 
            onClick={handleAnalyze}
            disabled={loading || !symbols.trim()}
            startIcon={<Plus size={16} />}
          >
            Analyze
          </Button>
        </Box>

        {error && (
          <Alert severity="error" sx={{ mb: 3 }}>
            {error}
          </Alert>
        )}

        {portfolio && (
          <>
            {/* Portfolio Summary */}
            <Box sx={{ mb: 3 }}>
              <Typography variant="subtitle1" gutterBottom fontWeight="bold">
                Portfolio Summary
              </Typography>
              
              <Box sx={{ 
                display: 'grid', 
                gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
                gap: 2, 
                mb: 2 
              }}>
                <Card variant="outlined">
                  <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
                      <DollarSign size={16} />
                      <Typography variant="caption" color="text.secondary">
                        Total Value
                      </Typography>
                    </Box>
                    <Typography variant="h6">
                      {formatCurrency(portfolio.total_value)}
                    </Typography>
                  </CardContent>
                </Card>

                <Card variant="outlined">
                  <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                    <Typography variant="caption" color="text.secondary">
                      Total P&L
                    </Typography>
                    <Typography 
                      variant="h6" 
                      color={getColorForPercentage(portfolio.total_pnl)}
                    >
                      {formatCurrency(portfolio.total_pnl)}
                    </Typography>
                    <Typography 
                      variant="caption" 
                      color={getColorForPercentage(portfolio.total_pnl_percent)}
                    >
                      {formatPercentage(portfolio.total_pnl_percent)}
                    </Typography>
                  </CardContent>
                </Card>

                <Card variant="outlined">
                  <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                    <Typography variant="caption" color="text.secondary">
                      Total Cost
                    </Typography>
                    <Typography variant="h6">
                      {formatCurrency(portfolio.total_cost)}
                    </Typography>
                  </CardContent>
                </Card>

                <Card variant="outlined">
                  <CardContent sx={{ p: 2, '&:last-child': { pb: 2 } }}>
                    <Typography variant="caption" color="text.secondary">
                      Cash Balance
                    </Typography>
                    <Typography variant="h6">
                      {formatCurrency(portfolio.cash_balance)}
                    </Typography>
                  </CardContent>
                </Card>
              </Box>
            </Box>

            <Divider sx={{ my: 3 }} />

            {/* Positions Table */}
            {portfolio.positions.length > 0 ? (
              <TableContainer component={Paper} variant="outlined">
                <Table size="small">
                  <TableHead>
                    <TableRow>
                      <TableCell>Symbol</TableCell>
                      <TableCell align="right">Shares</TableCell>
                      <TableCell align="right">Avg Cost</TableCell>
                      <TableCell align="right">Current Price</TableCell>
                      <TableCell align="right">Market Value</TableCell>
                      <TableCell align="right">P&L</TableCell>
                      <TableCell align="right">Day Change</TableCell>
                      <TableCell align="right">Weight</TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {portfolio.positions.map((position) => (
                      <TableRow key={position.symbol} hover>
                        <TableCell>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Typography variant="body2" fontWeight="bold">
                              {position.symbol}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell align="right">
                          {position.shares.toLocaleString()}
                        </TableCell>
                        <TableCell align="right">
                          {formatCurrency(position.avg_cost)}
                        </TableCell>
                        <TableCell align="right">
                          <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'flex-end', gap: 1 }}>
                            {formatCurrency(position.current_price)}
                            {position.day_change_percent >= 0 ? (
                              <TrendingUp size={12} color="green" />
                            ) : (
                              <TrendingDown size={12} color="red" />
                            )}
                          </Box>
                        </TableCell>
                        <TableCell align="right">
                          {formatCurrency(position.market_value)}
                        </TableCell>
                        <TableCell align="right">
                          <Box>
                            <Typography 
                              variant="body2" 
                              color={getColorForPercentage(position.unrealized_pnl)}
                            >
                              {formatCurrency(position.unrealized_pnl)}
                            </Typography>
                            <Typography 
                              variant="caption" 
                              color={getColorForPercentage(position.unrealized_pnl_percent)}
                            >
                              {formatPercentage(position.unrealized_pnl_percent)}
                            </Typography>
                          </Box>
                        </TableCell>
                        <TableCell align="right">
                          <Chip
                            label={formatPercentage(position.day_change_percent)}
                            size="small"
                            color={position.day_change_percent >= 0 ? 'success' : 'error'}
                            variant="outlined"
                          />
                        </TableCell>
                        <TableCell align="right">
                          {position.weight.toFixed(1)}%
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            ) : (
              <Alert severity="info">
                No positions found. Try adding some stock symbols.
              </Alert>
            )}
          </>
        )}
      </CardContent>
    </Card>
  );
};

export default PortfolioTracker;