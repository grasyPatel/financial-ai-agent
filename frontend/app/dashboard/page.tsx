'use client';

import React from 'react';
import { Container, Typography, Paper, Box, Grid, Card, CardContent, Button } from '@mui/material';
import { TrendingUp, Assessment, Chat, Security } from '@mui/icons-material';
import { useRouter } from 'next/navigation';
import Navigation from '../../components/Navigation';
import ProtectedRoute from '../../components/ProtectedRoute';
import { useAuth } from '../../contexts/AuthContext';

export default function DashboardPage() {
  const { user } = useAuth();
  const router = useRouter();

  return (
    <ProtectedRoute requireAuth={true}>
      <Navigation />
      <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Dashboard
        </Typography>
        
        <Typography variant="h6" color="text.secondary" gutterBottom>
          Welcome back, {user?.name}! Ready to dive into financial research?
        </Typography>

        <Grid container spacing={3} sx={{ mt: 2 }}>
          {/* Quick Stats */}
          <Grid size={{ xs: 12, sm: 6, md: 3 }}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" gap={2}>
                  <Chat color="primary" />
                  <Box>
                    <Typography variant="h6">0</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Research Chats
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid size={{ xs: 12, sm: 6, md: 3 }}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" gap={2}>
                  <Assessment color="primary" />
                  <Box>
                    <Typography variant="h6">0</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Reports Generated
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid size={{ xs: 12, sm: 6, md: 3 }}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" gap={2}>
                  <TrendingUp color="primary" />
                  <Box>
                    <Typography variant="h6">0</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Analyses Completed
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          <Grid size={{ xs: 12, sm: 6, md: 3 }}>
            <Card>
              <CardContent>
                <Box display="flex" alignItems="center" gap={2}>
                  <Security color="primary" />
                  <Box>
                    <Typography variant="h6">Active</Typography>
                    <Typography variant="body2" color="text.secondary">
                      Account Status
                    </Typography>
                  </Box>
                </Box>
              </CardContent>
            </Card>
          </Grid>

          {/* Welcome Card */}
          <Grid size={{ xs: 12 }}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h5" gutterBottom>
                🚀 Getting Started with Deep Finance Research
              </Typography>
              
              <Typography variant="body1" paragraph>
                Your intelligent finance research assistant is ready! Here's what you can do:
              </Typography>

              <Box sx={{ ml: 2 }}>
                <Typography variant="body2" paragraph>
                  • <strong>Start Research Chats:</strong> Ask complex financial questions and get AI-powered insights
                </Typography>
                <Typography variant="body2" paragraph>
                  • <strong>Generate Reports:</strong> Create comprehensive analysis documents from your research
                </Typography>
                <Typography variant="body2" paragraph>
                  • <strong>Track Sources:</strong> All research is backed by verifiable data sources
                </Typography>
                <Typography variant="body2" paragraph>
                  • <strong>Save & Share:</strong> Keep your research organized and accessible
                </Typography>
              </Box>

              <Box sx={{ mt: 3, display: 'flex', gap: 2 }}>
                <Button 
                  variant="contained" 
                  size="large"
                  startIcon={<Chat />}
                  onClick={() => router.push('/chat')}
                  sx={{ 
                    bgcolor: 'primary.main',
                    '&:hover': { bgcolor: 'primary.dark' },
                    px: 4,
                    py: 1.5,
                  }}
                >
                  Start Research Chat
                </Button>
                
                <Button 
                  variant="outlined" 
                  size="large"
                  startIcon={<TrendingUp />}
                  onClick={() => router.push('/chat')}
                  sx={{ px: 4, py: 1.5 }}
                >
                  Market Analysis
                </Button>
              </Box>
            </Paper>
          </Grid>
        </Grid>
      </Container>
    </ProtectedRoute>
  );
}