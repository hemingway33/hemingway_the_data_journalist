import React from 'react';
import { Routes, Route } from 'react-router-dom';
import { Box, ThemeProvider, createTheme } from '@mui/material';
import { WebSocketProvider } from './contexts/WebSocketContext';
import Layout from './components/Layout';
import Dashboard from './pages/Dashboard';
import AgentManagement from './pages/AgentManagement';
import PortfolioMetrics from './pages/PortfolioMetrics';
import RiskManagement from './pages/RiskManagement';
import AgentCommunication from './pages/AgentCommunication';
import SystemMonitoring from './pages/SystemMonitoring';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
      light: '#42a5f5',
      dark: '#1565c0',
    },
    secondary: {
      main: '#dc004e',
      light: '#ff5983',
      dark: '#9a0036',
    },
    success: {
      main: '#2e7d32',
    },
    warning: {
      main: '#ed6c02',
    },
    error: {
      main: '#d32f2f',
    },
    background: {
      default: '#f5f7fa',
      paper: '#ffffff',
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h4: {
      fontWeight: 600,
    },
    h5: {
      fontWeight: 600,
    },
    h6: {
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0 2px 12px rgba(0,0,0,0.08)',
          borderRadius: 12,
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          borderRadius: 8,
          padding: '8px 24px',
        },
      },
    },
  },
});

function App() {
  return (
    <ThemeProvider theme={theme}>
      <WebSocketProvider>
        <Box sx={{ display: 'flex', minHeight: '100vh' }}>
          <Layout>
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/agents" element={<AgentManagement />} />
              <Route path="/portfolio" element={<PortfolioMetrics />} />
              <Route path="/risk" element={<RiskManagement />} />
              <Route path="/communication" element={<AgentCommunication />} />
              <Route path="/monitoring" element={<SystemMonitoring />} />
            </Routes>
          </Layout>
        </Box>
      </WebSocketProvider>
    </ThemeProvider>
  );
}

export default App; 