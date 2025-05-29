import React, { useEffect, useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  Chip,
  LinearProgress,
  Alert,
  Paper,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  TrendingUp,
  TrendingDown,
  Security,
  Psychology,
  PlayArrow,
  Refresh,
  Assessment,
} from '@mui/icons-material';
import { useQuery } from 'react-query';
import { Line, Doughnut } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip as ChartTooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { useWebSocket } from '../contexts/WebSocketContext';
import { systemApi, portfolioApi, simulationApi, formatCurrency, formatPercentage } from '../services/api';
import toast from 'react-hot-toast';

// Register Chart.js components
ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  ChartTooltip,
  Legend,
  ArcElement
);

const MetricCard = ({ title, value, change, icon, color = 'primary', format = 'number' }) => {
  const formatValue = (val) => {
    if (format === 'currency') return formatCurrency(val);
    if (format === 'percentage') return formatPercentage(val);
    return val?.toLocaleString();
  };

  const isPositive = change > 0;
  const changeColor = isPositive ? 'success.main' : 'error.main';
  const TrendIcon = isPositive ? TrendingUp : TrendingDown;

  return (
    <Card className="metric-card">
      <CardContent>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
          <Box>
            <Typography variant="h4" sx={{ fontWeight: 600, color: `${color}.main` }}>
              {formatValue(value)}
            </Typography>
            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
              {title}
            </Typography>
            {change !== undefined && (
              <Box sx={{ display: 'flex', alignItems: 'center', mt: 1 }}>
                <TrendIcon sx={{ color: changeColor, fontSize: 16, mr: 0.5 }} />
                <Typography variant="caption" sx={{ color: changeColor }}>
                  {formatPercentage(Math.abs(change))} {isPositive ? 'up' : 'down'}
                </Typography>
              </Box>
            )}
          </Box>
          <Box sx={{ color: `${color}.main`, opacity: 0.7 }}>
            {icon}
          </Box>
        </Box>
      </CardContent>
    </Card>
  );
};

const Dashboard = () => {
  const { systemStatus, portfolioMetrics, isConnected, requestMetricsUpdate } = useWebSocket();
  const [isSimulating, setIsSimulating] = useState(false);

  // Fetch portfolio metrics
  const { data: metricsData, refetch: refetchMetrics, isLoading: metricsLoading } = useQuery(
    'portfolioMetrics',
    portfolioApi.getMetrics,
    {
      refetchInterval: 30000, // Refetch every 30 seconds
      enabled: isConnected,
    }
  );

  // Fetch portfolio history
  const { data: historyData, refetch: refetchHistory } = useQuery(
    'portfolioHistory',
    portfolioApi.getHistory,
    {
      refetchInterval: 60000, // Refetch every minute
      enabled: isConnected,
    }
  );

  const handleSimulationStep = async () => {
    setIsSimulating(true);
    try {
      await simulationApi.step();
      toast.success('Simulation step completed');
      // Refetch data after simulation
      setTimeout(() => {
        refetchMetrics();
        refetchHistory();
      }, 1000);
    } catch (error) {
      toast.error('Failed to execute simulation step');
    } finally {
      setIsSimulating(false);
    }
  };

  const handleRefresh = () => {
    requestMetricsUpdate();
    refetchMetrics();
    refetchHistory();
    toast.success('Data refreshed');
  };

  const metrics = portfolioMetrics || metricsData?.data;
  const history = historyData?.data?.history || [];

  // Prepare chart data
  const performanceChartData = {
    labels: history.slice(-20).map((item, index) => `Day ${item.day || index + 1}`),
    datasets: [
      {
        label: 'Portfolio ROE',
        data: history.slice(-20).map(item => (item.roe || 0) * 100),
        borderColor: '#1976d2',
        backgroundColor: 'rgba(25, 118, 210, 0.1)',
        tension: 0.4,
      },
      {
        label: 'VaR 95%',
        data: history.slice(-20).map(item => (item.var_95 || 0) * 100),
        borderColor: '#dc004e',
        backgroundColor: 'rgba(220, 0, 78, 0.1)',
        tension: 0.4,
      },
    ],
  };

  const riskDistributionData = {
    labels: ['Conservative', 'Moderate', 'Aggressive'],
    datasets: [
      {
        data: [30, 50, 20], // Sample data
        backgroundColor: ['#4caf50', '#ff9800', '#f44336'],
        borderWidth: 0,
      },
    ],
  };

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
      },
    },
    scales: {
      y: {
        beginAtZero: true,
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
      },
      x: {
        grid: {
          color: 'rgba(0, 0, 0, 0.05)',
        },
      },
    },
  };

  if (metricsLoading) {
    return (
      <Box sx={{ width: '100%' }}>
        <LinearProgress />
        <Typography sx={{ mt: 2, textAlign: 'center' }}>
          Loading portfolio metrics...
        </Typography>
      </Box>
    );
  }

  return (
    <Box className="fade-in">
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 600 }}>
          Portfolio Dashboard
        </Typography>
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Tooltip title="Refresh Data">
            <IconButton onClick={handleRefresh} color="primary">
              <Refresh />
            </IconButton>
          </Tooltip>
          <Button
            variant="contained"
            startIcon={<PlayArrow />}
            onClick={handleSimulationStep}
            disabled={isSimulating || !isConnected}
          >
            {isSimulating ? 'Simulating...' : 'Run Simulation Step'}
          </Button>
        </Box>
      </Box>

      {!isConnected && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          WebSocket connection lost. Real-time updates are disabled.
        </Alert>
      )}

      {/* Key Metrics */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Portfolio Value"
            value={metrics?.current_metrics?.portfolio_value || systemStatus?.portfolio_value || 0}
            format="currency"
            icon={<Assessment />}
            color="primary"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Return on Equity"
            value={metrics?.current_metrics?.roe || systemStatus?.current_roe || 0}
            change={0.12} // Sample change
            format="percentage"
            icon={<TrendingUp />}
            color="success"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Value at Risk (95%)"
            value={metrics?.current_metrics?.var_95 || systemStatus?.var_95 || 0}
            change={-0.05} // Sample change
            format="percentage"
            icon={<Security />}
            color="warning"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <MetricCard
            title="Active Agents"
            value={systemStatus?.total_agents || 0}
            icon={<Psychology />}
            color="secondary"
          />
        </Grid>
      </Grid>

      {/* Charts Section */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                Performance Trends
              </Typography>
              <Box sx={{ height: 300 }}>
                {history.length > 0 ? (
                  <Line data={performanceChartData} options={chartOptions} />
                ) : (
                  <Box sx={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    justifyContent: 'center', 
                    height: '100%',
                    color: 'text.secondary'
                  }}>
                    No historical data available
                  </Box>
                )}
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                Risk Distribution
              </Typography>
              <Box sx={{ height: 300 }}>
                <Doughnut 
                  data={riskDistributionData} 
                  options={{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                      legend: {
                        position: 'bottom',
                      },
                    },
                  }}
                />
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Market Conditions */}
      {metrics?.market_conditions && (
        <Card>
          <CardContent>
            <Typography variant="h6" sx={{ mb: 3, fontWeight: 600 }}>
              Market Conditions
            </Typography>
            <Grid container spacing={3}>
              <Grid item xs={12} sm={6} md={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="h5" sx={{ fontWeight: 600, color: 'primary.main' }}>
                    {formatPercentage(metrics.market_conditions.base_interest_rate)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Base Interest Rate
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="h5" sx={{ fontWeight: 600, color: 'warning.main' }}>
                    {formatPercentage(metrics.market_conditions.unemployment_rate)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Unemployment Rate
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="h5" sx={{ fontWeight: 600, color: 'success.main' }}>
                    {formatPercentage(metrics.market_conditions.gdp_growth)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    GDP Growth
                  </Typography>
                </Paper>
              </Grid>
              <Grid item xs={12} sm={6} md={3}>
                <Paper sx={{ p: 2, textAlign: 'center' }}>
                  <Typography variant="h5" sx={{ fontWeight: 600, color: 'error.main' }}>
                    {metrics.market_conditions.volatility_index?.toFixed(1)}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Volatility Index
                  </Typography>
                </Paper>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      )}
    </Box>
  );
};

export default Dashboard; 