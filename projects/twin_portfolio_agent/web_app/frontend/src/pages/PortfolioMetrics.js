import React, { useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip,
} from '@mui/material';
import {
  Optimize as OptimizeIcon,
  TrendingUp as TrendingUpIcon,
  Assessment as AssessmentIcon,
} from '@mui/icons-material';
import { useQuery, useMutation } from 'react-query';
import { useForm, Controller } from 'react-hook-form';
import { portfolioApi, agentApi, formatCurrency, formatPercentage } from '../services/api';
import { useWebSocket } from '../contexts/WebSocketContext';
import toast from 'react-hot-toast';

const OptimizationDialog = ({ open, onClose, userAgents }) => {
  const { control, handleSubmit, reset } = useForm({
    defaultValues: {
      user_id: '',
      target_roe: 0.15,
      max_risk_tolerance: 0.05,
      time_horizon: '6_months',
      regulatory_limits: true,
      liquidity_requirements: 0.15,
    },
  });

  const optimizeMutation = useMutation(portfolioApi.optimize, {
    onSuccess: () => {
      toast.success('Portfolio optimization completed');
      reset();
      onClose();
    },
    onError: (error) => {
      toast.error(`Optimization failed: ${error.response?.data?.detail || error.message}`);
    },
  });

  const onSubmit = (data) => {
    const optimizationRequest = {
      user_id: data.user_id,
      preferences: {
        target_roe: data.target_roe,
        max_risk_tolerance: data.max_risk_tolerance,
        time_horizon: data.time_horizon,
      },
      constraints: {
        regulatory_limits: data.regulatory_limits,
        liquidity_requirements: data.liquidity_requirements,
      },
    };
    optimizeMutation.mutate(optimizationRequest);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Request Portfolio Optimization</DialogTitle>
      <form onSubmit={handleSubmit(onSubmit)}>
        <DialogContent>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Controller
                name="user_id"
                control={control}
                rules={{ required: 'Please select a user agent' }}
                render={({ field, fieldState }) => (
                  <FormControl fullWidth error={!!fieldState.error}>
                    <InputLabel>User Agent</InputLabel>
                    <Select {...field} label="User Agent">
                      {Object.entries(userAgents).map(([userId, agent]) => (
                        <MenuItem key={userId} value={userId}>
                          {userId} ({agent.role})
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                )}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Controller
                name="target_roe"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Target ROE"
                    type="number"
                    fullWidth
                    inputProps={{ step: 0.01, min: 0, max: 1 }}
                  />
                )}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Controller
                name="max_risk_tolerance"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Max Risk Tolerance (VaR)"
                    type="number"
                    fullWidth
                    inputProps={{ step: 0.01, min: 0, max: 1 }}
                  />
                )}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Controller
                name="time_horizon"
                control={control}
                render={({ field }) => (
                  <FormControl fullWidth>
                    <InputLabel>Time Horizon</InputLabel>
                    <Select {...field} label="Time Horizon">
                      <MenuItem value="1_month">1 Month</MenuItem>
                      <MenuItem value="3_months">3 Months</MenuItem>
                      <MenuItem value="6_months">6 Months</MenuItem>
                      <MenuItem value="1_year">1 Year</MenuItem>
                    </Select>
                  </FormControl>
                )}
              />
            </Grid>
            <Grid item xs={12} sm={6}>
              <Controller
                name="liquidity_requirements"
                control={control}
                render={({ field }) => (
                  <TextField
                    {...field}
                    label="Liquidity Requirements"
                    type="number"
                    fullWidth
                    inputProps={{ step: 0.01, min: 0, max: 1 }}
                  />
                )}
              />
            </Grid>
          </Grid>
        </DialogContent>
        <DialogActions>
          <Button onClick={onClose}>Cancel</Button>
          <Button
            type="submit"
            variant="contained"
            disabled={optimizeMutation.isLoading}
          >
            {optimizeMutation.isLoading ? 'Optimizing...' : 'Optimize Portfolio'}
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  );
};

const PortfolioMetrics = () => {
  const [optimizeDialogOpen, setOptimizeDialogOpen] = useState(false);
  const { portfolioMetrics } = useWebSocket();

  const { data: metricsData } = useQuery('portfolioMetrics', portfolioApi.getMetrics);
  const { data: agentsData } = useQuery('agents', agentApi.listAgents);

  const metrics = portfolioMetrics || metricsData?.data;
  const userAgents = agentsData?.data?.agent_sessions || {};

  // Sample loan data for the table
  const sampleLoans = [
    { id: 'LOAN001', type: 'Mortgage', amount: 450000, rate: 3.25, status: 'Current', risk: 'Low' },
    { id: 'LOAN002', type: 'Auto', amount: 25000, rate: 4.5, status: 'Current', risk: 'Medium' },
    { id: 'LOAN003', type: 'Business', amount: 100000, rate: 6.75, status: 'Delinquent_30', risk: 'High' },
    { id: 'LOAN004', type: 'Consumer', amount: 15000, rate: 12.5, status: 'Current', risk: 'Medium' },
    { id: 'LOAN005', type: 'Credit Card', amount: 5000, rate: 18.5, status: 'Current', risk: 'High' },
  ];

  const getStatusColor = (status) => {
    switch (status) {
      case 'Current': return 'success';
      case 'Delinquent_30': return 'warning';
      case 'Delinquent_60': return 'error';
      case 'Default': return 'error';
      default: return 'default';
    }
  };

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'Low': return 'success';
      case 'Medium': return 'warning';
      case 'High': return 'error';
      default: return 'default';
    }
  };

  return (
    <Box className="fade-in">
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 600 }}>
          Portfolio Metrics & Optimization
        </Typography>
        <Button
          variant="contained"
          startIcon={<OptimizeIcon />}
          onClick={() => setOptimizeDialogOpen(true)}
          disabled={Object.keys(userAgents).length === 0}
        >
          Request Optimization
        </Button>
      </Box>

      {/* Portfolio Overview */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                Portfolio Composition
              </Typography>
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">Total Loans</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {metrics?.portfolio_composition?.total_loans || 0}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">Total Value</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {formatCurrency(metrics?.portfolio_composition?.total_value || 0)}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">Total Exposure</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600 }}>
                    {formatCurrency(metrics?.portfolio_composition?.total_exposure || 0)}
                  </Typography>
                </Box>
                <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
                  <Typography variant="body2">Expected Loss</Typography>
                  <Typography variant="body2" sx={{ fontWeight: 600, color: 'error.main' }}>
                    {formatCurrency(metrics?.portfolio_composition?.expected_loss || 0)}
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                Key Performance Indicators
              </Typography>
              <Grid container spacing={3}>
                <Grid item xs={6} sm={3}>
                  <Paper sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="h5" sx={{ fontWeight: 600, color: 'primary.main' }}>
                      {formatPercentage(metrics?.current_metrics?.roe || 0)}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Return on Equity
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Paper sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="h5" sx={{ fontWeight: 600, color: 'warning.main' }}>
                      {formatPercentage(metrics?.current_metrics?.var_95 || 0)}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      VaR 95%
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Paper sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="h5" sx={{ fontWeight: 600, color: 'error.main' }}>
                      {formatPercentage(metrics?.current_metrics?.expected_loss_rate || 0)}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Expected Loss Rate
                    </Typography>
                  </Paper>
                </Grid>
                <Grid item xs={6} sm={3}>
                  <Paper sx={{ p: 2, textAlign: 'center' }}>
                    <Typography variant="h5" sx={{ fontWeight: 600, color: 'secondary.main' }}>
                      {formatPercentage(metrics?.current_metrics?.delinquency_rate || 0)}
                    </Typography>
                    <Typography variant="caption" color="text.secondary">
                      Delinquency Rate
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Loan Portfolio Table */}
      <Card>
        <CardContent>
          <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
            Loan Portfolio Details
          </Typography>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Loan ID</TableCell>
                  <TableCell>Type</TableCell>
                  <TableCell align="right">Amount</TableCell>
                  <TableCell align="right">Rate</TableCell>
                  <TableCell>Status</TableCell>
                  <TableCell>Risk Level</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {sampleLoans.map((loan) => (
                  <TableRow key={loan.id}>
                    <TableCell>{loan.id}</TableCell>
                    <TableCell>{loan.type}</TableCell>
                    <TableCell align="right">{formatCurrency(loan.amount)}</TableCell>
                    <TableCell align="right">{formatPercentage(loan.rate / 100)}</TableCell>
                    <TableCell>
                      <Chip
                        label={loan.status}
                        color={getStatusColor(loan.status)}
                        size="small"
                      />
                    </TableCell>
                    <TableCell>
                      <Chip
                        label={loan.risk}
                        color={getRiskColor(loan.risk)}
                        size="small"
                        variant="outlined"
                      />
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </CardContent>
      </Card>

      <OptimizationDialog
        open={optimizeDialogOpen}
        onClose={() => setOptimizeDialogOpen(false)}
        userAgents={userAgents}
      />
    </Box>
  );
};

export default PortfolioMetrics; 