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
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  IconButton,
  Avatar,
  List,
  ListItem,
  ListItemAvatar,
  ListItemText,
  ListItemSecondaryAction,
  Divider,
  Alert,
} from '@mui/material';
import {
  Add as AddIcon,
  Psychology as PsychologyIcon,
  TrendingUp as TrendingUpIcon,
  Security as SecurityIcon,
  Gavel as GavelIcon,
  Close as CloseIcon,
  Person as PersonIcon,
} from '@mui/icons-material';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { useForm, Controller } from 'react-hook-form';
import { agentApi } from '../services/api';
import { useWebSocket } from '../contexts/WebSocketContext';
import toast from 'react-hot-toast';

const AGENT_ROLES = [
  { value: 'portfolio_manager', label: 'Portfolio Manager', icon: <TrendingUpIcon />, color: 'primary' },
  { value: 'risk_manager', label: 'Risk Manager', icon: <SecurityIcon />, color: 'warning' },
  { value: 'credit_officer', label: 'Credit Officer', icon: <GavelIcon />, color: 'success' },
  { value: 'compliance_officer', label: 'Compliance Officer', icon: <PersonIcon />, color: 'info' },
];

const RISK_TOLERANCE_OPTIONS = [
  { value: 'conservative', label: 'Conservative' },
  { value: 'moderate', label: 'Moderate' },
  { value: 'aggressive', label: 'Aggressive' },
];

const METRICS_OPTIONS = [
  'ROE', 'VaR', 'Sharpe_ratio', 'expected_loss', 'delinquency_rate', 
  'concentration_risk', 'liquidity_ratio', 'capital_ratio'
];

const DASHBOARD_LAYOUTS = [
  { value: 'standard', label: 'Standard' },
  { value: 'executive', label: 'Executive' },
  { value: 'detailed', label: 'Detailed' },
  { value: 'risk_focused', label: 'Risk Focused' },
];

const ALERT_FREQUENCIES = [
  { value: 'real_time', label: 'Real Time' },
  { value: 'hourly', label: 'Hourly' },
  { value: 'daily', label: 'Daily' },
  { value: 'weekly', label: 'Weekly' },
];

const AgentCard = ({ agent, role }) => {
  const roleInfo = AGENT_ROLES.find(r => r.value === role);
  
  return (
    <Card className="agent-card">
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Avatar sx={{ bgcolor: `${roleInfo?.color}.main`, mr: 2 }}>
            {roleInfo?.icon}
          </Avatar>
          <Box>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              {agent.user_id}
            </Typography>
            <Typography variant="body2" color="text.secondary">
              {roleInfo?.label}
            </Typography>
          </Box>
        </Box>
        
        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1, mb: 2 }}>
          <Chip size="small" label={`Risk: ${agent.preferences?.risk_tolerance || 'N/A'}`} />
          <Chip size="small" label={`Layout: ${agent.preferences?.dashboard_layout || 'N/A'}`} />
          <Chip size="small" label={`Alerts: ${agent.preferences?.alert_frequency || 'N/A'}`} />
        </Box>
        
        <Typography variant="caption" color="text.secondary">
          Created: {new Date(agent.created_at).toLocaleString()}
        </Typography>
      </CardContent>
    </Card>
  );
};

const CreateAgentDialog = ({ open, onClose, onSuccess }) => {
  const { control, handleSubmit, reset, watch } = useForm({
    defaultValues: {
      user_id: '',
      role: '',
      risk_tolerance: 'moderate',
      preferred_metrics: ['ROE', 'VaR'],
      dashboard_layout: 'standard',
      alert_frequency: 'daily',
      communication_style: 'detailed',
      time_horizon: 'quarterly',
    },
  });

  const selectedRole = watch('role');
  const roleInfo = AGENT_ROLES.find(r => r.value === selectedRole);

  const createAgentMutation = useMutation(agentApi.createAgent, {
    onSuccess: () => {
      toast.success('Agent created successfully');
      reset();
      onSuccess();
      onClose();
    },
    onError: (error) => {
      toast.error(`Failed to create agent: ${error.response?.data?.detail || error.message}`);
    },
  });

  const onSubmit = (data) => {
    createAgentMutation.mutate(data);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          Create New Agent
          <IconButton onClick={onClose}>
            <CloseIcon />
          </IconButton>
        </Box>
      </DialogTitle>
      
      <form onSubmit={handleSubmit(onSubmit)}>
        <DialogContent>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6}>
              <Controller
                name="user_id"
                control={control}
                rules={{ required: 'User ID is required' }}
                render={({ field, fieldState }) => (
                  <TextField
                    {...field}
                    label="User ID"
                    fullWidth
                    error={!!fieldState.error}
                    helperText={fieldState.error?.message}
                  />
                )}
              />
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <Controller
                name="role"
                control={control}
                rules={{ required: 'Role is required' }}
                render={({ field, fieldState }) => (
                  <FormControl fullWidth error={!!fieldState.error}>
                    <InputLabel>Agent Role</InputLabel>
                    <Select {...field} label="Agent Role">
                      {AGENT_ROLES.map((role) => (
                        <MenuItem key={role.value} value={role.value}>
                          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                            {role.icon}
                            {role.label}
                          </Box>
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                )}
              />
            </Grid>

            {roleInfo && (
              <Grid item xs={12}>
                <Alert severity="info">
                  <Typography variant="body2">
                    <strong>{roleInfo.label}</strong>: This agent will specialize in {
                      roleInfo.value === 'portfolio_manager' ? 'portfolio optimization and performance tracking' :
                      roleInfo.value === 'risk_manager' ? 'risk assessment and mitigation strategies' :
                      roleInfo.value === 'credit_officer' ? 'credit decisions and loan approval processes' :
                      'regulatory compliance and reporting'
                    }.
                  </Typography>
                </Alert>
              </Grid>
            )}
            
            <Grid item xs={12} sm={6}>
              <Controller
                name="risk_tolerance"
                control={control}
                render={({ field }) => (
                  <FormControl fullWidth>
                    <InputLabel>Risk Tolerance</InputLabel>
                    <Select {...field} label="Risk Tolerance">
                      {RISK_TOLERANCE_OPTIONS.map((option) => (
                        <MenuItem key={option.value} value={option.value}>
                          {option.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                )}
              />
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <Controller
                name="dashboard_layout"
                control={control}
                render={({ field }) => (
                  <FormControl fullWidth>
                    <InputLabel>Dashboard Layout</InputLabel>
                    <Select {...field} label="Dashboard Layout">
                      {DASHBOARD_LAYOUTS.map((layout) => (
                        <MenuItem key={layout.value} value={layout.value}>
                          {layout.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                )}
              />
            </Grid>
            
            <Grid item xs={12} sm={6}>
              <Controller
                name="alert_frequency"
                control={control}
                render={({ field }) => (
                  <FormControl fullWidth>
                    <InputLabel>Alert Frequency</InputLabel>
                    <Select {...field} label="Alert Frequency">
                      {ALERT_FREQUENCIES.map((freq) => (
                        <MenuItem key={freq.value} value={freq.value}>
                          {freq.label}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
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
                      <MenuItem value="daily">Daily</MenuItem>
                      <MenuItem value="weekly">Weekly</MenuItem>
                      <MenuItem value="monthly">Monthly</MenuItem>
                      <MenuItem value="quarterly">Quarterly</MenuItem>
                      <MenuItem value="annual">Annual</MenuItem>
                    </Select>
                  </FormControl>
                )}
              />
            </Grid>
            
            <Grid item xs={12}>
              <Controller
                name="preferred_metrics"
                control={control}
                render={({ field }) => (
                  <FormControl fullWidth>
                    <InputLabel>Preferred Metrics</InputLabel>
                    <Select
                      {...field}
                      multiple
                      label="Preferred Metrics"
                      renderValue={(selected) => (
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {selected.map((value) => (
                            <Chip key={value} label={value} size="small" />
                          ))}
                        </Box>
                      )}
                    >
                      {METRICS_OPTIONS.map((metric) => (
                        <MenuItem key={metric} value={metric}>
                          {metric}
                        </MenuItem>
                      ))}
                    </Select>
                  </FormControl>
                )}
              />
            </Grid>
          </Grid>
        </DialogContent>
        
        <DialogActions sx={{ p: 3 }}>
          <Button onClick={onClose}>Cancel</Button>
          <Button 
            type="submit" 
            variant="contained"
            disabled={createAgentMutation.isLoading}
          >
            {createAgentMutation.isLoading ? 'Creating...' : 'Create Agent'}
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  );
};

const AgentManagement = () => {
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const queryClient = useQueryClient();
  const { systemStatus } = useWebSocket();

  // Fetch agents
  const { data: agentsData, refetch } = useQuery('agents', agentApi.listAgents, {
    refetchInterval: 30000,
  });

  const agents = agentsData?.data?.agent_sessions || {};
  const analytics = agentsData?.data?.analytics || {};

  const handleCreateSuccess = () => {
    refetch();
    queryClient.invalidateQueries('agents');
  };

  return (
    <Box className="fade-in">
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 600 }}>
          Agent Management
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setCreateDialogOpen(true)}
        >
          Create Agent
        </Button>
      </Box>

      {/* System Overview */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 600, color: 'primary.main' }}>
                {analytics.total_agents || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Total Agents
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 600, color: 'success.main' }}>
                {analytics.user_agents || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                User Agents
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 600, color: 'secondary.main' }}>
                {analytics.total_messages || 0}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Messages Exchanged
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent sx={{ textAlign: 'center' }}>
              <Typography variant="h3" sx={{ fontWeight: 600, color: 'warning.main' }}>
                1
              </Typography>
              <Typography variant="body2" color="text.secondary">
                Portfolio Agent
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Agent List */}
      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                Active User Agents
              </Typography>
              
              {Object.keys(agents).length === 0 ? (
                <Box sx={{ 
                  textAlign: 'center', 
                  py: 4,
                  color: 'text.secondary'
                }}>
                  <PsychologyIcon sx={{ fontSize: 64, mb: 2, opacity: 0.5 }} />
                  <Typography variant="h6" sx={{ mb: 1 }}>
                    No Agents Created Yet
                  </Typography>
                  <Typography variant="body2">
                    Create your first agent to start using the multi-agent system.
                  </Typography>
                </Box>
              ) : (
                <Grid container spacing={2}>
                  {Object.entries(agents).map(([userId, agent]) => (
                    <Grid item xs={12} sm={6} key={userId}>
                      <AgentCard agent={agent} role={agent.role} />
                    </Grid>
                  ))}
                </Grid>
              )}
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                Agent Capabilities
              </Typography>
              
              <List dense>
                {Object.entries(analytics.agent_capabilities || {}).map(([agentId, capabilities]) => (
                  <React.Fragment key={agentId}>
                    <ListItem>
                      <ListItemAvatar>
                        <Avatar sx={{ bgcolor: 'primary.main', width: 32, height: 32 }}>
                          <PsychologyIcon fontSize="small" />
                        </Avatar>
                      </ListItemAvatar>
                      <ListItemText
                        primary={agentId.replace('_', ' ').toUpperCase()}
                        secondary={`${capabilities?.length || 0} capabilities`}
                      />
                    </ListItem>
                    {capabilities && capabilities.length > 0 && (
                      <Box sx={{ pl: 7, pb: 1 }}>
                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                          {capabilities.slice(0, 3).map((capability) => (
                            <Chip
                              key={capability}
                              label={capability}
                              size="small"
                              variant="outlined"
                            />
                          ))}
                          {capabilities.length > 3 && (
                            <Chip
                              label={`+${capabilities.length - 3} more`}
                              size="small"
                              variant="outlined"
                              color="primary"
                            />
                          )}
                        </Box>
                      </Box>
                    )}
                    <Divider />
                  </React.Fragment>
                ))}
              </List>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <CreateAgentDialog
        open={createDialogOpen}
        onClose={() => setCreateDialogOpen(false)}
        onSuccess={handleCreateSuccess}
      />
    </Box>
  );
};

export default AgentManagement; 