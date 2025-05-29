import React, { useState } from 'react';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemAvatar,
  Avatar,
  Chip,
  Paper,
  Divider,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
} from '@mui/material';
import {
  Chat as ChatIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  Error as ErrorIcon,
  CheckCircle as CheckCircleIcon,
  Psychology as PsychologyIcon,
  Send as SendIcon,
  Refresh as RefreshIcon,
} from '@mui/icons-material';
import { useWebSocket } from '../contexts/WebSocketContext';
import { alertApi } from '../services/api';
import { format } from 'date-fns';
import toast from 'react-hot-toast';

const MessageItem = ({ message }) => {
  const getIcon = () => {
    switch (message.type) {
      case 'agent_created':
        return <PsychologyIcon />;
      case 'optimization':
        return <CheckCircleIcon />;
      case 'alert':
        return <WarningIcon />;
      case 'error':
        return <ErrorIcon />;
      default:
        return <InfoIcon />;
    }
  };

  const getColor = () => {
    switch (message.type) {
      case 'agent_created':
        return 'primary';
      case 'optimization':
        return 'success';
      case 'alert':
        return 'warning';
      case 'error':
        return 'error';
      default:
        return 'info';
    }
  };

  return (
    <ListItem sx={{ bgcolor: 'background.paper', mb: 1, borderRadius: 1 }}>
      <ListItemAvatar>
        <Avatar sx={{ bgcolor: `${getColor()}.main` }}>
          {getIcon()}
        </Avatar>
      </ListItemAvatar>
      <ListItemText
        primary={message.content}
        secondary={format(new Date(message.timestamp), 'MMM dd, yyyy HH:mm:ss')}
        secondaryTypographyProps={{ variant: 'caption' }}
      />
    </ListItem>
  );
};

const AlertItem = ({ alert }) => {
  const getSeverityIcon = () => {
    switch (alert.severity) {
      case 'high':
        return <ErrorIcon />;
      case 'medium':
        return <WarningIcon />;
      default:
        return <InfoIcon />;
    }
  };

  const getSeverityColor = () => {
    switch (alert.severity) {
      case 'high':
        return 'error';
      case 'medium':
        return 'warning';
      default:
        return 'info';
    }
  };

  return (
    <Paper sx={{ p: 2, mb: 2 }}>
      <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
        <Avatar sx={{ bgcolor: `${getSeverityColor()}.main` }}>
          {getSeverityIcon()}
        </Avatar>
        <Box sx={{ flexGrow: 1 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1 }}>
            <Typography variant="h6" sx={{ fontWeight: 600 }}>
              {alert.alert_type.replace('_', ' ').toUpperCase()}
            </Typography>
            <Chip 
              label={alert.severity} 
              color={getSeverityColor()} 
              size="small" 
            />
          </Box>
          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
            {alert.message}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {format(new Date(alert.timestamp), 'MMM dd, yyyy HH:mm:ss')}
          </Typography>
          
          {alert.recommended_actions && (
            <Box sx={{ mt: 2 }}>
              <Typography variant="caption" sx={{ fontWeight: 600 }}>
                Recommended Actions:
              </Typography>
              <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mt: 0.5 }}>
                {alert.recommended_actions.map((action, index) => (
                  <Chip 
                    key={index} 
                    label={action.replace('_', ' ')} 
                    size="small" 
                    variant="outlined" 
                  />
                ))}
              </Box>
            </Box>
          )}
        </Box>
      </Box>
    </Paper>
  );
};

const AgentCommunication = () => {
  const [alertType, setAlertType] = useState('interest_rate_increase');
  const [alertSeverity, setAlertSeverity] = useState('medium');
  const { alerts, agentMessages, isConnected } = useWebSocket();

  const simulateAlert = async () => {
    try {
      await alertApi.simulate({ alert_type: alertType, severity: alertSeverity });
      toast.success('Market alert simulated');
    } catch (error) {
      toast.error('Failed to simulate alert');
    }
  };

  const alertTypes = [
    'interest_rate_increase',
    'interest_rate_decrease',
    'market_volatility',
    'economic_downturn',
    'liquidity_crisis',
    'regulatory_change',
    'credit_spread_widening',
    'default_spike',
  ];

  return (
    <Box className="fade-in">
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" sx={{ fontWeight: 600 }}>
          Agent Communication & Alerts
        </Typography>
        <Chip
          icon={isConnected ? <CheckCircleIcon /> : <ErrorIcon />}
          label={isConnected ? 'Real-time Connected' : 'Disconnected'}
          color={isConnected ? 'success' : 'error'}
        />
      </Box>

      {!isConnected && (
        <Alert severity="warning" sx={{ mb: 3 }}>
          Real-time communication is disabled. Please check your connection.
        </Alert>
      )}

      <Grid container spacing={3}>
        {/* Agent Messages */}
        <Grid item xs={12} md={8}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                Agent Messages
              </Typography>
              
              {agentMessages.length === 0 ? (
                <Box sx={{ 
                  textAlign: 'center', 
                  py: 4,
                  color: 'text.secondary'
                }}>
                  <ChatIcon sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
                  <Typography variant="h6" sx={{ mb: 1 }}>
                    No Messages Yet
                  </Typography>
                  <Typography variant="body2">
                    Agent communications will appear here in real-time.
                  </Typography>
                </Box>
              ) : (
                <List className="scrollbar-thin" sx={{ maxHeight: 400, overflow: 'auto' }}>
                  {agentMessages
                    .slice()
                    .reverse()
                    .map((message) => (
                      <MessageItem key={message.id} message={message} />
                    ))}
                </List>
              )}
            </CardContent>
          </Card>
        </Grid>

        {/* Alert Simulator */}
        <Grid item xs={12} md={4}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                Alert Simulator
              </Typography>
              
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <FormControl fullWidth>
                  <InputLabel>Alert Type</InputLabel>
                  <Select
                    value={alertType}
                    label="Alert Type"
                    onChange={(e) => setAlertType(e.target.value)}
                  >
                    {alertTypes.map((type) => (
                      <MenuItem key={type} value={type}>
                        {type.replace(/_/g, ' ').toUpperCase()}
                      </MenuItem>
                    ))}
                  </Select>
                </FormControl>
                
                <FormControl fullWidth>
                  <InputLabel>Severity</InputLabel>
                  <Select
                    value={alertSeverity}
                    label="Severity"
                    onChange={(e) => setAlertSeverity(e.target.value)}
                  >
                    <MenuItem value="low">Low</MenuItem>
                    <MenuItem value="medium">Medium</MenuItem>
                    <MenuItem value="high">High</MenuItem>
                  </Select>
                </FormControl>
                
                <Button
                  variant="contained"
                  startIcon={<SendIcon />}
                  onClick={simulateAlert}
                  disabled={!isConnected}
                  fullWidth
                >
                  Simulate Alert
                </Button>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        {/* Market Alerts */}
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6" sx={{ mb: 2, fontWeight: 600 }}>
                Market Alerts
              </Typography>
              
              {alerts.length === 0 ? (
                <Box sx={{ 
                  textAlign: 'center', 
                  py: 4,
                  color: 'text.secondary'
                }}>
                  <WarningIcon sx={{ fontSize: 48, mb: 2, opacity: 0.5 }} />
                  <Typography variant="h6" sx={{ mb: 1 }}>
                    No Active Alerts
                  </Typography>
                  <Typography variant="body2">
                    Market alerts and system notifications will appear here.
                  </Typography>
                </Box>
              ) : (
                <Box className="scrollbar-thin" sx={{ maxHeight: 400, overflow: 'auto' }}>
                  {alerts.map((alert) => (
                    <AlertItem key={alert.id} alert={alert} />
                  ))}
                </Box>
              )}
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default AgentCommunication; 