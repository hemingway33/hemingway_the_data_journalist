import React, { useState } from 'react';
import {
  Box,
  Drawer,
  AppBar,
  Toolbar,
  List,
  Typography,
  Divider,
  IconButton,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Badge,
  Chip,
  Avatar,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard as DashboardIcon,
  Psychology as PsychologyIcon,
  TrendingUp as TrendingUpIcon,
  Security as SecurityIcon,
  Chat as ChatIcon,
  MonitorHeart as MonitorHeartIcon,
  Notifications as NotificationsIcon,
  Settings as SettingsIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useWebSocket } from '../contexts/WebSocketContext';

const drawerWidth = 280;

const navigationItems = [
  { text: 'Dashboard', icon: <DashboardIcon />, path: '/' },
  { text: 'Agent Management', icon: <PsychologyIcon />, path: '/agents' },
  { text: 'Portfolio Metrics', icon: <TrendingUpIcon />, path: '/portfolio' },
  { text: 'Risk Management', icon: <SecurityIcon />, path: '/risk' },
  { text: 'Agent Communication', icon: <ChatIcon />, path: '/communication' },
  { text: 'System Monitoring', icon: <MonitorHeartIcon />, path: '/monitoring' },
];

const Layout = ({ children }) => {
  const [mobileOpen, setMobileOpen] = useState(false);
  const navigate = useNavigate();
  const location = useLocation();
  const { isConnected, systemStatus, alerts } = useWebSocket();

  const handleDrawerToggle = () => {
    setMobileOpen(!mobileOpen);
  };

  const getConnectionStatus = () => {
    if (isConnected) {
      return { color: 'success', text: 'Connected' };
    } else {
      return { color: 'error', text: 'Disconnected' };
    }
  };

  const drawer = (
    <Box>
      <Toolbar sx={{ px: 3, py: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Avatar sx={{ bgcolor: 'primary.main', width: 40, height: 40 }}>
            <PsychologyIcon />
          </Avatar>
          <Box>
            <Typography variant="h6" sx={{ fontWeight: 600, fontSize: '1.1rem' }}>
              Digital Twin
            </Typography>
            <Typography variant="caption" sx={{ color: 'text.secondary' }}>
              Portfolio AI
            </Typography>
          </Box>
        </Box>
      </Toolbar>
      
      <Divider />
      
      <Box sx={{ px: 2, py: 1 }}>
        <Chip
          size="small"
          label={getConnectionStatus().text}
          color={getConnectionStatus().color}
          variant="outlined"
          sx={{ fontSize: '0.75rem' }}
        />
      </Box>
      
      <List sx={{ px: 1 }}>
        {navigationItems.map((item) => {
          const isActive = location.pathname === item.path;
          
          return (
            <ListItem key={item.text} disablePadding sx={{ mb: 0.5 }}>
              <ListItemButton
                onClick={() => navigate(item.path)}
                sx={{
                  borderRadius: 2,
                  mx: 1,
                  bgcolor: isActive ? 'primary.main' : 'transparent',
                  color: isActive ? 'white' : 'text.primary',
                  '&:hover': {
                    bgcolor: isActive ? 'primary.dark' : 'action.hover',
                  },
                }}
              >
                <ListItemIcon sx={{ 
                  color: isActive ? 'white' : 'primary.main',
                  minWidth: 40,
                }}>
                  {item.text === 'Agent Communication' && alerts.length > 0 ? (
                    <Badge badgeContent={alerts.length} color="error">
                      {item.icon}
                    </Badge>
                  ) : (
                    item.icon
                  )}
                </ListItemIcon>
                <ListItemText 
                  primary={item.text}
                  primaryTypographyProps={{
                    fontSize: '0.9rem',
                    fontWeight: isActive ? 600 : 400,
                  }}
                />
              </ListItemButton>
            </ListItem>
          );
        })}
      </List>
      
      <Divider sx={{ my: 2 }} />
      
      {systemStatus && (
        <Box sx={{ px: 3, py: 2 }}>
          <Typography variant="subtitle2" sx={{ mb: 1, color: 'text.secondary' }}>
            System Status
          </Typography>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="caption">Agents</Typography>
              <Typography variant="caption" sx={{ fontWeight: 600 }}>
                {systemStatus.total_agents}
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="caption">Portfolio Value</Typography>
              <Typography variant="caption" sx={{ fontWeight: 600 }}>
                ${(systemStatus.portfolio_value / 1e6).toFixed(1)}M
              </Typography>
            </Box>
            <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
              <Typography variant="caption">ROE</Typography>
              <Typography 
                variant="caption" 
                sx={{ 
                  fontWeight: 600,
                  color: systemStatus.current_roe > 0 ? 'success.main' : 'error.main'
                }}
              >
                {(systemStatus.current_roe * 100).toFixed(2)}%
              </Typography>
            </Box>
          </Box>
        </Box>
      )}
    </Box>
  );

  return (
    <Box sx={{ display: 'flex' }}>
      <AppBar
        position="fixed"
        sx={{
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          ml: { sm: `${drawerWidth}px` },
          bgcolor: 'white',
          color: 'text.primary',
          boxShadow: '0 1px 3px rgba(0,0,0,0.12)',
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="open drawer"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { sm: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            Multi-Agent Portfolio Management
          </Typography>
          
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {alerts.length > 0 && (
              <Badge badgeContent={alerts.length} color="error">
                <NotificationsIcon color="action" />
              </Badge>
            )}
            
            <IconButton color="inherit">
              <SettingsIcon />
            </IconButton>
          </Box>
        </Toolbar>
      </AppBar>
      
      <Box
        component="nav"
        sx={{ width: { sm: drawerWidth }, flexShrink: { sm: 0 } }}
      >
        <Drawer
          variant="temporary"
          open={mobileOpen}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true,
          }}
          sx={{
            display: { xs: 'block', sm: 'none' },
            '& .MuiDrawer-paper': { boxSizing: 'border-box', width: drawerWidth },
          }}
        >
          {drawer}
        </Drawer>
        
        <Drawer
          variant="permanent"
          sx={{
            display: { xs: 'none', sm: 'block' },
            '& .MuiDrawer-paper': { 
              boxSizing: 'border-box', 
              width: drawerWidth,
              borderRight: '1px solid rgba(0, 0, 0, 0.08)',
            },
          }}
          open
        >
          {drawer}
        </Drawer>
      </Box>
      
      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: 3,
          width: { sm: `calc(100% - ${drawerWidth}px)` },
          mt: 8,
          bgcolor: 'background.default',
          minHeight: 'calc(100vh - 64px)',
        }}
      >
        {children}
      </Box>
    </Box>
  );
};

export default Layout; 