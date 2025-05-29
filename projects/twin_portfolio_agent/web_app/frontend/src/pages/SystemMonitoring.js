import React from 'react';
import { Box, Typography, Card, CardContent, Grid } from '@mui/material';
import { MonitorHeart as MonitorHeartIcon } from '@mui/icons-material';

const SystemMonitoring = () => {
  return (
    <Box className="fade-in">
      <Typography variant="h4" sx={{ fontWeight: 600, mb: 3 }}>
        System Monitoring
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent sx={{ textAlign: 'center', py: 8 }}>
              <MonitorHeartIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h5" sx={{ mb: 2 }}>
                System Monitoring Dashboard
              </Typography>
              <Typography variant="body1" color="text.secondary">
                Real-time system health monitoring and performance metrics coming soon.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default SystemMonitoring; 