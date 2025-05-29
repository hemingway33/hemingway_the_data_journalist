import React from 'react';
import { Box, Typography, Card, CardContent, Grid } from '@mui/material';
import { Security as SecurityIcon } from '@mui/icons-material';

const RiskManagement = () => {
  return (
    <Box className="fade-in">
      <Typography variant="h4" sx={{ fontWeight: 600, mb: 3 }}>
        Risk Management
      </Typography>
      
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent sx={{ textAlign: 'center', py: 8 }}>
              <SecurityIcon sx={{ fontSize: 64, color: 'text.secondary', mb: 2 }} />
              <Typography variant="h5" sx={{ mb: 2 }}>
                Risk Management Dashboard
              </Typography>
              <Typography variant="body1" color="text.secondary">
                Advanced risk analytics and stress testing features coming soon.
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export default RiskManagement; 