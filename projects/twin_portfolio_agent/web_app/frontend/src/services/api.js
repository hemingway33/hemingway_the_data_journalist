import axios from 'axios';

const BASE_URL = process.env.NODE_ENV === 'production' 
  ? '/api' 
  : 'http://localhost:8000/api';

const api = axios.create({
  baseURL: BASE_URL,
  timeout: 10000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    console.log(`Making ${config.method?.toUpperCase()} request to ${config.url}`);
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor
api.interceptors.response.use(
  (response) => {
    return response;
  },
  (error) => {
    console.error('API Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

// System APIs
export const systemApi = {
  getStatus: () => api.get('/system/status'),
  getHealth: () => api.get('/health'),
};

// Agent APIs
export const agentApi = {
  createAgent: (preferences) => api.post('/agents/create', preferences),
  listAgents: () => api.get('/agents'),
};

// Portfolio APIs
export const portfolioApi = {
  getMetrics: () => api.get('/portfolio/metrics'),
  getHistory: () => api.get('/portfolio/history'),
  optimize: (request) => api.post('/portfolio/optimize', request),
};

// Simulation APIs
export const simulationApi = {
  step: () => api.post('/simulation/step'),
};

// Alert APIs
export const alertApi = {
  simulate: (request) => api.post('/alerts/simulate', request),
};

// Utility functions
export const formatCurrency = (value) => {
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  }).format(value);
};

export const formatPercentage = (value, decimals = 2) => {
  return new Intl.NumberFormat('en-US', {
    style: 'percent',
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
};

export const formatNumber = (value, decimals = 0) => {
  return new Intl.NumberFormat('en-US', {
    minimumFractionDigits: decimals,
    maximumFractionDigits: decimals,
  }).format(value);
};

export default api; 