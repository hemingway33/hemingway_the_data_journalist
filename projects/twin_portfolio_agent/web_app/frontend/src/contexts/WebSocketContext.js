import React, { createContext, useContext, useEffect, useState, useCallback } from 'react';
import toast from 'react-hot-toast';

const WebSocketContext = createContext();

export const useWebSocket = () => {
  const context = useContext(WebSocketContext);
  if (!context) {
    throw new Error('useWebSocket must be used within a WebSocketProvider');
  }
  return context;
};

export const WebSocketProvider = ({ children }) => {
  const [socket, setSocket] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [systemStatus, setSystemStatus] = useState(null);
  const [portfolioMetrics, setPortfolioMetrics] = useState(null);
  const [alerts, setAlerts] = useState([]);
  const [agentMessages, setAgentMessages] = useState([]);

  const clientId = `client_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

  const connect = useCallback(() => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      return;
    }

    const wsUrl = process.env.NODE_ENV === 'production' 
      ? `wss://${window.location.host}/ws/${clientId}`
      : `ws://localhost:8000/ws/${clientId}`;

    console.log('Connecting to WebSocket:', wsUrl);
    
    const newSocket = new WebSocket(wsUrl);

    newSocket.onopen = () => {
      console.log('WebSocket connected');
      setIsConnected(true);
      toast.success('Connected to real-time updates');
      
      // Send initial ping
      newSocket.send(JSON.stringify({ type: 'ping' }));
    };

    newSocket.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        handleMessage(message);
      } catch (error) {
        console.error('Error parsing WebSocket message:', error);
      }
    };

    newSocket.onclose = (event) => {
      console.log('WebSocket disconnected:', event.code, event.reason);
      setIsConnected(false);
      setSocket(null);
      
      if (!event.wasClean) {
        toast.error('Connection lost. Attempting to reconnect...');
        // Attempt to reconnect after 3 seconds
        setTimeout(connect, 3000);
      }
    };

    newSocket.onerror = (error) => {
      console.error('WebSocket error:', error);
      toast.error('WebSocket connection error');
    };

    setSocket(newSocket);
  }, [clientId]);

  const disconnect = useCallback(() => {
    if (socket) {
      socket.close(1000, 'User initiated disconnect');
      setSocket(null);
      setIsConnected(false);
    }
  }, [socket]);

  const sendMessage = useCallback((message) => {
    if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(message));
    } else {
      console.warn('WebSocket not connected. Cannot send message:', message);
    }
  }, [socket]);

  const requestMetricsUpdate = useCallback(() => {
    sendMessage({ type: 'request_metrics' });
  }, [sendMessage]);

  const handleMessage = useCallback((message) => {
    switch (message.type) {
      case 'system_status':
        setSystemStatus(message.data);
        break;
        
      case 'metrics_update':
        setPortfolioMetrics(message.data);
        break;
        
      case 'market_alert':
        const alertWithId = {
          ...message.data,
          id: Date.now(),
          timestamp: new Date().toISOString(),
        };
        setAlerts(prev => [alertWithId, ...prev.slice(0, 9)]); // Keep last 10 alerts
        toast.error(`Market Alert: ${message.data.alert_type}`, {
          duration: 6000,
        });
        break;
        
      case 'agent_created':
        toast.success(`Agent created: ${message.data.role} (${message.data.user_id})`);
        setAgentMessages(prev => [...prev, {
          id: Date.now(),
          type: 'agent_created',
          content: `New ${message.data.role} agent created for ${message.data.user_id}`,
          timestamp: new Date().toISOString(),
        }]);
        break;
        
      case 'optimization_complete':
        toast.success(`Portfolio optimization completed for ${message.data.user_id}`);
        setAgentMessages(prev => [...prev, {
          id: Date.now(),
          type: 'optimization',
          content: `Portfolio optimization completed for ${message.data.user_id}`,
          data: message.data.result,
          timestamp: new Date().toISOString(),
        }]);
        break;
        
      case 'simulation_step':
        // Update metrics silently during simulation
        if (message.data.info) {
          const updatedMetrics = {
            portfolio_value: message.data.info.total_value || 0,
            current_day: message.data.day,
            reward: message.data.reward,
          };
          setPortfolioMetrics(prev => ({
            ...prev,
            simulation: updatedMetrics,
          }));
        }
        break;
        
      case 'pong':
        // Handle ping response
        break;
        
      default:
        console.log('Unknown message type:', message.type, message);
    }
  }, []);

  // Auto-connect on mount
  useEffect(() => {
    connect();
    
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  // Periodic ping to keep connection alive
  useEffect(() => {
    if (!isConnected) return;

    const pingInterval = setInterval(() => {
      sendMessage({ type: 'ping' });
    }, 30000); // Ping every 30 seconds

    return () => clearInterval(pingInterval);
  }, [isConnected, sendMessage]);

  const value = {
    socket,
    isConnected,
    connect,
    disconnect,
    sendMessage,
    systemStatus,
    portfolioMetrics,
    alerts,
    agentMessages,
    requestMetricsUpdate,
    clientId,
  };

  return (
    <WebSocketContext.Provider value={value}>
      {children}
    </WebSocketContext.Provider>
  );
}; 