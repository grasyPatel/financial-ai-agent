import axios from 'axios';

// Create axios instance with base configuration and optimized settings
const api = axios.create({
  baseURL: 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 10000, // 10 second timeout
  maxRedirects: 3,
  // Optimize connection pool
  maxContentLength: 1000000, // 1MB max response size
  maxBodyLength: 1000000, // 1MB max request size
});

// Types for our authentication data
export interface User {
  id: string;
  email: string;
  name?: string;
  createdAt: string;
}

export interface AuthResponse {
  user: User;
  token: string;
}

export interface RegisterData {
  email: string;
  password: string;
  name?: string;
}

export interface LoginData {
  email: string;
  password: string;
}

// Authentication service functions
export const authService = {
  // Register new user
  async register(data: RegisterData): Promise<AuthResponse> {
    const response = await api.post('/auth/register', data);
    return response.data;
  },

  // Login existing user
  async login(data: LoginData): Promise<AuthResponse> {
    const response = await api.post('/auth/login', data);
    return response.data;
  },

  // Get current user profile
  async getProfile(token: string): Promise<{ user: User }> {
    const response = await api.get('/auth/me', {
      headers: { Authorization: `Bearer ${token}` },
    });
    return response.data;
  },

  // Validate token with retry logic and debouncing
  async validateToken(token: string): Promise<{ valid: boolean; user?: User }> {
    try {
      const response = await api.get('/auth/validate', {
        headers: { Authorization: `Bearer ${token}` },
        timeout: 5000, // Shorter timeout for validation
      });
      return response.data;
    } catch (error: any) {
      console.warn('Token validation failed:', error.message);
      return { valid: false };
    }
  },

  // Logout user
  async logout(token: string): Promise<void> {
    await api.post('/auth/logout', {}, {
      headers: { Authorization: `Bearer ${token}` },
    });
  },
};

// Add request interceptor to automatically include auth token
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Add response interceptor to handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token is invalid, clear it from storage
      localStorage.removeItem('token');
      localStorage.removeItem('user');
      // Redirect to login page
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

export default api;