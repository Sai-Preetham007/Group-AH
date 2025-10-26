import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Create axios instance
const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor to add auth token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor to handle errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// Types
export interface User {
  id: number;
  username: string;
  email: string;
  full_name: string;
  is_active: boolean;
  is_admin: boolean;
  created_at: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

export interface LoginResponse {
  access_token: string;
  token_type: string;
}

export interface RegisterRequest {
  username: string;
  email: string;
  password: string;
  full_name: string;
}

export interface SymptomInput {
  symptoms: string[];
  age?: number;
  gender?: string;
}

export interface RiskFactor {
  factor: string;
  level: string;
  description: string;
}

export interface DrugInfo {
  name: string;
  type: string;
}

export interface TreatmentInfo {
  drugs: DrugInfo[];
  treatments: string[];
  duration: string;
  notes: string;
}

export interface DiseasePrediction {
  disease: string;
  confidence: number;
  description: string;
  precautions: string[];
  risk_factors: RiskFactor[];
  severity_score: number;
  urgency_level: string;
  treatment_info: TreatmentInfo | null;
}

export interface PredictionResponse {
  predictions: DiseasePrediction[];
  input_symptoms: string[];
  total_symptoms: number;
  analysis_summary: string;
}

export interface DiseaseInfo {
  name: string;
  description: string;
  symptoms: string[];
  precautions: string[];
  treatments: string[];
}

export interface UserStats {
  total_predictions: number;
  accuracy_rate: number;
  sources_verified: number;
  last_activity: string;
}

// API Service
export const apiService = {
  // Authentication
  login: (credentials: LoginRequest): Promise<LoginResponse> => {
    return api.post('/auth/login', credentials).then(response => response.data);
  },

  register: (userData: RegisterRequest): Promise<User> => {
    return api.post('/auth/register', userData).then(response => response.data);
  },

  getCurrentUser: (): Promise<User> => {
    return api.get('/auth/me').then(response => response.data);
  },

  // Medical endpoints
  predictDisease: (symptoms: string[]): Promise<PredictionResponse> => {
    return api.post('/predict-disease', { symptoms }).then(response => response.data);
  },

  getSymptoms: (): Promise<{ symptoms: string[] }> => {
    return api.get('/symptoms').then(response => response.data);
  },

  getDiseaseInfo: (diseaseName: string): Promise<DiseaseInfo> => {
    return api.get(`/disease/${diseaseName}`).then(response => response.data);
  },

  // User management
  getUsers: (): Promise<User[]> => {
    return api.get('/users').then(response => response.data);
  },

  getUser: (userId: number): Promise<User> => {
    return api.get(`/users/${userId}`).then(response => response.data);
  },

  getUserStats: (): Promise<UserStats> => {
    return api.get('/users/me/stats').then(response => response.data);
  },

  // Health check
  healthCheck: (): Promise<{ status: string; message: string; version: string }> => {
    return api.get('/').then(response => response.data);
  }
};

export default apiService;
