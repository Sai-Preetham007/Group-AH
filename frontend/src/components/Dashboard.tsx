import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { motion } from 'framer-motion';
import { apiService, UserStats } from '../services/api';
import { 
  Stethoscope, 
  Activity, 
  TrendingUp, 
  Shield, 
  Clock,
  ArrowRight,
  Brain,
  FileText,
  Database,
  Zap,
  LogOut
} from 'lucide-react';

interface Stats {
  totalPredictions: number;
  accuracyRate: number;
  sourcesVerified: number;
  lastActivity: string;
}

const Dashboard: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [stats, setStats] = useState<Stats>({
    totalPredictions: 0,
    accuracyRate: 0,
    sourcesVerified: 0,
    lastActivity: 'Never'
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, [user]); // Refresh when user changes (login/logout)

  const loadDashboardData = async () => {
    try {
      const userStats: UserStats = await apiService.getUserStats();
      setStats({
        totalPredictions: userStats.total_predictions,
        accuracyRate: userStats.accuracy_rate,
        sourcesVerified: userStats.sources_verified,
        lastActivity: userStats.last_activity
      });
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
      // Fallback to default values if API fails
      setStats({
        totalPredictions: 0,
        accuracyRate: 0,
        sourcesVerified: 0,
        lastActivity: 'Never'
      });
    } finally {
      setLoading(false);
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  const quickActions = [
    {
      title: 'Disease Prediction',
      description: 'Input symptoms to get AI-powered disease predictions',
      icon: Brain,
      color: 'bg-blue-500',
      onClick: () => navigate('/chat')
    },
    {
      title: 'Browse Symptoms',
      description: 'Explore available symptoms in our database',
      icon: Activity,
      color: 'bg-green-500',
      onClick: () => navigate('/symptoms')
    },
    {
      title: 'Disease Database',
      description: 'Search and explore disease information',
      icon: Database,
      color: 'bg-purple-500',
      onClick: () => navigate('/diseases')
    },
    {
      title: 'API Documentation',
      description: 'View interactive API documentation',
      icon: FileText,
      color: 'bg-orange-500',
      onClick: () => window.open('http://localhost:8000/docs', '_blank')
    }
  ];

  const features = [
    {
      title: 'AI-Powered Diagnosis',
      description: 'Advanced machine learning models trained on medical datasets',
      icon: Brain
    },
    {
      title: 'Source Verification',
      description: 'All information verified against WHO and openFDA databases',
      icon: Shield
    },
    {
      title: 'Real-time Analysis',
      description: 'Instant symptom analysis with confidence scoring',
      icon: Zap
    }
  ];

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-medical-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 via-blue-50 to-medical-50">
      {/* Enhanced Header */}
      <header className="bg-gradient-to-r from-medical-500 to-medical-600 shadow-lg">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6 header-container" style={{paddingLeft: '1.5rem'}}>
            <div className="flex items-center">
              <div className="bg-white/20 backdrop-blur-sm rounded-xl p-3 mr-4 header-logo-spacing">
                <Stethoscope className="h-8 w-8 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-white">Medical RAG Chatbot</h1>
                <p className="text-medical-100 text-sm">AI-Powered Medical Assistant</p>
              </div>
            </div>
            <div className="flex items-center space-x-6">
              <div className="bg-white/10 backdrop-blur-sm rounded-lg px-4 py-2">
                <span className="text-white font-medium">
                  Welcome, {user?.full_name || user?.username}
                </span>
              </div>
              <button
                onClick={handleLogout}
                className="logout-button"
              >
                <LogOut className="logout-icon" />
                <span className="logout-text">Logout</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Enhanced Welcome Section */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="mb-8"
        >
          <div className="bg-white/80 backdrop-blur-sm rounded-2xl shadow-xl p-8 border border-white/20">
            <div className="flex items-center mb-4" style={{alignItems: 'center'}}>
              <div className="welcome-logo-container" style={{display: 'flex', alignItems: 'center', justifyContent: 'center', marginRight: '8px'}}>
                <Brain className="h-10 w-10 text-white drop-shadow-sm relative z-10" />
              </div>
              <div style={{width: '8px', flexShrink: 0}}></div>
              <div className="welcome-text-container">
                <h2 className="text-3xl font-bold bg-gradient-to-r from-gray-900 to-medical-700 bg-clip-text text-transparent welcome-title">
                  Welcome to Medical RAG Chatbot
                </h2>
                <p className="text-lg text-gray-600 welcome-subtitle">
                  Your AI-powered medical assistant for accurate disease prediction and health information
                </p>
              </div>
            </div>
            <div className="flex items-center space-x-6 text-sm text-gray-500">
              <div className="flex items-center">
                <Shield className="h-4 w-4 text-medical-500 mr-2" />
                <span>WHO & FDA Verified</span>
              </div>
              <div className="flex items-center">
                <Zap className="h-4 w-4 text-medical-500 mr-2" />
                <span>Real-time Analysis</span>
              </div>
              <div className="flex items-center">
                <Activity className="h-4 w-4 text-medical-500 mr-2" />
                <span>AI-Powered Diagnosis</span>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Stats Cards */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
        >
          <div className="bg-white/90 backdrop-blur-sm rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 p-6 border border-white/20 hover:scale-105">
            <div className="flex items-center">
              <div className="bg-blue-100 rounded-full p-3 mr-4 border-2 border-blue-200">
                <Activity className="h-8 w-8 text-blue-600" />
              </div>
              <div>
                <p className="text-sm font-medium text-gray-600">Total Predictions</p>
                <p className="text-2xl font-bold text-gray-900">{stats.totalPredictions}</p>
              </div>
            </div>
          </div>

          <div className="bg-white/90 backdrop-blur-sm rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 p-6 border border-white/20 hover:scale-105">
            <div className="flex items-center">
              <div className="bg-green-100 rounded-full p-3 mr-4 border-2 border-green-200">
                <TrendingUp className="h-8 w-8 text-green-600" />
              </div>
              <div>
                <p className="text-sm font-medium text-gray-600">Accuracy Rate</p>
                <p className="text-2xl font-bold text-gray-900">{stats.accuracyRate}%</p>
              </div>
            </div>
          </div>

          <div className="bg-white/90 backdrop-blur-sm rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 p-6 border border-white/20 hover:scale-105">
            <div className="flex items-center">
              <div className="bg-purple-100 rounded-full p-3 mr-4 border-2 border-purple-200">
                <Shield className="h-8 w-8 text-purple-600" />
              </div>
              <div>
                <p className="text-sm font-medium text-gray-600">Sources Verified</p>
                <p className="text-2xl font-bold text-gray-900">{stats.sourcesVerified}</p>
              </div>
            </div>
          </div>

          <div className="bg-white/90 backdrop-blur-sm rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 p-6 border border-white/20 hover:scale-105">
            <div className="flex items-center">
              <div className="bg-orange-100 rounded-full p-3 mr-4 border-2 border-orange-200">
                <Clock className="h-8 w-8 text-orange-600" />
              </div>
              <div>
                <p className="text-sm font-medium text-gray-600">Last Activity</p>
                <p className="text-2xl font-bold text-gray-900">{stats.lastActivity}</p>
              </div>
            </div>
          </div>
        </motion.div>

        {/* Enhanced Quick Actions */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-8"
        >
          <div className="flex items-center mb-6">
            <div className="bg-gradient-to-r from-medical-500 to-medical-600 rounded-lg p-2 mr-3">
              <Zap className="h-6 w-6 text-white" />
            </div>
            <h3 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-medical-700 bg-clip-text text-transparent">
              Quick Actions
            </h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            {quickActions.map((action, index) => (
              <motion.div
                key={action.title}
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.3 + index * 0.1 }}
                onClick={action.onClick}
                className="bg-white/90 backdrop-blur-sm rounded-xl shadow-lg hover:shadow-xl transition-all duration-300 cursor-pointer p-6 border border-white/20 hover:scale-105 group"
              >
                <div className="flex items-center mb-4">
                  <div className={`p-3 rounded-full border-2 group-hover:scale-110 transition-transform duration-200 ${action.color === 'bg-blue-500' ? 'bg-blue-100 border-blue-200' : action.color === 'bg-green-500' ? 'bg-green-100 border-green-200' : action.color === 'bg-purple-500' ? 'bg-purple-100 border-purple-200' : 'bg-orange-100 border-orange-200'}`}>
                    <action.icon className={`h-6 w-6 ${action.color === 'bg-blue-500' ? 'text-blue-600' : action.color === 'bg-green-500' ? 'text-green-600' : action.color === 'bg-purple-500' ? 'text-purple-600' : 'text-orange-600'}`} />
                  </div>
                  <h4 className="text-lg font-semibold text-gray-900 ml-3">{action.title}</h4>
                </div>
                <p className="text-gray-600 mb-4">{action.description}</p>
                <div className="flex items-center text-medical-600 font-medium group-hover:text-medical-700 transition-colors">
                  <span>Get Started</span>
                  <ArrowRight className="h-4 w-4 ml-2 group-hover:translate-x-1 transition-transform" />
                </div>
              </motion.div>
            ))}
          </div>
        </motion.div>

        {/* Enhanced Features */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="bg-white/90 backdrop-blur-sm rounded-2xl shadow-xl p-8 border border-white/20"
        >
          <div className="flex items-center mb-8">
            <div className="bg-gradient-to-r from-medical-500 to-medical-600 rounded-lg p-2 mr-3">
              <Shield className="h-6 w-6 text-white" />
            </div>
            <h3 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-medical-700 bg-clip-text text-transparent">
              Key Features
            </h3>
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <motion.div 
                key={feature.title} 
                className="text-center group"
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.5 + index * 0.1 }}
              >
                <div className="bg-gradient-to-r from-medical-100 to-medical-200 rounded-full p-4 w-16 h-16 mx-auto mb-4 flex items-center justify-center group-hover:scale-110 transition-transform duration-200 shadow-lg">
                  <feature.icon className="h-8 w-8 text-medical-600" />
                </div>
                <h4 className="text-lg font-semibold text-gray-900 mb-2">{feature.title}</h4>
                <p className="text-gray-600">{feature.description}</p>
              </motion.div>
            ))}
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Dashboard;
