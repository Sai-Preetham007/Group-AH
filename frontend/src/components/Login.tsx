import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { motion } from 'framer-motion';
import { Stethoscope, User, Lock, Eye, EyeOff, AlertCircle } from 'lucide-react';
import toast from 'react-hot-toast';

const Login: React.FC = () => {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    username: '',
    email: '',
    password: '',
    full_name: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [loading, setLoading] = useState(false);
  const [errors, setErrors] = useState<{[key: string]: string}>({});

  const { login, register } = useAuth();
  const navigate = useNavigate();

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
    // Clear error when user starts typing
    if (errors[name]) {
      setErrors(prev => ({
        ...prev,
        [name]: ''
      }));
    }
  };

  const validateForm = () => {
    const newErrors: {[key: string]: string} = {};

    if (!formData.username.trim()) {
      newErrors.username = 'Username is required';
    }

    if (!formData.password.trim()) {
      newErrors.password = 'Password is required';
    }

    if (!isLogin) {
      if (!formData.email.trim()) {
        newErrors.email = 'Email is required';
      } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
        newErrors.email = 'Email is invalid';
      }

      if (!formData.full_name.trim()) {
        newErrors.full_name = 'Full name is required';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!validateForm()) {
      return;
    }

    setLoading(true);

    try {
      let result;
      if (isLogin) {
        result = await login(formData.username, formData.password);
      } else {
        result = await register(formData);
      }

      if (result.success) {
        toast.success(isLogin ? 'Login successful!' : 'Registration successful! Please login.');
        if (isLogin) {
          navigate('/');
        } else {
          setIsLogin(true);
          setFormData({ username: '', email: '', password: '', full_name: '' });
        }
      } else {
        toast.error(result.error || 'An error occurred');
      }
    } catch (error) {
      toast.error('An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-container">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="login-form"
      >
        <div className="login-header">
          <motion.div
            initial={{ scale: 0 }}
            animate={{ scale: 1 }}
            transition={{ delay: 0.2, type: "spring", stiffness: 200 }}
            className="login-logo"
          >
            <Stethoscope size={32} color="white" />
          </motion.div>
          <h2 className="login-title">
            {isLogin ? 'Welcome Back' : 'Create Account'}
          </h2>
          <p className="login-subtitle">
            {isLogin 
              ? 'Sign in to your Medical RAG Chatbot account' 
              : 'Join the Medical RAG Chatbot platform'
            }
          </p>
        </div>

        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.3 }}
        >
          <form className="space-y-6" onSubmit={handleSubmit}>
            {!isLogin && (
              <div className="form-group">
                <label htmlFor="full_name" className="form-label">
                  Full Name
                </label>
                <div className="relative">
                  <User className="input-icon" size={16} />
                  <input
                    id="full_name"
                    name="full_name"
                    type="text"
                    value={formData.full_name}
                    onChange={handleInputChange}
                    className={`form-input with-icon ${errors.full_name ? 'error' : ''}`}
                    placeholder="Enter your full name"
                  />
                </div>
                {errors.full_name && (
                  <div className="error-message">
                    <AlertCircle size={14} />
                    {errors.full_name}
                  </div>
                )}
              </div>
            )}

            {!isLogin && (
              <div className="form-group">
                <label htmlFor="email" className="form-label">
                  Email Address
                </label>
                <div className="relative">
                  <input
                    id="email"
                    name="email"
                    type="email"
                    value={formData.email}
                    onChange={handleInputChange}
                    className={`form-input ${errors.email ? 'error' : ''}`}
                    placeholder="Enter your email"
                  />
                </div>
                {errors.email && (
                  <div className="error-message">
                    <AlertCircle size={14} />
                    {errors.email}
                  </div>
                )}
              </div>
            )}

            <div className="form-group">
              <label htmlFor="username" className="form-label">
                Username
              </label>
              <div className="relative">
                <User className="input-icon" size={16} />
                <input
                  id="username"
                  name="username"
                  type="text"
                  value={formData.username}
                  onChange={handleInputChange}
                  className={`form-input with-icon ${errors.username ? 'error' : ''}`}
                  placeholder="Enter your username"
                />
              </div>
              {errors.username && (
                <div className="error-message">
                  <AlertCircle size={14} />
                  {errors.username}
                </div>
              )}
            </div>

            <div className="form-group">
              <label htmlFor="password" className="form-label">
                Password
              </label>
              <div className="relative">
                <Lock className="input-icon" size={16} />
                <input
                  id="password"
                  name="password"
                  type={showPassword ? 'text' : 'password'}
                  value={formData.password}
                  onChange={handleInputChange}
                  className={`form-input with-icon ${errors.password ? 'error' : ''}`}
                  placeholder="Enter your password"
                />
                <button
                  type="button"
                  className="absolute right-3 top-1/2 transform -translate-y-1/2 translate-y-2 text-gray-400 hover:text-gray-600 transition-colors"
                  onClick={() => setShowPassword(!showPassword)}
                >
                  {showPassword ? <EyeOff size={16} /> : <Eye size={16} />}
                </button>
              </div>
              {errors.password && (
                <div className="error-message">
                  <AlertCircle size={14} />
                  {errors.password}
                </div>
              )}
            </div>

            <button
              type="submit"
              disabled={loading}
              className="btn btn-primary w-full"
            >
              {loading ? (
                <div className="flex items-center">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  {isLogin ? 'Signing in...' : 'Creating account...'}
                </div>
              ) : (
                isLogin ? 'Sign In' : 'Create Account'
              )}
            </button>

            <div className="text-center">
              <button
                type="button"
                onClick={() => {
                  setIsLogin(!isLogin);
                  setFormData({ username: '', email: '', password: '', full_name: '' });
                  setErrors({});
                }}
                className="text-sm text-medical font-medium hover:text-medical-primary-light transition-colors"
              >
                {isLogin 
                  ? "Don't have an account? Sign up" 
                  : "Already have an account? Sign in"
                }
              </button>
            </div>
          </form>
        </motion.div>

        <div className="text-center text-sm text-gray-600 mt-6">
          <p className="font-medium">Medical Knowledge RAG Chatbot</p>
          <p className="mt-1 text-xs">Powered by AI & WHO/openFDA verification</p>
        </div>
      </motion.div>
    </div>
  );
};

export default Login;
