import React, { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Send, 
  Bot, 
  Activity,
  Shield
} from 'lucide-react';
import { apiService, PredictionResponse, DiseasePrediction } from '../services/api';
import toast from 'react-hot-toast';
import ReactMarkdown from 'react-markdown';

interface Message {
  id: string;
  type: 'user' | 'bot';
  content: string;
  timestamp: Date;
  prediction?: DiseasePrediction[];
}

const Chat: React.FC = () => {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [symptoms, setSymptoms] = useState<string[]>([]);
  const [availableSymptoms, setAvailableSymptoms] = useState<string[]>([]);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadSymptoms();
    addWelcomeMessage();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadSymptoms = async () => {
    try {
      const response = await apiService.getSymptoms();
      setAvailableSymptoms(response.symptoms);
    } catch (error) {
      console.error('Failed to load symptoms:', error);
      toast.error('Failed to load symptoms');
    }
  };

  const addWelcomeMessage = () => {
    const welcomeMessage: Message = {
      id: 'welcome',
      type: 'bot',
      content: `# 🏥 Welcome to Medical RAG Chatbot!

I'm your AI-powered medical assistant. I can help you:

🔍 **Predict diseases** from your symptoms
📊 **Analyze risk factors** and severity
💊 **Suggest treatments** and medications
⚠️ **Provide precautions** and safety advice

**How to use:**
1. Describe your symptoms (e.g., "fever, headache, nausea")
2. I'll analyze and provide predictions
3. Get detailed information about each condition

**Important:** This is for educational purposes only. Always consult a healthcare professional for medical advice.

What symptoms are you experiencing?`,
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || loading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      type: 'user',
      content: input.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setLoading(true);

    try {
      // Extract symptoms from user input
      const symptomList = extractSymptoms(input.trim());
      setSymptoms(symptomList);

      if (symptomList.length === 0) {
        const botMessage: Message = {
          id: Date.now().toString() + '_response',
          type: 'bot',
          content: `I couldn't identify specific symptoms in your message. Please describe your symptoms clearly, such as:

- "I have fever and headache"
- "Experiencing nausea and dizziness"
- "Pain in chest and shortness of breath"

You can also select from common symptoms: ${availableSymptoms.slice(0, 10).join(', ')}...`,
          timestamp: new Date()
        };
        setMessages(prev => [...prev, botMessage]);
        return;
      }

      // Get disease prediction
      const prediction = await apiService.predictDisease(symptomList);
      
      const botMessage: Message = {
        id: Date.now().toString() + '_response',
        type: 'bot',
        content: formatDiseasePrediction(prediction),
        timestamp: new Date(),
        prediction: prediction.predictions
      };

      setMessages(prev => [...prev, botMessage]);
    } catch (error: any) {
      const errorMessage: Message = {
        id: Date.now().toString() + '_error',
        type: 'bot',
        content: `❌ **Error occurred while analyzing your symptoms.**

Please try again or contact support if the issue persists.

**Error:** ${error.response?.data?.detail || 'Unknown error'}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
      toast.error('Failed to get prediction');
    } finally {
      setLoading(false);
    }
  };

  const extractSymptoms = (text: string): string[] => {
    const lowerText = text.toLowerCase();
    const foundSymptoms: string[] = [];
    
    // First, try to split by common separators (comma, semicolon, "and", "with")
    const possibleSymptoms = lowerText
      .split(/[,;]| and | with | plus | also /)
      .map(s => s.trim())
      .filter(s => s.length > 0);
    
    // Check each possible symptom against available symptoms
    possibleSymptoms.forEach(possibleSymptom => {
      // Direct match
      if (availableSymptoms.includes(possibleSymptom)) {
        foundSymptoms.push(possibleSymptom);
        return;
      }
      
      // Partial match - check if any available symptom contains this text
      const matchingSymptom = availableSymptoms.find(symptom => 
        symptom.toLowerCase().includes(possibleSymptom) || 
        possibleSymptom.includes(symptom.toLowerCase())
      );
      
      if (matchingSymptom && !foundSymptoms.includes(matchingSymptom)) {
        foundSymptoms.push(matchingSymptom);
      }
    });
    
    // If still no matches, try the original method for single symptoms
    if (foundSymptoms.length === 0) {
      availableSymptoms.forEach(symptom => {
        if (lowerText.includes(symptom.toLowerCase())) {
          foundSymptoms.push(symptom);
        }
      });
    }
    
    // Fallback to common symptoms if still no matches
    if (foundSymptoms.length === 0) {
      const commonSymptoms = ['fever', 'headache', 'nausea', 'pain', 'fatigue', 'dizziness', 'cough', 'sore throat', 'cold', 'runny nose'];
      commonSymptoms.forEach(symptom => {
        if (lowerText.includes(symptom)) {
          foundSymptoms.push(symptom);
        }
      });
    }

    return foundSymptoms;
  };

  const formatDiseasePrediction = (prediction: PredictionResponse): string => {
    const { predictions, input_symptoms, total_symptoms, analysis_summary } = prediction;
    
    let content = `## 🏥 Disease Prediction Results\n\n`;
    content += `**Input Symptoms:** ${input_symptoms?.join(', ') || 'Not specified'}\n`;
    content += `**Total Symptoms:** ${total_symptoms || 0}\n\n`;
    
    if (predictions && predictions.length > 0) {
      predictions.forEach((pred, index) => {
        const { disease, confidence, severity_score, urgency_level, risk_factors, treatment_info, precautions } = pred;
        
        content += `### ${index + 1}. ${disease}\n`;
        content += `**Confidence:** ${(confidence * 100).toFixed(1)}%\n`;
        content += `**Severity Score:** ${severity_score}/10\n`;
        content += `**Urgency Level:** ${urgency_level}\n\n`;

        if (risk_factors && risk_factors.length > 0) {
          content += `**Risk Factors:**\n`;
          risk_factors.forEach(factor => {
            content += `- **${factor.factor}:** ${factor.level} - ${factor.description}\n`;
          });
          content += `\n`;
        }

        if (treatment_info && treatment_info.drugs && treatment_info.drugs.length > 0) {
          content += `**Treatment Information:**\n`;
          content += `**Drugs:** ${treatment_info.drugs.map(drug => `${drug.name} (${drug.type})`).join(', ')}\n`;
          if (treatment_info.treatments) {
            content += `**Treatments:** ${treatment_info.treatments.join(', ')}\n`;
          }
          if (treatment_info.duration) {
            content += `**Duration:** ${treatment_info.duration}\n`;
          }
          content += `\n`;
        }

        if (precautions && precautions.length > 0) {
          content += `**Precautions:**\n`;
          precautions.forEach(precaution => {
            content += `- ${precaution}\n`;
          });
          content += `\n`;
        }
      });
    }

    if (analysis_summary) {
      content += `### 📊 Analysis Summary\n${analysis_summary}\n\n`;
    }

    content += `### ⚠️ Important Medical Disclaimer\n`;
    content += `This prediction is for educational purposes only and should not replace professional medical advice. Please consult a healthcare professional for proper diagnosis and treatment.`;

    return content;
  };


  return (
    <div className="chat-container">
      {/* Header */}
      <div className="chat-header">
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <div className="w-10 h-10 bg-medical rounded-full flex items-center justify-center mr-3 shadow-md">
              <Bot size={20} color="white" />
            </div>
            <div>
              <h1 className="chat-title">Medical RAG Chatbot</h1>
              <p className="chat-subtitle">AI-powered disease prediction</p>
            </div>
          </div>
          <div className="flex items-center space-x-4">
            <div className="flex items-center text-sm text-gray-600 bg-gray-100 px-3 py-1 rounded-full">
              <Activity size={16} className="mr-1" />
              {symptoms.length} symptoms
            </div>
            <div className="flex items-center text-sm text-gray-600 bg-green-100 text-green-700 px-3 py-1 rounded-full">
              <Shield size={16} className="mr-1" />
              WHO/openFDA verified
            </div>
          </div>
        </div>
      </div>

      {/* Messages */}
      <div className="chat-messages">
        <AnimatePresence>
          {messages.map((message) => (
            <motion.div
              key={message.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className={`message ${message.type}`}
            >
              <div className="message-content">
                <div className="prose prose-sm max-w-none">
                  <ReactMarkdown>{message.content}</ReactMarkdown>
                </div>
                <div className="message-time">
                  {message.timestamp.toLocaleTimeString()}
                </div>
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
        
        {loading && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="message bot"
          >
            <div className="message-content">
              <div className="flex items-center">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-medical mr-2"></div>
                <span className="text-sm text-gray-600">Analyzing symptoms...</span>
              </div>
            </div>
          </motion.div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="chat-input">
        <form onSubmit={handleSubmit} className="flex space-x-4">
          <div className="flex-1">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Describe your symptoms (e.g., fever, headache, nausea)..."
              className="form-input"
              disabled={loading}
            />
          </div>
          <button
            type="submit"
            disabled={!input.trim() || loading}
            className="btn btn-primary"
          >
            {loading ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
            ) : (
              <Send size={16} />
            )}
          </button>
        </form>
        
        {symptoms.length > 0 && (
          <div className="mt-4">
            <p className="text-sm text-gray-600 mb-2 font-medium">Detected symptoms:</p>
            <div className="flex flex-wrap gap-2">
              {symptoms.map((symptom, index) => (
                <span
                  key={index}
                  className="px-3 py-1 bg-medical text-white text-xs rounded-full font-medium shadow-sm"
                >
                  {symptom}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Chat;
