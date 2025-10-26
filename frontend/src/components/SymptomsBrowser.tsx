import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { Filter, List, Grid, ChevronLeft, Activity, AlertCircle } from 'lucide-react';
import { apiService } from '../services/api';
import toast from 'react-hot-toast';

interface Symptom {
  name: string;
  category?: string;
  severity?: string;
}

const SymptomsBrowser: React.FC = () => {
  const [symptoms, setSymptoms] = useState<string[]>([]);
  const [filteredSymptoms, setFilteredSymptoms] = useState<string[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [viewMode, setViewMode] = useState<'grid' | 'list'>('grid');
  const [loading, setLoading] = useState(true);
  const [selectedSymptoms, setSelectedSymptoms] = useState<string[]>([]);

  useEffect(() => {
    loadSymptoms();
  }, []);

  useEffect(() => {
    filterSymptoms();
  }, [symptoms, searchTerm, selectedCategory]);

  const loadSymptoms = async () => {
    try {
      setLoading(true);
      const response = await apiService.getSymptoms();
      setSymptoms(response.symptoms);
    } catch (error) {
      console.error('Failed to load symptoms:', error);
      toast.error('Failed to load symptoms');
    } finally {
      setLoading(false);
    }
  };

  const filterSymptoms = () => {
    let filtered = symptoms;

    // Filter by search term
    if (searchTerm) {
      filtered = filtered.filter(symptom =>
        symptom.toLowerCase().includes(searchTerm.toLowerCase())
      );
    }

    // Filter by category (simplified categorization)
    if (selectedCategory !== 'all') {
      filtered = filtered.filter(symptom => {
        const symptomLower = symptom.toLowerCase();
        switch (selectedCategory) {
          case 'respiratory':
            return symptomLower.includes('cough') || symptomLower.includes('breath') || 
                   symptomLower.includes('chest') || symptomLower.includes('throat');
          case 'neurological':
            return symptomLower.includes('headache') || symptomLower.includes('dizziness') || 
                   symptomLower.includes('seizure') || symptomLower.includes('memory');
          case 'gastrointestinal':
            return symptomLower.includes('nausea') || symptomLower.includes('vomit') || 
                   symptomLower.includes('stomach') || symptomLower.includes('diarrhea');
          case 'musculoskeletal':
            return symptomLower.includes('pain') || symptomLower.includes('ache') || 
                   symptomLower.includes('joint') || symptomLower.includes('muscle');
          case 'general':
            return symptomLower.includes('fever') || symptomLower.includes('fatigue') || 
                   symptomLower.includes('weakness') || symptomLower.includes('weight');
          default:
            return true;
        }
      });
    }

    setFilteredSymptoms(filtered);
  };

  const categories = [
    { value: 'all', label: 'All Symptoms', count: symptoms.length },
    { value: 'respiratory', label: 'Respiratory', count: 0 },
    { value: 'neurological', label: 'Neurological', count: 0 },
    { value: 'gastrointestinal', label: 'Gastrointestinal', count: 0 },
    { value: 'musculoskeletal', label: 'Musculoskeletal', count: 0 },
    { value: 'general', label: 'General', count: 0 }
  ];

  const handleSymptomClick = (symptom: string) => {
    setSelectedSymptoms(prev => {
      if (prev.includes(symptom)) {
        // Remove if already selected
        const newSelection = prev.filter(s => s !== symptom);
        toast.success(`Removed ${symptom} from selection`);
        return newSelection;
      } else {
        // Add if not selected
        const newSelection = [...prev, symptom];
        toast.success(`Added ${symptom} to selection`);
        return newSelection;
      }
    });
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-medical-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Enhanced Header */}
      <div className="bg-gradient-to-r from-white via-gray-50 to-white shadow-lg border-b border-gray-100">
        <div className="container mx-auto px-6 py-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <button
                onClick={() => window.history.back()}
                className="mr-6 p-3 hover:bg-gray-100 rounded-xl transition-all duration-200 hover:shadow-md"
              >
                <ChevronLeft size={24} className="text-gray-600" />
              </button>
              <div>
                <h1 className="text-3xl font-bold text-gray-900 mb-2">Browse Symptoms</h1>
                <p className="text-lg text-gray-600">Explore our comprehensive database of medical symptoms</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <div className="flex items-center bg-white rounded-xl shadow-sm border border-gray-200 p-1">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`p-3 rounded-lg transition-all duration-200 ${
                    viewMode === 'grid' 
                      ? 'bg-medical-100 text-medical-600 shadow-sm' 
                      : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <Grid size={20} />
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`p-3 rounded-lg transition-all duration-200 ${
                    viewMode === 'list' 
                      ? 'bg-medical-100 text-medical-600 shadow-sm' 
                      : 'text-gray-500 hover:text-gray-700 hover:bg-gray-50'
                  }`}
                >
                  <List size={20} />
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-6">
        {/* Enhanced Search and Filters */}
        <div className="bg-gradient-to-r from-white to-gray-50 rounded-2xl shadow-lg border border-gray-100 p-8 mb-8">
          <div className="text-center mb-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-2">Explore Symptoms Database</h2>
            <p className="text-gray-600">Find and explore symptoms to understand your health better</p>
          </div>
          
          <div className="flex flex-col lg:flex-row gap-6">
            {/* Enhanced Search */}
            <div className="flex-1">
              <div className="relative group">
                <input
                  type="text"
                  placeholder="Search symptoms (e.g., fever, headache, nausea)..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="w-full px-6 py-4 border-2 border-gray-200 rounded-xl focus:ring-4 focus:ring-medical-100 focus:border-medical-500 transition-all duration-200 text-lg bg-white"
                />
                {searchTerm && (
                  <button
                    onClick={() => setSearchTerm('')}
                    className="absolute right-4 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors bg-white rounded-full p-1 hover:bg-gray-100 shadow-sm"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                )}
              </div>
            </div>

            {/* Enhanced Category Filter */}
            <div className="lg:w-80">
              <div className="relative">
                <select
                  value={selectedCategory}
                  onChange={(e) => setSelectedCategory(e.target.value)}
                  className="w-full px-4 py-4 border-2 border-gray-200 rounded-xl focus:ring-4 focus:ring-medical-100 focus:border-medical-500 transition-all duration-200 text-lg appearance-none bg-white cursor-pointer"
                >
                  {categories.map(category => (
                    <option key={category.value} value={category.value}>
                      {category.label} ({category.count})
                    </option>
                  ))}
                </select>
                <div className="absolute right-4 top-1/2 transform -translate-y-1/2 pointer-events-none">
                  <svg className="w-5 h-5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </div>
              </div>
            </div>
          </div>

          {/* Enhanced Results Summary */}
          <div className="mt-6 flex flex-col sm:flex-row items-center justify-between gap-4">
            <div className="flex items-center space-x-4">
              <div className="flex items-center bg-medical-100 text-medical-800 px-4 py-2 rounded-full">
                <Activity size={16} className="mr-2" />
                <span className="font-medium">{filteredSymptoms.length} symptoms found</span>
              </div>
              <div className="text-gray-600">
                of {symptoms.length} total symptoms
              </div>
            </div>
            
            {searchTerm && (
              <button
                onClick={() => setSearchTerm('')}
                className="flex items-center space-x-2 text-medical-600 hover:text-medical-700 font-medium transition-colors"
              >
                <span>Clear search</span>
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            )}
          </div>
        </div>

        {/* Enhanced Symptoms Grid/List */}
        {filteredSymptoms.length === 0 ? (
          <div className="text-center py-16">
            <div className="bg-gray-100 rounded-full w-24 h-24 flex items-center justify-center mx-auto mb-6">
              <AlertCircle size={48} className="text-gray-400" />
            </div>
            <h3 className="text-xl font-semibold text-gray-900 mb-3">No symptoms found</h3>
            <p className="text-gray-600 mb-6">Try adjusting your search or filter criteria</p>
            <button
              onClick={() => setSearchTerm('')}
              className="inline-flex items-center px-6 py-3 bg-medical-600 text-white rounded-lg hover:bg-medical-700 transition-colors"
            >
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
              Clear search
            </button>
          </div>
        ) : (
          <div className="space-y-6">
            {/* Selection Summary and View Toggle */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-4">
                <h3 className="text-lg font-semibold text-gray-900">Available Symptoms</h3>
                {selectedSymptoms.length > 0 && (
                  <div className="flex items-center space-x-2">
                    <div className="bg-gradient-to-r from-medical-500 to-medical-600 text-white px-4 py-2 rounded-full text-sm font-bold shadow-md">
                      {selectedSymptoms.length} selected
                    </div>
                    <button
                      onClick={() => setSelectedSymptoms([])}
                      className="text-sm text-medical-600 hover:text-medical-800 font-medium transition-colors"
                    >
                      Clear all
                    </button>
                  </div>
                )}
              </div>
              <div className="flex items-center bg-gray-100 rounded-lg p-1">
                <button
                  onClick={() => setViewMode('grid')}
                  className={`p-2 rounded-md transition-colors ${
                    viewMode === 'grid' 
                      ? 'bg-white shadow-sm text-medical-600' 
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  <Grid size={18} />
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`p-2 rounded-md transition-colors ${
                    viewMode === 'list' 
                      ? 'bg-white shadow-sm text-medical-600' 
                      : 'text-gray-600 hover:text-gray-900'
                  }`}
                >
                  <List size={18} />
                </button>
              </div>
            </div>

            {/* Symptoms Display */}
            <div className={
              viewMode === 'grid' 
                ? 'grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4'
                : 'space-y-3'
            }>
              {filteredSymptoms.map((symptom, index) => {
                const isSelected = selectedSymptoms.includes(symptom);
                return (
                  <motion.div
                    key={symptom}
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.02 }}
                    onClick={() => handleSymptomClick(symptom)}
                    className={`
                      group relative overflow-hidden rounded-xl border-2 transition-all duration-200 cursor-pointer
                      ${isSelected 
                        ? 'border-medical-500 bg-gradient-to-br from-medical-50 to-medical-100 shadow-lg ring-2 ring-medical-200' 
                        : 'border-gray-100 hover:border-medical-200'
                      }
                      ${viewMode === 'grid' 
                        ? isSelected ? 'bg-gradient-to-br from-medical-50 to-medical-100 p-6 shadow-lg' : 'bg-white p-6 hover:shadow-lg' 
                        : isSelected ? 'bg-gradient-to-br from-medical-50 to-medical-100 p-4 shadow-md flex items-center justify-between' : 'bg-white p-4 hover:shadow-md flex items-center justify-between'
                      }
                    `}
                  >
                  {/* Gradient overlay on hover */}
                  <div className="absolute inset-0 bg-gradient-to-br from-medical-50 to-transparent opacity-0 group-hover:opacity-100 transition-opacity duration-200"></div>
                  
                  <div className="relative flex items-center">
                    <div className={`
                      rounded-xl flex items-center justify-center transition-all duration-200 group-hover:scale-110
                      ${isSelected 
                        ? 'bg-gradient-to-br from-medical-500 to-medical-600 shadow-md' 
                        : viewMode === 'grid' 
                          ? 'w-12 h-12 bg-gradient-to-br from-medical-100 to-medical-200 mr-4' 
                          : 'w-10 h-10 bg-medical-100 mr-3'
                      }
                      ${viewMode === 'grid' ? 'w-12 h-12 mr-4' : 'w-10 h-10 mr-3'}
                    `}>
                      {isSelected ? (
                        <svg className="w-5 h-5 text-white drop-shadow-sm" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={3} d="M5 13l4 4L19 7" />
                        </svg>
                      ) : (
                        <Activity size={viewMode === 'grid' ? 24 : 20} className="text-medical-600" />
                      )}
                    </div>
                    
                    <div className="flex-1">
                      <h3 className={`
                        font-semibold capitalize transition-colors
                        ${isSelected 
                          ? 'text-medical-800 font-bold' 
                          : 'text-gray-900 group-hover:text-medical-700'
                        }
                        ${viewMode === 'grid' ? 'text-lg mb-1' : 'text-base'}
                      `}>
                        {symptom.replace(/_/g, ' ')}
                      </h3>
                      {viewMode === 'list' && (
                        <p className={`text-sm transition-colors font-medium ${
                          isSelected 
                            ? 'text-medical-700' 
                            : 'text-gray-500 group-hover:text-gray-600'
                        }`}>
                          {isSelected ? 'Selected for prediction' : 'Click to use in disease prediction'}
                        </p>
                      )}
                    </div>
                  </div>
                  
                  {viewMode === 'list' && (
                    <div className="text-medical-500 group-hover:text-medical-600 transition-colors">
                      <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                      </svg>
                    </div>
                  )}
                  
                  {/* Hover effect indicator */}
                  <div className={`absolute bottom-0 left-0 right-0 h-1 transition-transform duration-200 ${
                    isSelected 
                      ? 'bg-gradient-to-r from-medical-500 to-medical-600 scale-x-100' 
                      : 'bg-gradient-to-r from-medical-500 to-medical-600 transform scale-x-0 group-hover:scale-x-100'
                  }`}></div>
                </motion.div>
                );
              })}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default SymptomsBrowser;
