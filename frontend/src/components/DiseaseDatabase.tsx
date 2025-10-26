import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { ChevronLeft, Heart, AlertTriangle, Info } from 'lucide-react';
import { apiService } from '../services/api';
import toast from 'react-hot-toast';

interface DiseaseInfo {
  name: string;
  description: string;
  symptoms: string[];
  precautions: string[];
  treatments: string[];
}

const DiseaseDatabase: React.FC = () => {
  const [diseases, setDiseases] = useState<DiseaseInfo[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedDisease, setSelectedDisease] = useState<DiseaseInfo | null>(null);
  const [loading, setLoading] = useState(true);
  const [searchLoading, setSearchLoading] = useState(false);

  useEffect(() => {
    // Load some common diseases initially
    loadCommonDiseases();
  }, []);

  // Removed filterDiseases useEffect as it's not needed for this implementation

  const loadCommonDiseases = () => {
    // Common diseases for initial display
    const commonDiseases = [
      'Common Cold', 'Influenza', 'Diabetes', 'Hypertension', 'Asthma',
      'Migraine', 'Sinus Infection', 'Allergy', 'Pneumonia', 'Bronchitis'
    ];
    
    // For now, we'll show these as placeholders
    // In a real implementation, you'd fetch from the API
    setDiseases([]);
    setLoading(false);
  };

  const searchDisease = async (diseaseName: string) => {
    if (!diseaseName.trim()) return;
    
    try {
      setSearchLoading(true);
      const response = await apiService.getDiseaseInfo(diseaseName);
      setSelectedDisease(response);
    } catch (error) {
      console.error('Failed to search disease:', error);
      toast.error('Disease not found or error occurred');
      setSelectedDisease(null);
    } finally {
      setSearchLoading(false);
    }
  };

  // Removed filterDiseases function as it's not needed for this implementation

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchTerm.trim()) {
      searchDisease(searchTerm);
    }
  };

  const commonDiseases = [
    'Common Cold', 'Influenza', 'Diabetes', 'Hypertension', 'Asthma',
    'Migraine', 'Sinus Infection', 'Allergy', 'Pneumonia', 'Bronchitis',
    'Malaria', 'Typhoid', 'Dengue', 'Tuberculosis', 'Hepatitis'
  ];

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-medical-500"></div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white shadow-sm border-b">
        <div className="container mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <button
                onClick={() => window.history.back()}
                className="mr-4 p-2 hover:bg-gray-100 rounded-lg transition-colors"
              >
                <ChevronLeft size={24} className="text-gray-600" />
              </button>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">Disease Database</h1>
                <p className="text-gray-600">Search and explore disease information</p>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="container mx-auto px-6 py-6">
        {/* Search */}
        <div className="bg-white rounded-xl shadow-sm border p-6 mb-6">
          <form onSubmit={handleSearch} className="flex gap-4">
            <div className="flex-1 relative">
              <input
                type="text"
                placeholder="Search for a disease (e.g., Diabetes, Hypertension, Common Cold)..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full px-4 py-3 border border-gray-200 rounded-lg focus:ring-2 focus:ring-medical-500 focus:border-transparent bg-white"
              />
            </div>
            <button
              type="submit"
              disabled={searchLoading}
              className="px-6 py-3 bg-medical-600 text-white rounded-lg hover:bg-medical-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              {searchLoading ? 'Searching...' : 'Search'}
            </button>
          </form>
        </div>

        {/* Search Results */}
        {selectedDisease ? (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="bg-white rounded-xl shadow-sm border p-6 mb-6"
          >
            <div className="flex items-start justify-between mb-4">
              <div>
                <h2 className="text-2xl font-bold text-gray-900 mb-2">{selectedDisease.name}</h2>
                <p className="text-gray-600">{selectedDisease.description}</p>
              </div>
              <button
                onClick={() => setSelectedDisease(null)}
                className="text-gray-400 hover:text-gray-600"
              >
                ×
              </button>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              {/* Symptoms */}
              {selectedDisease.symptoms && selectedDisease.symptoms.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center">
                    <AlertTriangle size={20} className="mr-2 text-orange-500" />
                    Common Symptoms
                  </h3>
                  <div className="flex flex-wrap gap-2">
                    {selectedDisease.symptoms.map((symptom, index) => (
                      <span
                        key={index}
                        className="px-3 py-1 bg-orange-100 text-orange-800 rounded-full text-sm"
                      >
                        {symptom}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Precautions */}
              {selectedDisease.precautions && selectedDisease.precautions.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center">
                    <Info size={20} className="mr-2 text-blue-500" />
                    Precautions
                  </h3>
                  <ul className="space-y-2">
                    {selectedDisease.precautions.map((precaution, index) => (
                      <li key={index} className="flex items-start">
                        <span className="w-2 h-2 bg-blue-500 rounded-full mt-2 mr-3 flex-shrink-0"></span>
                        <span className="text-gray-700">{precaution}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Treatment Information */}
              {selectedDisease.treatments && selectedDisease.treatments.length > 0 && (
                <div className="md:col-span-2">
                  <h3 className="text-lg font-semibold text-gray-900 mb-3 flex items-center">
                    <Heart size={20} className="mr-2 text-green-500" />
                    Treatment Information
                  </h3>
                  <div className="grid md:grid-cols-2 gap-4">
                    <div>
                      <h4 className="font-medium text-gray-900 mb-2">Treatments:</h4>
                      <ul className="space-y-1">
                        {selectedDisease.treatments.map((treatment, index) => (
                          <li key={index} className="text-sm text-gray-600">
                            • {treatment}
                          </li>
                        ))}
                      </ul>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        ) : (
          /* Common Diseases Grid */
          <div>
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Common Diseases</h2>
            <p className="text-gray-600 mb-6">Click on a disease to search for detailed information</p>
            
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {commonDiseases.map((disease, index) => (
                <motion.div
                  key={disease}
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  transition={{ delay: index * 0.05 }}
                  onClick={() => {
                    setSearchTerm(disease);
                    searchDisease(disease);
                  }}
                  className="bg-white rounded-lg p-4 shadow-sm border hover:shadow-md transition-all cursor-pointer"
                >
                  <div className="flex items-center">
                    <div className="w-10 h-10 bg-medical-100 rounded-lg flex items-center justify-center mr-3">
                      <Heart size={20} className="text-medical-600" />
                    </div>
                    <div>
                      <h3 className="font-medium text-gray-900">{disease}</h3>
                      <p className="text-sm text-gray-500">Click to view details</p>
                    </div>
                  </div>
                </motion.div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DiseaseDatabase;
