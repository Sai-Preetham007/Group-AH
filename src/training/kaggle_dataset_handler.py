"""
Kaggle dataset handler for disease-symptom dataset
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
import logging
from sklearn.model_selection import train_test_split
import requests
import zipfile
import os

logger = logging.getLogger(__name__)


class KaggleDiseaseDatasetHandler:
    """Handle Kaggle disease-symptom dataset"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.kaggle_dir = self.raw_dir / "kaggle_disease_dataset"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.kaggle_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def download_kaggle_dataset(self, kaggle_username: str = None, kaggle_key: str = None):
        """Download dataset from Kaggle using kaggle API"""
        try:
            # Check if kaggle is installed
            import kaggle
            from kaggle.api.kaggle_api_extended import KaggleApi
            
            # Initialize Kaggle API
            api = KaggleApi()
            api.authenticate()
            
            # Download dataset
            dataset_name = "itachi9604/disease-symptom-description-dataset"
            logger.info(f"Downloading dataset: {dataset_name}")
            
            api.dataset_download_files(
                dataset_name,
                path=str(self.kaggle_dir),
                unzip=True
            )
            
            logger.info(f"Dataset downloaded to {self.kaggle_dir}")
            return self.kaggle_dir
            
        except ImportError:
            logger.error("Kaggle API not installed. Please install with: pip install kaggle")
            raise
        except Exception as e:
            logger.error(f"Error downloading Kaggle dataset: {e}")
            raise
    
    def download_manual_dataset(self, dataset_url: str = None):
        """Manual download method if Kaggle API is not available"""
        try:
            # This would require manual download and upload
            logger.info("Please manually download the dataset from:")
            logger.info("https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset")
            logger.info(f"Extract files to: {self.kaggle_dir}")
            
            # Check if files exist
            expected_files = [
                "symptom_precaution.csv",
                "dataset.csv",  # Main dataset
                "symptom_Description.csv"
            ]
            
            missing_files = []
            for file in expected_files:
                if not (self.kaggle_dir / file).exists():
                    missing_files.append(file)
            
            if missing_files:
                logger.warning(f"Missing files: {missing_files}")
                logger.info("Please ensure all files are downloaded and extracted")
            
            return self.kaggle_dir
            
        except Exception as e:
            logger.error(f"Error in manual download setup: {e}")
            raise
    
    def load_disease_dataset(self) -> Dict[str, pd.DataFrame]:
        """Load all files from the disease dataset"""
        try:
            datasets = {}
            
            # Load main dataset
            main_file = self.kaggle_dir / "dataset.csv"
            if main_file.exists():
                datasets['main'] = pd.read_csv(main_file)
                logger.info(f"Loaded main dataset: {len(datasets['main'])} rows")
            
            # Load symptom precaution dataset
            precaution_file = self.kaggle_dir / "symptom_precaution.csv"
            if precaution_file.exists():
                datasets['precaution'] = pd.read_csv(precaution_file)
                logger.info(f"Loaded precaution dataset: {len(datasets['precaution'])} rows")
            
            # Load symptom description dataset
            description_file = self.kaggle_dir / "symptom_Description.csv"
            if description_file.exists():
                datasets['description'] = pd.read_csv(description_file)
                logger.info(f"Loaded description dataset: {len(datasets['description'])} rows")
            
            return datasets
            
        except Exception as e:
            logger.error(f"Error loading disease dataset: {e}")
            raise
    
    def preprocess_disease_data(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Preprocess and combine disease datasets"""
        try:
            processed_data = []
            
            # Process main dataset
            if 'main' in datasets:
                main_df = datasets['main']
                for _, row in main_df.iterrows():
                    # Create Q&A pairs from main dataset
                    if 'Disease' in row and 'Symptom' in row:
                        question = f"What are the symptoms of {row['Disease']}?"
                        answer = f"The symptoms of {row['Disease']} include: {row['Symptom']}"
                        processed_data.append({
                            'question': question,
                            'answer': answer,
                            'disease': row['Disease'],
                            'category': 'symptoms',
                            'source': 'main_dataset'
                        })
            
            # Process precaution dataset
            if 'precaution' in datasets:
                precaution_df = datasets['precaution']
                for _, row in precaution_df.iterrows():
                    if 'Disease' in row and 'Precaution' in row:
                        question = f"What precautions should be taken for {row['Disease']}?"
                        answer = f"For {row['Disease']}, the following precautions are recommended: {row['Precaution']}"
                        processed_data.append({
                            'question': question,
                            'answer': answer,
                            'disease': row['Disease'],
                            'category': 'precautions',
                            'source': 'precaution_dataset'
                        })
            
            # Process description dataset
            if 'description' in datasets:
                desc_df = datasets['description']
                for _, row in desc_df.iterrows():
                    if 'Disease' in row and 'Description' in row:
                        question = f"What is {row['Disease']}?"
                        answer = f"{row['Disease']} is: {row['Description']}"
                        processed_data.append({
                            'question': question,
                            'answer': answer,
                            'disease': row['Disease'],
                            'category': 'description',
                            'source': 'description_dataset'
                        })
            
            # Create DataFrame
            df = pd.DataFrame(processed_data)
            
            # Add metadata
            df['text_length'] = df['question'].str.len() + df['answer'].str.len()
            df['has_disease'] = df['disease'].notna()
            
            # Remove duplicates
            df = df.drop_duplicates(subset=['question', 'answer'])
            
            logger.info(f"Preprocessed dataset: {len(df)} Q&A pairs")
            logger.info(f"Categories: {df['category'].value_counts().to_dict()}")
            logger.info(f"Diseases covered: {df['disease'].nunique()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing disease data: {e}")
            raise
    
    def create_medical_qa_pairs(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive medical Q&A pairs"""
        try:
            qa_pairs = []
            
            for _, row in df.iterrows():
                disease = row['disease']
                category = row['category']
                
                # Create multiple question variations
                question_variations = self._create_question_variations(disease, category)
                
                for question in question_variations:
                    qa_pairs.append({
                        'question': question,
                        'answer': row['answer'],
                        'disease': disease,
                        'category': category,
                        'source': row['source'],
                        'context': f"Disease: {disease}, Category: {category}"
                    })
            
            qa_df = pd.DataFrame(qa_pairs)
            
            # Add training metadata
            qa_df['text_length'] = qa_df['question'].str.len() + qa_df['answer'].str.len()
            qa_df['is_training'] = True
            
            logger.info(f"Created {len(qa_df)} Q&A pairs for training")
            return qa_df
            
        except Exception as e:
            logger.error(f"Error creating Q&A pairs: {e}")
            raise
    
    def _create_question_variations(self, disease: str, category: str) -> List[str]:
        """Create multiple question variations for each disease-category pair"""
        variations = []
        
        if category == 'symptoms':
            variations = [
                f"What are the symptoms of {disease}?",
                f"How do I know if I have {disease}?",
                f"What signs indicate {disease}?",
                f"What are the warning signs of {disease}?"
            ]
        elif category == 'precautions':
            variations = [
                f"What precautions should I take for {disease}?",
                f"How can I prevent {disease}?",
                f"What should I avoid if I have {disease}?",
                f"How to manage {disease} safely?"
            ]
        elif category == 'description':
            variations = [
                f"What is {disease}?",
                f"Tell me about {disease}",
                f"Can you explain {disease}?",
                f"What does {disease} mean?"
            ]
        
        return variations
    
    def create_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create stratified train-test split by disease"""
        try:
            # Ensure we have diseases in both train and test
            train_df, test_df = train_test_split(
                df, 
                test_size=test_size, 
                stratify=df['disease'],
                random_state=42
            )
            
            logger.info(f"Train set: {len(train_df)} samples")
            logger.info(f"Test set: {len(test_df)} samples")
            logger.info(f"Train diseases: {train_df['disease'].nunique()}")
            logger.info(f"Test diseases: {test_df['disease'].nunique()}")
            
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error creating train-test split: {e}")
            raise
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save processed dataset"""
        try:
            output_path = self.processed_dir / filename
            df.to_csv(output_path, index=False)
            logger.info(f"Processed data saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving processed data: {e}")
            raise
    
    def load_processed_data(self, filename: str) -> pd.DataFrame:
        """Load processed dataset"""
        try:
            input_path = self.processed_dir / filename
            df = pd.read_csv(input_path)
            logger.info(f"Loaded processed data from {input_path}")
            return df
        except Exception as e:
            logger.error(f"Error loading processed data: {e}")
            raise
    
    def get_dataset_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get comprehensive dataset statistics"""
        try:
            stats = {
                'total_samples': len(df),
                'unique_diseases': df['disease'].nunique(),
                'categories': df['category'].value_counts().to_dict(),
                'sources': df['source'].value_counts().to_dict(),
                'avg_text_length': df['text_length'].mean(),
                'min_text_length': df['text_length'].min(),
                'max_text_length': df['text_length'].max(),
                'diseases_list': df['disease'].unique().tolist()
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating dataset statistics: {e}")
            return {}
    
    def create_disease_knowledge_base(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Create structured knowledge base by disease"""
        try:
            knowledge_base = {}
            
            for disease in df['disease'].unique():
                disease_data = df[df['disease'] == disease]
                
                knowledge_base[disease] = {
                    'symptoms': disease_data[disease_data['category'] == 'symptoms']['answer'].tolist(),
                    'precautions': disease_data[disease_data['category'] == 'precautions']['answer'].tolist(),
                    'description': disease_data[disease_data['category'] == 'description']['answer'].tolist(),
                    'total_qa_pairs': len(disease_data)
                }
            
            return knowledge_base
            
        except Exception as e:
            logger.error(f"Error creating knowledge base: {e}")
            return {}
