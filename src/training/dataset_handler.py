"""
Dataset handling and preprocessing for ML training
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import json
import logging
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import requests
import zipfile
import tarfile

logger = logging.getLogger(__name__)


class MedicalDatasetHandler:
    """Handle medical datasets for training"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        self.models_dir = self.data_dir / "models"
        
        # Create directories
        for dir_path in [self.raw_dir, self.processed_dir, self.models_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def download_dataset(self, url: str, filename: str = None) -> Path:
        """Download dataset from URL"""
        try:
            if filename is None:
                filename = url.split("/")[-1]
            
            file_path = self.raw_dir / filename
            
            if file_path.exists():
                logger.info(f"Dataset already exists: {file_path}")
                return file_path
            
            logger.info(f"Downloading dataset from {url}")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Dataset downloaded to {file_path}")
            return file_path
            
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            raise
    
    def extract_dataset(self, file_path: Path, extract_dir: str = None) -> Path:
        """Extract compressed dataset"""
        try:
            if extract_dir is None:
                extract_dir = file_path.stem
            extract_path = self.raw_dir / extract_dir
            extract_path.mkdir(exist_ok=True)
            
            if file_path.suffix == '.zip':
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(extract_path)
            elif file_path.suffix in ['.tar', '.gz']:
                with tarfile.open(file_path, 'r:*') as tar_ref:
                    tar_ref.extractall(extract_path)
            else:
                logger.warning(f"Unknown file format: {file_path.suffix}")
                return file_path
            
            logger.info(f"Dataset extracted to {extract_path}")
            return extract_path
            
        except Exception as e:
            logger.error(f"Error extracting dataset: {e}")
            raise
    
    def load_medical_qa_dataset(self, file_path: Path) -> pd.DataFrame:
        """Load medical Q&A dataset"""
        try:
            if file_path.suffix == '.json':
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
            elif file_path.suffix == '.csv':
                df = pd.read_csv(file_path)
            elif file_path.suffix == '.tsv':
                df = pd.read_csv(file_path, sep='\t')
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
            
            logger.info(f"Loaded dataset with {len(df)} samples")
            return df
            
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def preprocess_medical_qa(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess medical Q&A dataset"""
        try:
            # Standardize column names
            column_mapping = {
                'question': ['question', 'q', 'query', 'input'],
                'answer': ['answer', 'a', 'response', 'output'],
                'context': ['context', 'passage', 'text'],
                'category': ['category', 'type', 'label']
            }
            
            for target_col, possible_cols in column_mapping.items():
                for col in possible_cols:
                    if col in df.columns:
                        df[target_col] = df[col]
                        break
            
            # Clean text data
            if 'question' in df.columns:
                df['question'] = df['question'].astype(str).str.strip()
            if 'answer' in df.columns:
                df['answer'] = df['answer'].astype(str).str.strip()
            if 'context' in df.columns:
                df['context'] = df['context'].astype(str).str.strip()
            
            # Remove empty rows
            df = df.dropna(subset=['question', 'answer'])
            
            # Add metadata
            df['text_length'] = df['question'].str.len() + df['answer'].str.len()
            df['has_context'] = df['context'].notna() if 'context' in df.columns else False
            
            logger.info(f"Preprocessed dataset: {len(df)} samples")
            return df
            
        except Exception as e:
            logger.error(f"Error preprocessing dataset: {e}")
            raise
    
    def create_train_test_split(self, df: pd.DataFrame, test_size: float = 0.2, 
                              stratify_col: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Create train-test split"""
        try:
            if stratify_col and stratify_col in df.columns:
                train_df, test_df = train_test_split(
                    df, test_size=test_size, stratify=df[stratify_col], random_state=42
                )
            else:
                train_df, test_df = train_test_split(
                    df, test_size=test_size, random_state=42
                )
            
            logger.info(f"Train set: {len(train_df)} samples, Test set: {len(test_df)} samples")
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


class MedicalQADataset(Dataset):
    """PyTorch Dataset for medical Q&A"""
    
    def __init__(self, df: pd.DataFrame, tokenizer, max_length: int = 512):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Create input text
        question = str(row['question'])
        context = str(row.get('context', ''))
        
        if context and context != 'nan':
            input_text = f"Context: {context}\nQuestion: {question}"
        else:
            input_text = f"Question: {question}"
        
        # Tokenize
        encoding = self.tokenizer(
            input_text,
            str(row['answer']),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze()
        }


def create_data_loader(df: pd.DataFrame, tokenizer, batch_size: int = 8, 
                      max_length: int = 512, shuffle: bool = True) -> DataLoader:
    """Create PyTorch DataLoader"""
    dataset = MedicalQADataset(df, tokenizer, max_length)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
