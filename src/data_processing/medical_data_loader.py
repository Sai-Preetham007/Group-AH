"""
Medical data loading and preprocessing module
"""
import requests
import pandas as pd
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MedicalDataLoader:
    """Load and preprocess medical data from various sources"""
    
    def __init__(self, fda_base_url: str = "https://api.fda.gov", who_base_url: str = "https://www.who.int"):
        self.fda_base_url = fda_base_url
        self.who_base_url = who_base_url
        self.clinical_trials_base_url = "https://clinicaltrials.gov/api"
        
    def load_fda_drug_data(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Load drug data from FDA API"""
        try:
            url = f"{self.fda_base_url}/drug/label.json"
            params = {"limit": limit}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            drugs = data.get("results", [])
            
            logger.info(f"Loaded {len(drugs)} drug records from FDA")
            return drugs
            
        except Exception as e:
            logger.error(f"Error loading FDA drug data: {e}")
            return []
    
    def load_fda_adverse_events(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Load adverse events data from FDA"""
        try:
            url = f"{self.fda_base_url}/drug/event.json"
            params = {"limit": limit}
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            events = data.get("results", [])
            
            logger.info(f"Loaded {len(events)} adverse event records from FDA")
            return events
            
        except Exception as e:
            logger.error(f"Error loading FDA adverse events: {e}")
            return []
    
    def load_clinical_trials(self, condition: str = "", limit: int = 1000) -> List[Dict[str, Any]]:
        """Load clinical trials data"""
        try:
            url = f"{self.clinical_trials_base_url}/v2/studies"
            params = {
                "query.cond": condition,
                "format": "json",
                "limit": limit
            }
            
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            trials = data.get("studies", [])
            
            logger.info(f"Loaded {len(trials)} clinical trial records")
            return trials
            
        except Exception as e:
            logger.error(f"Error loading clinical trials: {e}")
            return []
    
    def preprocess_drug_data(self, drugs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess drug data for RAG system"""
        processed_drugs = []
        
        for drug in drugs:
            processed_drug = {
                "id": drug.get("id", ""),
                "generic_name": drug.get("generic_name", ""),
                "brand_name": drug.get("brand_name", ""),
                "indications": drug.get("indications_and_usage", ""),
                "dosage": drug.get("dosage_and_administration", ""),
                "contraindications": drug.get("contraindications", ""),
                "warnings": drug.get("warnings_and_cautions", ""),
                "adverse_reactions": drug.get("adverse_reactions", ""),
                "drug_interactions": drug.get("drug_interactions", ""),
                "pregnancy_category": drug.get("pregnancy", ""),
                "source": "FDA",
                "content": self._create_drug_content(drug)
            }
            processed_drugs.append(processed_drug)
        
        return processed_drugs
    
    def preprocess_clinical_trials(self, trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess clinical trials data"""
        processed_trials = []
        
        for trial in trials:
            protocol = trial.get("protocolSection", {})
            identification = protocol.get("identificationModule", {})
            status = protocol.get("statusModule", {})
            design = protocol.get("designModule", {})
            eligibility = protocol.get("eligibilityModule", {})
            
            processed_trial = {
                "id": identification.get("nctId", ""),
                "title": identification.get("briefTitle", ""),
                "description": identification.get("briefSummary", ""),
                "phase": design.get("phases", []),
                "status": status.get("overallStatus", ""),
                "conditions": eligibility.get("conditions", []),
                "interventions": design.get("interventions", []),
                "outcomes": design.get("primaryOutcomeMeasures", []),
                "source": "ClinicalTrials.gov",
                "content": self._create_trial_content(trial)
            }
            processed_trials.append(processed_trial)
        
        return processed_trials
    
    def _create_drug_content(self, drug: Dict[str, Any]) -> str:
        """Create comprehensive content string for drug"""
        content_parts = []
        
        if drug.get("generic_name"):
            content_parts.append(f"Generic Name: {drug['generic_name']}")
        
        if drug.get("brand_name"):
            content_parts.append(f"Brand Name: {drug['brand_name']}")
        
        if drug.get("indications_and_usage"):
            content_parts.append(f"Indications: {drug['indications_and_usage']}")
        
        if drug.get("dosage_and_administration"):
            content_parts.append(f"Dosage: {drug['dosage_and_administration']}")
        
        if drug.get("contraindications"):
            content_parts.append(f"Contraindications: {drug['contraindications']}")
        
        if drug.get("warnings_and_cautions"):
            content_parts.append(f"Warnings: {drug['warnings_and_cautions']}")
        
        if drug.get("adverse_reactions"):
            content_parts.append(f"Adverse Reactions: {drug['adverse_reactions']}")
        
        return "\n".join(content_parts)
    
    def _create_trial_content(self, trial: Dict[str, Any]) -> str:
        """Create comprehensive content string for clinical trial"""
        content_parts = []
        
        protocol = trial.get("protocolSection", {})
        identification = protocol.get("identificationModule", {})
        design = protocol.get("designModule", {})
        eligibility = protocol.get("eligibilityModule", {})
        
        if identification.get("briefTitle"):
            content_parts.append(f"Title: {identification['briefTitle']}")
        
        if identification.get("briefSummary"):
            content_parts.append(f"Summary: {identification['briefSummary']}")
        
        if design.get("phases"):
            content_parts.append(f"Phase: {', '.join(design['phases'])}")
        
        if eligibility.get("conditions"):
            content_parts.append(f"Conditions: {', '.join(eligibility['conditions'])}")
        
        if design.get("interventions"):
            interventions = [interv.get("name", "") for interv in design["interventions"]]
            content_parts.append(f"Interventions: {', '.join(interventions)}")
        
        return "\n".join(content_parts)
    
    def save_processed_data(self, data: List[Dict[str, Any]], filename: str) -> None:
        """Save processed data to file"""
        output_path = Path("data/processed") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved processed data to {output_path}")
    
    def load_processed_data(self, filename: str) -> List[Dict[str, Any]]:
        """Load processed data from file"""
        input_path = Path("data/processed") / filename
        
        if not input_path.exists():
            logger.warning(f"File {input_path} does not exist")
            return []
        
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"Loaded processed data from {input_path}")
        return data
