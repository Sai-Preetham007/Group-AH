"""
Evaluation framework for Medical RAG Chatbot
"""
import json
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import numpy as np
from rouge_score import rouge_scorer
import sacrebleu
from config import settings

logger = logging.getLogger(__name__)


class MedicalRAGEvaluator:
    """Evaluation framework for medical RAG system"""
    
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.evaluation_metrics = settings.evaluation_metrics
    
    def evaluate_response(self, 
                         query: str, 
                         generated_response: str, 
                         reference_response: Optional[str] = None,
                         sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Evaluate a single response"""
        try:
            evaluation_result = {
                "query": query,
                "generated_response": generated_response,
                "metrics": {},
                "sources_evaluation": {},
                "overall_score": 0.0
            }
            
            # ROUGE scores
            if "rouge" in self.evaluation_metrics and reference_response:
                rouge_scores = self._calculate_rouge_scores(generated_response, reference_response)
                evaluation_result["metrics"]["rouge"] = rouge_scores
            
            # BLEU score
            if "bleu" in self.evaluation_metrics and reference_response:
                bleu_score = self._calculate_bleu_score(generated_response, reference_response)
                evaluation_result["metrics"]["bleu"] = bleu_score
            
            # Faithfulness (source-based evaluation)
            if "faithfulness" in self.evaluation_metrics and sources:
                faithfulness_score = self._calculate_faithfulness(generated_response, sources)
                evaluation_result["metrics"]["faithfulness"] = faithfulness_score
            
            # Source quality evaluation
            if sources:
                source_eval = self._evaluate_sources(sources)
                evaluation_result["sources_evaluation"] = source_eval
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(evaluation_result["metrics"])
            evaluation_result["overall_score"] = overall_score
            
            return evaluation_result
            
        except Exception as e:
            logger.error(f"Error evaluating response: {e}")
            return {
                "query": query,
                "generated_response": generated_response,
                "error": str(e),
                "overall_score": 0.0
            }
    
    def evaluate_batch(self, test_cases: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a batch of test cases"""
        try:
            results = []
            total_score = 0.0
            
            for test_case in test_cases:
                result = self.evaluate_response(
                    query=test_case["query"],
                    generated_response=test_case["generated_response"],
                    reference_response=test_case.get("reference_response"),
                    sources=test_case.get("sources", [])
                )
                results.append(result)
                total_score += result["overall_score"]
            
            # Calculate aggregate metrics
            avg_score = total_score / len(results) if results else 0.0
            
            # Calculate metric averages
            metric_averages = self._calculate_metric_averages(results)
            
            return {
                "total_cases": len(results),
                "average_score": avg_score,
                "metric_averages": metric_averages,
                "individual_results": results
            }
            
        except Exception as e:
            logger.error(f"Error evaluating batch: {e}")
            return {"error": str(e)}
    
    def _calculate_rouge_scores(self, generated: str, reference: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        try:
            scores = self.rouge_scorer.score(reference, generated)
            return {
                "rouge1": scores["rouge1"].fmeasure,
                "rouge2": scores["rouge2"].fmeasure,
                "rougeL": scores["rougeL"].fmeasure
            }
        except Exception as e:
            logger.error(f"Error calculating ROUGE scores: {e}")
            return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    
    def _calculate_bleu_score(self, generated: str, reference: str) -> float:
        """Calculate BLEU score"""
        try:
            # Tokenize and prepare for BLEU calculation
            generated_tokens = generated.split()
            reference_tokens = reference.split()
            
            # Calculate BLEU score
            bleu = sacrebleu.sentence_bleu(generated, [reference])
            return bleu.score / 100.0  # Normalize to 0-1 range
            
        except Exception as e:
            logger.error(f"Error calculating BLEU score: {e}")
            return 0.0
    
    def _calculate_faithfulness(self, response: str, sources: List[Dict[str, Any]]) -> float:
        """Calculate faithfulness score based on source alignment"""
        try:
            if not sources:
                return 0.0
            
            # Extract key information from sources
            source_content = " ".join([source.get("content", "") for source in sources])
            
            # Simple keyword overlap as faithfulness measure
            response_words = set(response.lower().split())
            source_words = set(source_content.lower().split())
            
            if not source_words:
                return 0.0
            
            overlap = len(response_words.intersection(source_words))
            faithfulness = overlap / len(source_words)
            
            return min(faithfulness, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating faithfulness: {e}")
            return 0.0
    
    def _evaluate_sources(self, sources: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate quality of retrieved sources"""
        try:
            if not sources:
                return {"count": 0, "average_similarity": 0.0, "source_types": []}
            
            # Count sources
            source_count = len(sources)
            
            # Calculate average similarity
            similarities = [source.get("similarity", 0) for source in sources]
            avg_similarity = np.mean(similarities) if similarities else 0.0
            
            # Get source types
            source_types = list(set(source.get("source", "Unknown") for source in sources))
            
            # Check for credible sources
            credible_sources = ["FDA", "WHO", "ClinicalTrials.gov", "PubMed"]
            has_credible = any(any(credible in source_type for credible in credible_sources) 
                             for source_type in source_types)
            
            return {
                "count": source_count,
                "average_similarity": avg_similarity,
                "source_types": source_types,
                "has_credible_sources": has_credible
            }
            
        except Exception as e:
            logger.error(f"Error evaluating sources: {e}")
            return {"error": str(e)}
    
    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall evaluation score"""
        try:
            scores = []
            
            # ROUGE score
            if "rouge" in metrics:
                rouge_scores = metrics["rouge"]
                avg_rouge = (rouge_scores["rouge1"] + rouge_scores["rouge2"] + rouge_scores["rougeL"]) / 3
                scores.append(avg_rouge)
            
            # BLEU score
            if "bleu" in metrics:
                scores.append(metrics["bleu"])
            
            # Faithfulness score
            if "faithfulness" in metrics:
                scores.append(metrics["faithfulness"])
            
            return np.mean(scores) if scores else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating overall score: {e}")
            return 0.0
    
    def _calculate_metric_averages(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate average metrics across all results"""
        try:
            metric_totals = {}
            metric_counts = {}
            
            for result in results:
                if "metrics" in result:
                    for metric_name, metric_value in result["metrics"].items():
                        if isinstance(metric_value, dict):
                            # Handle nested metrics like ROUGE
                            for sub_metric, sub_value in metric_value.items():
                                key = f"{metric_name}_{sub_metric}"
                                metric_totals[key] = metric_totals.get(key, 0) + sub_value
                                metric_counts[key] = metric_counts.get(key, 0) + 1
                        else:
                            # Handle simple metrics
                            metric_totals[metric_name] = metric_totals.get(metric_name, 0) + metric_value
                            metric_counts[metric_name] = metric_counts.get(metric_name, 0) + 1
            
            # Calculate averages
            averages = {}
            for metric_name in metric_totals:
                if metric_counts[metric_name] > 0:
                    averages[metric_name] = metric_totals[metric_name] / metric_counts[metric_name]
            
            return averages
            
        except Exception as e:
            logger.error(f"Error calculating metric averages: {e}")
            return {}
    
    def save_evaluation_results(self, results: Dict[str, Any], filename: str) -> None:
        """Save evaluation results to file"""
        try:
            output_path = Path("data/evaluation") / filename
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Evaluation results saved to {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving evaluation results: {e}")
    
    def load_test_cases(self, filename: str) -> List[Dict[str, Any]]:
        """Load test cases from file"""
        try:
            input_path = Path("data/evaluation") / filename
            
            if not input_path.exists():
                logger.warning(f"Test cases file {input_path} does not exist")
                return []
            
            with open(input_path, 'r', encoding='utf-8') as f:
                test_cases = json.load(f)
            
            logger.info(f"Loaded {len(test_cases)} test cases from {input_path}")
            return test_cases
            
        except Exception as e:
            logger.error(f"Error loading test cases: {e}")
            return []
