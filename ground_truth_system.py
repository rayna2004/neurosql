# ============================================================================
# VERIFIABLE GROUND TRUTH SYSTEM
# ============================================================================

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from typing import Dict, List, Tuple, Optional
import json
import pickle
from datetime import datetime, timedelta
import hashlib
import sqlite3
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# ============================================================================
# GROUND TRUTH DATASETS
# ============================================================================

class NeuroTruthDataset:
    """Curated ground truth dataset for neuroscience"""
    
    def __init__(self):
        self.datasets = self._load_datasets()
        self.validation_results = []
        
    def _load_datasets(self) -> Dict[str, List[Dict]]:
        """Load ground truth datasets"""
        
        # Dataset 1: Core neuroscience facts (high confidence)
        core_facts = [
            {'query': 'dopamine modulates reward', 'truth': 1, 'confidence': 0.95, 'evidence': 45},
            {'query': 'serotonin regulates mood', 'truth': 1, 'confidence': 0.90, 'evidence': 38},
            {'query': 'glutamate is excitatory neurotransmitter', 'truth': 1, 'confidence': 0.98, 'evidence': 52},
            {'query': 'gaba is inhibitory neurotransmitter', 'truth': 1, 'confidence': 0.97, 'evidence': 48},
            {'query': 'hippocampus supports memory formation', 'truth': 1, 'confidence': 0.94, 'evidence': 67},
            {'query': 'prefrontal cortex regulates executive function', 'truth': 1, 'confidence': 0.91, 'evidence': 41},
            {'query': 'amygdala processes emotional responses', 'truth': 1, 'confidence': 0.93, 'evidence': 56},
            {'query': 'striatum mediates movement coordination', 'truth': 1, 'confidence': 0.88, 'evidence': 34},
            {'query': 'dopamine cures all neurological disorders', 'truth': 0, 'confidence': 0.99, 'evidence': 5},
            {'query': 'serotonin directly causes happiness', 'truth': 0, 'confidence': 0.95, 'evidence': 2},
            {'query': 'glutamate is highly addictive', 'truth': 0, 'confidence': 0.96, 'evidence': 3},
            {'query': 'gaba causes memory loss', 'truth': 0, 'confidence': 0.94, 'evidence': 4},
        ]
        
        # Dataset 2: Controversial/debated relationships (medium confidence)
        debated_facts = [
            {'query': 'dopamine causes addiction', 'truth': 0.7, 'confidence': 0.75, 'evidence': 28},
            {'query': 'serotonin deficiency causes depression', 'truth': 0.8, 'confidence': 0.78, 'evidence': 31},
            {'query': 'glutamate causes excitotoxicity', 'truth': 0.9, 'confidence': 0.85, 'evidence': 42},
            {'query': 'gaba regulates anxiety', 'truth': 0.85, 'confidence': 0.82, 'evidence': 37},
        ]
        
        # Dataset 3: Emerging research (low confidence)
        emerging_facts = [
            {'query': 'gut microbiome affects brain function', 'truth': 0.6, 'confidence': 0.65, 'evidence': 19},
            {'query': 'neuroinflammation causes depression', 'truth': 0.5, 'confidence': 0.60, 'evidence': 15},
            {'query': 'optogenetics can restore vision', 'truth': 0.7, 'confidence': 0.70, 'evidence': 22},
        ]
        
        return {
            'core_facts': core_facts,
            'debated_facts': debated_facts,
            'emerging_facts': emerging_facts,
            'all_facts': core_facts + debated_facts + emerging_facts
        }
    
    def evaluate_model(self, model_predictions: Dict[str, float]) -> Dict:
        """Evaluate model predictions against ground truth"""
        
        # Match predictions with ground truth
        matched_data = []
        for item in self.datasets['all_facts']:
            query = item['query']
            if query in model_predictions:
                matched_data.append({
                    'query': query,
                    'true_value': item['truth'],
                    'true_confidence': item['confidence'],
                    'predicted_value': model_predictions[query],
                    'evidence': item['evidence']
                })
        
        if not matched_data:
            return {'error': 'No matching predictions found'}
        
        # Convert to arrays
        y_true = np.array([d['true_value'] for d in matched_data])
        y_pred = np.array([d['predicted_value'] for d in matched_data])
        y_true_binary = (y_true > 0.5).astype(int)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        # Calculate metrics
        metrics = {}
        
        # Brier Score (probability calibration)
        metrics['brier_score'] = brier_score_loss(y_true, y_pred)
        
        # Log Loss
        epsilon = 1e-15
        y_pred_clipped = np.clip(y_pred, epsilon, 1 - epsilon)
        metrics['log_loss'] = log_loss(y_true, y_pred_clipped)
        
        # ROC AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true_binary, y_pred)
        except:
            metrics['roc_auc'] = 0.5
        
        # Accuracy
        metrics['accuracy'] = np.mean(y_true_binary == y_pred_binary)
        
        # Precision, Recall, F1
        tp = np.sum((y_true_binary == 1) & (y_pred_binary == 1))
        fp = np.sum((y_true_binary == 0) & (y_pred_binary == 1))
        fn = np.sum((y_true_binary == 1) & (y_pred_binary == 0))
        
        metrics['precision'] = tp / (tp + fp) if (tp + fp) > 0 else 0
        metrics['recall'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['f1_score'] = 2 * metrics['precision'] * metrics['recall'] / (metrics['precision'] + metrics['recall']) if (metrics['precision'] + metrics['recall']) > 0 else 0
        
        # Calibration curve
        prob_true, prob_pred = calibration_curve(y_true_binary, y_pred, n_bins=10)
        metrics['calibration_curve'] = {
            'prob_true': prob_true.tolist(),
            'prob_pred': prob_pred.tolist()
        }
        
        # Expected Calibration Error (ECE)
        bin_edges = np.linspace(0, 1, 11)
        ece = 0
        for i in range(10):
            mask = (y_pred >= bin_edges[i]) & (y_pred <= bin_edges[i + 1])
            if mask.any():
                bin_acc = y_true_binary[mask].mean()
                bin_conf = y_pred[mask].mean()
                bin_weight = mask.mean()
                ece += np.abs(bin_acc - bin_conf) * bin_weight
        
        metrics['expected_calibration_error'] = ece
        
        # Confidence intervals for accuracy
        n = len(matched_data)
        accuracy_se = np.sqrt(metrics['accuracy'] * (1 - metrics['accuracy']) / n)
        metrics['accuracy_95ci'] = (
            metrics['accuracy'] - 1.96 * accuracy_se,
            metrics['accuracy'] + 1.96 * accuracy_se
        )
        
        # By evidence strength
        strong_evidence = [d for d in matched_data if d['evidence'] >= 30]
        weak_evidence = [d for d in matched_data if d['evidence'] < 30]
        
        if strong_evidence:
            y_true_strong = np.array([d['true_value'] for d in strong_evidence])
            y_pred_strong = np.array([d['predicted_value'] for d in strong_evidence])
            metrics['accuracy_strong_evidence'] = np.mean((y_true_strong > 0.5) == (y_pred_strong > 0.5))
        
        if weak_evidence:
            y_true_weak = np.array([d['true_value'] for d in weak_evidence])
            y_pred_weak = np.array([d['predicted_value'] for d in weak_evidence])
            metrics['accuracy_weak_evidence'] = np.mean((y_true_weak > 0.5) == (y_pred_weak > 0.5))
        
        # Store evaluation results
        evaluation = {
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(matched_data),
            'metrics': metrics,
            'matched_queries': [d['query'] for d in matched_data],
            'calibration_status': 'WELL_CALIBRATED' if ece < 0.1 else 'NEEDS_CALIBRATION'
        }
        
        self.validation_results.append(evaluation)
        
        return evaluation
    
    def generate_calibration_report(self) -> Dict:
        """Generate comprehensive calibration report"""
        if not self.validation_results:
            return {'error': 'No validation results available'}
        
        latest = self.validation_results[-1]
        
        report = {
            'report_id': f"cal_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'generated_at': datetime.now().isoformat(),
            'summary': {
                'sample_size': latest['sample_size'],
                'brier_score': latest['metrics']['brier_score'],
                'expected_calibration_error': latest['metrics']['expected_calibration_error'],
                'accuracy': latest['metrics']['accuracy'],
                'calibration_status': latest['calibration_status']
            },
            'detailed_metrics': latest['metrics'],
            'recommendations': self._generate_calibration_recommendations(latest['metrics'])
        }
        
        return report
    
    def _generate_calibration_recommendations(self, metrics: Dict) -> List[str]:
        """Generate calibration recommendations"""
        recommendations = []
        
        if metrics['brier_score'] > 0.25:
            recommendations.append("High Brier score indicates poor probability estimates. Consider recalibrating the model.")
        
        if metrics['expected_calibration_error'] > 0.1:
            recommendations.append("High calibration error detected. Apply Platt scaling or isotonic regression for calibration.")
        
        if metrics['accuracy'] < 0.7:
            recommendations.append("Accuracy below 70%. Consider retraining with more data or feature engineering.")
        
        if 'accuracy_strong_evidence' in metrics and 'accuracy_weak_evidence' in metrics:
            diff = abs(metrics['accuracy_strong_evidence'] - metrics['accuracy_weak_evidence'])
            if diff > 0.2:
                recommendations.append(f"Large accuracy difference ({diff:.2f}) between strong and weak evidence. Model may be overfitting to evidence strength.")
        
        if not recommendations:
            recommendations.append("Model calibration is within acceptable ranges. Continue monitoring.")
        
        return recommendations

# ============================================================================
# CONFIDENCE CALIBRATION ENGINE
# ============================================================================

class ConfidenceCalibrationEngine:
    """Real confidence calibration engine"""
    
    def __init__(self, calibration_data_path: str = "data/calibration_data.pkl"):
        self.calibration_data_path = calibration_data_path
        self.calibration_model = None
        self.calibration_history = []
        self._load_or_train_calibration_model()
    
    def _load_or_train_calibration_model(self):
        """Load or train calibration model"""
        try:
            if Path(self.calibration_data_path).exists():
                with open(self.calibration_data_path, 'rb') as f:
                    self.calibration_model = pickle.load(f)
                logger.info("Loaded existing calibration model")
            else:
                self._train_calibration_model()
        except:
            self._train_calibration_model()
    
    def _train_calibration_model(self):
        """Train calibration model on synthetic data"""
        # Generate synthetic calibration data
        np.random.seed(42)
        
        # True probabilities
        n_samples = 1000
        y_true = np.random.binomial(1, 0.7, n_samples)
        
        # Raw model scores (with some miscalibration)
        y_raw = np.zeros(n_samples)
        for i in range(n_samples):
            if y_true[i] == 1:
                # Positive cases: model is overconfident
                y_raw[i] = np.random.beta(8, 2)
            else:
                # Negative cases: model is underconfident
                y_raw[i] = np.random.beta(2, 8)
        
        # Train calibration model (Platt scaling)
        X = y_raw.reshape(-1, 1)
        
        from sklearn.linear_model import LogisticRegression
        from sklearn.calibration import CalibratedClassifierCV
        
        lr = LogisticRegression()
        self.calibration_model = CalibratedClassifierCV(lr, method='sigmoid', cv=3)
        self.calibration_model.fit(X, y_true)
        
        # Save model
        Path("data").mkdir(exist_ok=True)
        with open(self.calibration_data_path, 'wb') as f:
            pickle.dump(self.calibration_model, f)
        
        logger.info("Trained and saved new calibration model")
    
    def calibrate_confidence(self, raw_score: float, 
                           evidence_count: int,
                           evidence_quality: float,
                           inference_type: str) -> Dict:
        """Calibrate raw confidence score"""
        
        start_time = datetime.now()
        
        # Base calibration using trained model
        if self.calibration_model:
            calibrated = self.calibration_model.predict_proba([[raw_score]])[0, 1]
        else:
            calibrated = raw_score  # Fallback
        
        # Adjust for evidence
        evidence_factor = self._calculate_evidence_factor(evidence_count, evidence_quality)
        calibrated = calibrated * evidence_factor
        
        # Adjust for inference type
        type_factor = self._get_inference_type_factor(inference_type)
        calibrated = calibrated * type_factor
        
        # Ensure bounds
        calibrated = np.clip(calibrated, 0.0, 1.0)
        
        # Calculate uncertainty
        uncertainty = self._calculate_uncertainty(calibrated, evidence_count, evidence_quality)
        
        # Log calibration
        calibration_entry = {
            'timestamp': datetime.now().isoformat(),
            'raw_score': raw_score,
            'calibrated_score': calibrated,
            'evidence_count': evidence_count,
            'evidence_quality': evidence_quality,
            'inference_type': inference_type,
            'uncertainty': uncertainty,
            'calibration_time_ms': (datetime.now() - start_time).total_seconds() * 1000
        }
        
        self.calibration_history.append(calibration_entry)
        
        return {
            'raw_confidence': raw_score,
            'calibrated_confidence': calibrated,
            'calibration_method': 'platt_scaling',
            'uncertainty': uncertainty,
            'evidence_adjustment': evidence_factor,
            'type_adjustment': type_factor,
            'confidence_interval': self._calculate_confidence_interval(calibrated, evidence_count)
        }
    
    def _calculate_evidence_factor(self, count: int, quality: float) -> float:
        """Calculate evidence adjustment factor"""
        if count >= 5 and quality >= 0.8:
            return 1.0  # Strong evidence, no adjustment
        elif count >= 3:
            return 0.9  # Moderate evidence, slight conservative adjustment
        elif count >= 1:
            return 0.7  # Weak evidence, conservative adjustment
        else:
            return 0.5  # No evidence, heavy regularization
    
    def _get_inference_type_factor(self, inference_type: str) -> float:
        """Get adjustment factor for inference type"""
        factors = {
            'direct_evidence': 1.0,
            'transitive': 0.9,
            'probabilistic': 0.85,
            'similarity': 0.7,
            'default': 0.8
        }
        return factors.get(inference_type, factors['default'])
    
    def _calculate_uncertainty(self, confidence: float, 
                             evidence_count: int, 
                             evidence_quality: float) -> Dict:
        """Calculate uncertainty metrics"""
        
        # Aleatoric uncertainty (data uncertainty)
        aleatoric = confidence * (1 - confidence)
        
        # Epistemic uncertainty (model uncertainty)
        if evidence_count >= 5:
            epistemic = 0.1
        elif evidence_count >= 3:
            epistemic = 0.3
        else:
            epistemic = 0.5
        
        # Evidence uncertainty
        evidence_uncertainty = 1.0 - evidence_quality
        
        # Combined uncertainty
        combined = (aleatoric * 0.4 + epistemic * 0.3 + evidence_uncertainty * 0.3)
        
        # Classification
        if combined < 0.2:
            level = 'LOW'
        elif combined < 0.4:
            level = 'MEDIUM'
        else:
            level = 'HIGH'
        
        return {
            'score': combined,
            'level': level,
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'evidence_uncertainty': evidence_uncertainty
        }
    
    def _calculate_confidence_interval(self, probability: float, n: int) -> Tuple[float, float]:
        """Calculate 95% confidence interval for probability"""
        if n <= 0:
            return (0.0, 1.0)
        
        se = np.sqrt(probability * (1 - probability) / n)
        ci_lower = max(0.0, probability - 1.96 * se)
        ci_upper = min(1.0, probability + 1.96 * se)
        
        return (float(ci_lower), float(ci_upper))
    
    def evaluate_calibration_performance(self) -> Dict:
        """Evaluate calibration performance"""
        if len(self.calibration_history) < 10:
            return {'message': 'Insufficient calibration data'}
        
        # Extract data
        raw_scores = [h['raw_score'] for h in self.calibration_history]
        calibrated_scores = [h['calibrated_score'] for h in self.calibration_history]
        
        # Calculate calibration metrics
        mse = np.mean((np.array(raw_scores) - np.array(calibrated_scores)) ** 2)
        mae = np.mean(np.abs(np.array(raw_scores) - np.array(calibrated_scores)))
        
        # Calculate calibration curve
        from sklearn.calibration import calibration_curve
        # For this demo, we don't have true labels for calibration history
        # In production, we would store true labels with calibration data
        
        return {
            'calibration_samples': len(self.calibration_history),
            'mean_squared_error': mse,
            'mean_absolute_error': mae,
            'calibration_trend': 'conservative' if np.mean(calibrated_scores) < np.mean(raw_scores) else 'overconfident',
            'recommendations': self._generate_calibration_recommendations(mse, mae)
        }
    
    def _generate_calibration_recommendations(self, mse: float, mae: float) -> List[str]:
        """Generate calibration recommendations"""
        recommendations = []
        
        if mse > 0.05:
            recommendations.append("High calibration error detected. Consider retraining calibration model.")
        
        if mae > 0.1:
            recommendations.append("Large absolute calibration errors. Review evidence quality assessment.")
        
        if len(self.calibration_history) < 100:
            recommendations.append("Limited calibration data. Collect more labeled data for better calibration.")
        
        if not recommendations:
            recommendations.append("Calibration performance within acceptable ranges.")
        
        return recommendations

# ============================================================================
# ERROR ESTIMATION SYSTEM
# ============================================================================

class ErrorEstimationSystem:
    """Estimate errors and uncertainties"""
    
    def __init__(self):
        self.error_log = []
        self.confidence_intervals = {}
    
    def estimate_error(self, confidence: float, 
                      evidence_count: int, 
                      evidence_quality: float,
                      inference_type: str) -> Dict:
        """Estimate error for an inference"""
        
        # Base error rate from inference type
        base_error_rates = {
            'direct_evidence': 0.05,
            'transitive': 0.15,
            'probabilistic': 0.10,
            'similarity': 0.25,
            'default': 0.20
        }
        
        base_error = base_error_rates.get(inference_type, base_error_rates['default'])
        
        # Adjust for evidence
        if evidence_count >= 5 and evidence_quality >= 0.8:
            evidence_factor = 0.5
        elif evidence_count >= 3:
            evidence_factor = 0.7
        elif evidence_count >= 1:
            evidence_factor = 0.9
        else:
            evidence_factor = 1.2
        
        adjusted_error = base_error * evidence_factor
        
        # Adjust for confidence (inverse relationship)
        confidence_factor = 1.0 - confidence
        final_error = adjusted_error * (0.5 + 0.5 * confidence_factor)
        
        # Calculate confidence interval
        if evidence_count > 0:
            se = np.sqrt(confidence * (1 - confidence) / evidence_count)
            ci_lower = max(0, confidence - 2 * se)
            ci_upper = min(1, confidence + 2 * se)
        else:
            se = 0.5
            ci_lower, ci_upper = 0, 1
        
        error_estimate = {
            'estimated_error_rate': final_error,
            'base_error_rate': base_error,
            'evidence_adjustment': evidence_factor,
            'confidence_adjustment': confidence_factor,
            'standard_error': se,
            'confidence_interval': (ci_lower, ci_upper),
            'margin_of_error': 2 * se,
            'reliability': 'HIGH' if final_error < 0.1 else 'MEDIUM' if final_error < 0.3 else 'LOW'
        }
        
        # Log error estimate
        self.error_log.append({
            'timestamp': datetime.now().isoformat(),
            'confidence': confidence,
            'evidence_count': evidence_count,
            'estimated_error': final_error,
            'reliability': error_estimate['reliability']
        })
        
        return error_estimate
    
    def get_error_statistics(self) -> Dict:
        """Get error estimation statistics"""
        if not self.error_log:
            return {'message': 'No error estimates available'}
        
        errors = [e['estimated_error'] for e in self.error_log]
        confidences = [e['confidence'] for e in self.error_log]
        
        return {
            'total_estimates': len(self.error_log),
            'mean_error_rate': np.mean(errors),
            'median_error_rate': np.median(errors),
            'std_error_rate': np.std(errors),
            'mean_confidence': np.mean(confidences),
            'error_confidence_correlation': np.corrcoef(errors, confidences)[0, 1] if len(errors) > 1 else 0,
            'reliability_distribution': {
                'HIGH': sum(1 for e in self.error_log if e['reliability'] == 'HIGH'),
                'MEDIUM': sum(1 for e in self.error_log if e['reliability'] == 'MEDIUM'),
                'LOW': sum(1 for e in self.error_log if e['reliability'] == 'LOW')
            }
        }
