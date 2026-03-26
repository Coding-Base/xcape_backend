"""
Baseline Matching Algorithm
Manual parameter tuning for history matching
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from datetime import datetime
from scipy.optimize import minimize, differential_evolution

logger = logging.getLogger(__name__)


class BaselineMatcher:
    """Implements baseline/manual history matching algorithm"""
    
    def __init__(self):
        """Initialize baseline matcher"""
        self.match_history = []
        self.best_params = {}
        self.best_quality = 0.0
    
    def manual_match(
        self,
        observed_data: Dict,
        simulated_data: Dict,
        parameters: Dict,
        parameter_bounds: Dict = None
    ) -> Dict:
        """
        Perform manual baseline matching
        
        Args:
            observed_data: Observed production data from field
            simulated_data: Simulated production data
            parameters: Current reservoir parameters
            parameter_bounds: Bounds for parameter adjustment
            
        Returns:
            Matching results with quality metrics
        """
        logger.info("Starting baseline matching")
        
        # Validate inputs
        if not observed_data or not simulated_data:
            return {'status': 'failed', 'error': 'Missing data'}
        
        # Calculate match quality
        quality = self._calculate_match_quality(observed_data, simulated_data)
        
        # Extract time series
        obs_series = self._extract_time_series(observed_data)
        sim_series = self._extract_time_series(simulated_data)
        
        # Calculate error metrics
        errors = self._calculate_errors(obs_series, sim_series)
        
        results = {
            'status': 'completed',
            'match_quality': quality,
            'parameters': parameters,
            'errors': errors,
            'obs_vs_sim': {
                'observed': {k: v[:20] for k, v in obs_series.items()},
                'simulated': {k: v[:20] for k, v in sim_series.items()},
            },
            'recommendations': self._get_recommendations(errors)
        }
        
        # Store in history
        self.match_history.append({
            'timestamp': datetime.now().isoformat(),
            'parameters': parameters,
            'quality': quality,
            'errors': errors
        })
        
        # Update best match
        if quality > self.best_quality:
            self.best_quality = quality
            self.best_params = parameters.copy()
        
        return results
    
    def automated_tune(
        self,
        observed_data: Dict,
        forward_model_fn,
        initial_params: Dict,
        parameter_bounds: Dict = None,
        max_iterations: int = 100
    ) -> Dict:
        """
        Perform automated parameter tuning
        
        Args:
            observed_data: Observed production data
            forward_model_fn: Function to run forward model
            initial_params: Initial parameter guess
            parameter_bounds: Parameter bounds for optimization
            max_iterations: Maximum iterations
            
        Returns:
            Optimized parameters and match quality
        """
        logger.info("Starting automated parameter tuning")
        
        # Set default bounds if not provided
        if parameter_bounds is None:
            parameter_bounds = self._get_default_bounds(initial_params)
        
        # Define objective function
        def objective(params_array):
            params_dict = self._array_to_dict(params_array, initial_params)
            simulated = forward_model_fn(params_dict)
            quality = self._calculate_match_quality(observed_data, simulated)
            return -quality  # Minimize negative quality (i.e., maximize quality)
        
        # Get bounds
        bounds = [(parameter_bounds[k]['min'], parameter_bounds[k]['max']) 
                  for k in initial_params.keys()]
        
        # Run optimization
        try:
            result = differential_evolution(
                objective,
                bounds,
                maxiter=max_iterations,
                seed=42,
                workers=1,
                updating='immediate'
            )
            
            optimal_params = self._array_to_dict(result.x, initial_params)
            optimal_quality = -result.fun
            
            logger.info(f"Optimization converged with quality: {optimal_quality:.2f}%")
            
            return {
                'status': 'completed',
                'optimal_parameters': optimal_params,
                'match_quality': optimal_quality,
                'iterations': result.nit,
                'success': result.success,
                'message': 'Parameter tuning completed successfully'
            }
            
        except Exception as e:
            logger.error(f"Parameter tuning failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _calculate_match_quality(self, observed: Dict, simulated: Dict) -> float:
        """
        Calculate match quality as percentage (0-100)
        Higher is better
        """
        try:
            obs_series = self._extract_time_series(observed)
            sim_series = self._extract_time_series(simulated)
            
            if not obs_series or not sim_series:
                return 0.0
            
            # Calculate normalized RMSE for each series
            quality_scores = []
            
            for key in obs_series:
                if key not in sim_series:
                    continue
                
                obs_data = np.array(obs_series[key])
                sim_data = np.array(sim_series[key])
                
                if len(obs_data) != len(sim_data):
                    # Interpolate to same length
                    sim_data = np.interp(
                        np.linspace(0, 1, len(obs_data)),
                        np.linspace(0, 1, len(sim_data)),
                        sim_data
                    )
                
                # Normalized RMSE
                rmse = np.sqrt(np.mean((obs_data - sim_data) ** 2))
                nrmse = rmse / (np.max(obs_data) - np.min(obs_data)) if np.ptp(obs_data) > 0 else 1.0
                
                # Convert to quality score (0-100, inverted)
                quality = max(0, 100 * (1 - nrmse))
                quality_scores.append(quality)
            
            # Average quality across all series
            avg_quality = np.mean(quality_scores) if quality_scores else 0.0
            return float(avg_quality)
            
        except Exception as e:
            logger.error(f"Error calculating match quality: {e}")
            return 0.0
    
    def _extract_time_series(self, data: Dict) -> Dict:
        """Extract time series data from data dict"""
        series = {}
        
        if 'production_data' in data:
            prod = data['production_data']
            for key in ['oil', 'water', 'gas', 'pressure']:
                if key in prod:
                    series[key] = prod[key]
        
        return series
    
    def _calculate_errors(self, observed: Dict, simulated: Dict) -> Dict:
        """Calculate error metrics between observed and simulated"""
        errors = {}
        
        for key in observed:
            if key not in simulated:
                continue
            
            obs = np.array(observed[key])
            sim = np.array(simulated[key])
            
            # Interpolate if needed
            if len(obs) != len(sim):
                sim = np.interp(np.linspace(0, 1, len(obs)), 
                               np.linspace(0, 1, len(sim)), sim)
            
            mae = float(np.mean(np.abs(obs - sim)))
            rmse = float(np.sqrt(np.mean((obs - sim) ** 2)))
            mape = float(np.mean(np.abs((obs - sim) / (obs + 1e-6)))) * 100
            
            errors[key] = {
                'mae': mae,
                'rmse': rmse,
                'mape': mape
            }
        
        return errors
    
    def _get_recommendations(self, errors: Dict) -> List[str]:
        """Get recommendations for parameter adjustment"""
        recommendations = []
        
        for key, error_dict in errors.items():
            mape = error_dict.get('mape', 0)
            if mape > 20:
                recommendations.append(
                    f"{key.upper()}: High error ({mape:.1f}%). "
                    f"Adjust {key}-related parameters."
                )
        
        if not recommendations:
            recommendations.append("Match quality is good. Fine-tuning may provide marginal improvements.")
        
        return recommendations
    
    def _get_default_bounds(self, params: Dict) -> Dict:
        """Get default parameter bounds"""
        bounds = {}
        for key, value in params.items():
            # Default bounds are ±30% of current value
            bounds[key] = {
                'min': value * 0.7,
                'max': value * 1.3
            }
        return bounds
    
    def _array_to_dict(self, arr: np.ndarray, template: Dict) -> Dict:
        """Convert array to parameter dict"""
        return {k: float(arr[i]) for i, k in enumerate(template.keys())}
    
    def get_history(self) -> List[Dict]:
        """Get matching history"""
        return self.match_history
    
    def get_best_match(self) -> Dict:
        """Get best matching result"""
        return {
            'parameters': self.best_params,
            'quality': self.best_quality
        }
