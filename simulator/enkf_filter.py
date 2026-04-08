"""
Ensemble Kalman Filter (EnKF) Algorithm
Advanced automated calibration using Ensemble Kalman Filter
"""

import logging
from typing import Dict, List, Tuple, Optional, Callable, Union
import numpy as np
from datetime import datetime
from scipy import stats

logger = logging.getLogger(__name__)


class EnKFFilter:
    """Implements Ensemble Kalman Filter for parameter estimation"""
    
    def __init__(self, ensemble_size: int = 50):
        """
        Initialize EnKF Filter
        
        Args:
            ensemble_size: Number of ensemble members
        """
        self.ensemble_size = ensemble_size
        self.ensemble_history = []
        self.parameter_history = []
    
    def run_enkf(
        self,
        observed_data: Dict,
        forward_model_fn: Callable,
        initial_ensemble: np.ndarray,
        measurement_error: Optional[Dict] = None,
        num_iterations: int = 10,
        progress_callback: Optional[Callable] = None
    ) -> Dict:
        """
        Run Ensemble Kalman Filter
        
        Args:
            observed_data: Observed measurements
            forward_model_fn: Forward model function
            initial_ensemble: Initial ensemble of parameters (N x M)
                             N = ensemble size, M = number of parameters
            measurement_error: Measurement error std dev for each observation
            num_iterations: Number of EnKF iterations
            progress_callback: Function for progress updates
            
        Returns:
            EnKF results with updated parameters and uncertainty
        """
        logger.info(f"Starting EnKF with {self.ensemble_size} ensemble members")
        
        # Initialize
        ensemble = initial_ensemble.copy()
        observations = self._extract_observations(observed_data)
        
        if measurement_error is None:
            # Default: 5% measurement error
            measurement_error = {k: np.max(v) * 0.05 
                                for k, v in observations.items()}
        
        best_quality = 0.0
        best_params = None
        
        # Main EnKF loop
        for iteration in range(num_iterations):
            logger.info(f"EnKF iteration {iteration + 1}/{num_iterations}")
            
            # Run forward model for all ensemble members
            simulated_data = []
            for i, params in enumerate(ensemble):
                params_dict = self._array_to_params(params)
                sim_output = forward_model_fn(params_dict)
                simulated_data.append(self._extract_observations(sim_output))
            
            # Calculate match quality
            quality = self._calculate_ensemble_quality(observations, simulated_data)
            
            if quality > best_quality:
                best_quality = quality
                best_params = np.mean(ensemble, axis=0)
            
            # Update progress
            if progress_callback:
                progress_callback(
                    (iteration + 1) / num_iterations * 100,
                    f"Iteration {iteration + 1}: Quality = {quality:.2f}%"
                )
            
            # Perform update step
            if iteration < num_iterations - 1:  # No update on last iteration
                ensemble = self._update_ensemble(
                    ensemble,
                    observations,
                    simulated_data,
                    measurement_error
                )
            
            # Store history
            self.ensemble_history.append({
                'iteration': iteration,
                'ensemble': ensemble.copy(),
                'quality': quality
            })
        
        # Calculate statistics
        mean_params = np.mean(ensemble, axis=0)
        std_params = np.std(ensemble, axis=0)
        
        results = {
            'status': 'completed',
            'iterations': num_iterations,
            'ensemble_size': self.ensemble_size,
            'mean_parameters': mean_params.tolist(),
            'std_parameters': std_params.tolist(),
            'best_quality': float(best_quality),
            'final_ensemble': ensemble.tolist(),
            'covariance_matrix': np.cov(ensemble.T).tolist(),
            'message': 'EnKF calibration completed'
        }
        
        logger.info(f"EnKF completed. Best quality: {best_quality:.2f}%")
        
        return results
    
    def _update_ensemble(
        self,
        ensemble: np.ndarray,
        observations: Dict,
        simulated_data: List[Dict],
        measurement_error: Dict
    ) -> np.ndarray:
        """
        Update ensemble using Kalman update equations
        
        Args:
            ensemble: Current ensemble (N x M)
            observations: Dictionary of observed data
            simulated_data: List of simulated dictionaries
            measurement_error: Measurement error dict
            
        Returns:
            Updated ensemble
        """
        N = self.ensemble_size  # Ensemble size
        M = ensemble.shape[1]   # Number of parameters
        
        # Concatenate all observations into single vector
        obs_vec = np.concatenate([observations[k] for k in observations.keys()])
        H_obs = len(obs_vec)  # Total number of observations
        
        # Build simulated observation matrix (H x N)
        H_matrix = []
        for sim_dict in simulated_data:
            sim_vec = np.concatenate([sim_dict.get(k, np.zeros(len(observations[k]))) 
                                     for k in observations.keys()])
            H_matrix.append(sim_vec)
        H_matrix = np.array(H_matrix).T  # H x N
        
        # Calculate mean simulated observations
        h_mean = np.mean(H_matrix, axis=1, keepdims=True)
        
        # Anomaly matrices
        Y_anom = H_matrix - h_mean  # H x N (observation space anomalies)
        X_anom = (ensemble - np.mean(ensemble, axis=0, keepdims=True)).T  # M x N (parameter space anomalies)
        
        # Measurement error covariance matrix - one error per observation (not per type)
        # Create error array for each observation point
        error_values = []
        for k in observations.keys():
            n_obs_type = len(observations[k])
            error_std = measurement_error.get(k, 0.1)
            error_values.extend([error_std] * n_obs_type)
        
        R = np.diag(np.array(error_values) ** 2)  # H x H covariance matrix
        
        # Kalman gain: K = X_anom * Y_anom.T * (Y_anom * Y_anom.T + R)^{-1}
        try:
            denominator = Y_anom @ Y_anom.T + R
            K_gain = (X_anom @ Y_anom.T) @ np.linalg.inv(denominator)
            
            # Update each ensemble member
            updated_ensemble = ensemble.copy()
            for i in range(N):
                # Innovation: real observations - simulated for this member
                innovation = obs_vec - H_matrix[:, i]
                updated_ensemble[i, :] += K_gain @ innovation
            
            return updated_ensemble
            
        except np.linalg.LinAlgError as e:
            logger.warning(f"Singular matrix in Kalman update: {e}. Using ensemble mean.")
            return ensemble
    
    def _calculate_ensemble_quality(self, observations: Dict, 
                                   simulated_data: List[Dict]) -> float:
        """Calculate match quality for entire ensemble"""
        qualities = []
        
        for sim_dict in simulated_data:
            quality = self._calculate_match_quality(observations, sim_dict)
            qualities.append(quality)
        
        return float(np.mean(qualities))
    
    def _calculate_match_quality(self, observed: Dict, simulated: Dict) -> float:
        """Calculate match quality between observed and simulated data"""
        try:
            errors = []
            
            for key in observed:
                if key not in simulated:
                    continue
                
                obs = np.array(observed[key])
                sim = np.array(simulated[key])
                
                # Interpolate if lengths differ
                if len(obs) != len(sim):
                    sim = np.interp(np.linspace(0, 1, len(obs)),
                                   np.linspace(0, 1, len(sim)), sim)
                
                # Normalized RMSE
                rmse = np.sqrt(np.mean((obs - sim) ** 2))
                nrmse = rmse / (np.max(obs) - np.min(obs)) if np.ptp(obs) > 0 else 1.0
                error = max(0, 100 * (1 - nrmse))
                errors.append(error)
            
            return float(np.mean(errors)) if errors else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating quality: {e}")
            return 0.0
    
    def _extract_observations(self, data: Dict) -> Dict:
        """Extract observations from data"""
        try:
            # Check if data already has oil/water/gas/pressure keys (format from run_enkf_with_forecasts)
            if all(key in data for key in ['oil', 'water', 'gas', 'pressure']):
                return {
                    'oil': np.array(data['oil'], dtype=float),
                    'water': np.array(data['water'], dtype=float),
                    'gas': np.array(data['gas'], dtype=float),
                    'pressure': np.array(data['pressure'], dtype=float)
                }
            
            # Check if data has production_data sub-dict
            if 'production_data' in data:
                prod_data = data['production_data']
                return {
                    'oil': np.array(prod_data.get('oil', prod_data.get('Oil_bbl', [])), dtype=float),
                    'water': np.array(prod_data.get('water', prod_data.get('Water_bbl', [])), dtype=float),
                    'gas': np.array(prod_data.get('gas', prod_data.get('Gas_scf', [])), dtype=float),
                    'pressure': np.array(prod_data.get('pressure', prod_data.get('Pressure_psi', [])), dtype=float)
                }
            
            # Last resort fallback if data is completely empty
            logger.warning("No observation data found in extract_observations. Using empty dict.")
            return {}
        except Exception as e:
            logger.warning(f"Could not extract observations: {e}")
            return {}
    
    def _array_to_params(self, param_array: np.ndarray) -> Dict:
        """Convert parameter array to dictionary"""
        param_names = ['initial_pressure', 'porosity', 'permeability', 'water_saturation']
        return {name: float(value) for name, value in zip(param_names, param_array)}
    
    def initialize_ensemble(
        self,
        mean_params: Dict,
        std_params: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Initialize ensemble from mean and std of parameters
        
        Args:
            mean_params: Mean parameter values
            std_params: Standard deviation of parameters (default 10%)
            
        Returns:
            Ensemble matrix (ensemble_size x num_params)
        """
        if std_params is None:
            std_params = {k: v * 0.1 for k, v in mean_params.items()}
        
        param_names = list(mean_params.keys())
        means = np.array([mean_params[k] for k in param_names])
        stds = np.array([std_params.get(k, means[i] * 0.1) 
                        for i, k in enumerate(param_names)])
        
        # Generate ensemble from normal distribution
        ensemble = np.zeros((self.ensemble_size, len(param_names)))
        for i in range(len(param_names)):
            ensemble[:, i] = np.random.normal(means[i], stds[i], self.ensemble_size)
        
        return ensemble
    
    def get_history(self) -> List[Dict]:
        """Get EnKF iteration history"""
        return self.ensemble_history
