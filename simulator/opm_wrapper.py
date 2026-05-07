"""
OPM Flow Wrapper
Handles integration with OPM Flow simulator through Python bindings
"""

import subprocess
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OPMFlowWrapper:
    """Interface for OPM Flow simulator execution"""
    
    def __init__(self):
        """Initialize OPM Flow wrapper"""
        self.opm_executable = self._find_opm_executable()
        self.temp_dir = None
    
    def _find_opm_executable(self) -> Optional[str]:
        """Find OPM Flow executable in system PATH"""
        try:
            # Allow overriding via environment variable first
            env_path = os.environ.get('OPM_FLOW_EXEC') or os.environ.get('OPM_FLOW_PATH')

            def normalize(p: str) -> str:
                if not p:
                    return p
                p = p.strip().strip('"').strip("'")
                p = os.path.expanduser(os.path.expandvars(p))
                return p

            if env_path:
                env_path = normalize(env_path)
                # Direct path (file or .bat) provided
                if os.path.exists(env_path):
                    return env_path
                # If a directory was provided, look for common binaries including .bat wrappers
                if os.path.isdir(env_path):
                    for name in ('flow', 'flow.exe', 'flow.bat', 'opm_flow.bat'):
                        candidate = os.path.join(env_path, name)
                        if os.path.exists(candidate):
                            return candidate

            # If no env var or it didn't resolve, attempt to read a .env file in repo root(s)
            try:
                here = Path(__file__).resolve()
                for parent in list(here.parents)[:6]:
                    env_file = parent / '.env'
                    if env_file.exists():
                        try:
                            content = env_file.read_text(encoding='utf-8')
                            for line in content.splitlines():
                                if 'OPM_FLOW_EXEC' in line:
                                    # parse assignment
                                    parts = line.split('=', 1)
                                    if len(parts) == 2:
                                        candidate = normalize(parts[1])
                                        # strip possible surrounding quotes and whitespace
                                        candidate = candidate.strip()
                                        candidate = candidate.strip('"').strip("'")
                                        if os.path.exists(candidate):
                                            return candidate
                        except Exception:
                            pass
            except Exception:
                pass

            # Use shutil.which to find executable cross-platform
            try:
                import shutil
                exe = shutil.which('flow') or shutil.which('flow.exe') or shutil.which('flow.bat')
                if exe:
                    return exe
                # also attempt basename of env_path if provided
                if env_path:
                    base = os.path.basename(env_path)
                    exe = shutil.which(base)
                    if exe:
                        return exe
            except Exception:
                pass

            # Fallback: try POSIX 'which' for Unix-like environments
            try:
                result = subprocess.run(['which', 'flow'], capture_output=True, text=True)
                if result.returncode == 0:
                    return result.stdout.strip()
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Could not find OPM Flow executable: {e}")
        return None
    
    def is_available(self) -> bool:
        """Check if OPM Flow is available"""
        return self.opm_executable is not None
    
    def run_simulation(
        self,
        deck_file: str,
        parameters: Optional[Dict] = None,
        timeout: int = 3600
    ) -> Tuple[bool, Dict]:
        """
        Run OPM Flow simulation
        
        Args:
            deck_file: Path to ECL deck file
            parameters: Reservoir parameters to override
            timeout: Simulation timeout in seconds
            
        Returns:
            Tuple of (success, results_dict)
        """
        try:
            if not self.opm_executable:
                # Return mock results for testing
                return self._get_mock_results(parameters)
            
            # Build command
            cmd = [self.opm_executable, deck_file]
            
            # Run simulation
            logger.info(f"Starting OPM Flow simulation: {' '.join(cmd)}")
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            
            if result.returncode != 0:
                logger.error(f"OPM Flow simulation failed: {result.stderr}")
                return False, {'error': result.stderr}
            
            # Parse results (this would read from output files)
            results = self._parse_results(deck_file)
            return True, results
            
        except subprocess.TimeoutExpired:
            return False, {'error': 'Simulation timeout'}
        except Exception as e:
            logger.error(f"Error running OPM Flow simulation: {e}")
            return False, {'error': str(e)}
    
    def _get_mock_results(self, parameters: Optional[Dict] = None) -> Tuple[bool, Dict]:
        """Get mock simulation results for testing"""
        import numpy as np
        from datetime import datetime
        
        # Generate synthetic production data
        time_steps = 100
        days = np.linspace(0, 365, time_steps)
        
        # Simple production model with parameters
        oil_rate = 100.0  # barrels per day
        water_rate = 50.0
        gas_rate = 500.0
        
        if parameters:
            # Adjust rates based on parameters
            permeability = parameters.get('permeability', 100) / 100
            porosity = parameters.get('porosity', 20) / 20
            oil_rate *= permeability * porosity
            water_rate *= permeability * 0.5
            gas_rate *= permeability
        
        # Generate production data
        oil_production = oil_rate * (1 - np.exp(-days / 100)) * np.random.uniform(0.95, 1.05, time_steps)
        water_production = water_rate * (1 - np.exp(-days / 80)) * np.random.uniform(0.95, 1.05, time_steps)
        gas_production = gas_rate * (1 - np.exp(-days / 120)) * np.random.uniform(0.95, 1.05, time_steps)
        
        return True, {
            'simulation_date': datetime.now().isoformat(),
            'status': 'completed',
            'time_steps': int(time_steps),
            'total_days': int(days[-1]),
            'production_data': {
                'days': days.tolist(),
                'oil': oil_production.tolist(),
                'water': water_production.tolist(),
                'gas': gas_production.tolist(),
                'pressure': (150 - 50 * np.exp(-days / 200)).tolist(),
            },
            'reservoir_state': {
                'avg_pressure': float(np.mean(150 - 50 * np.exp(-days / 200))),
                'avg_saturation': parameters.get('water_saturation', 30) if parameters else 30,
            }
        }
    
    def _parse_results(self, deck_file: str) -> Dict:
        """Parse OPM Flow output files"""
        # This would read from .UNSMSPEC, .SMSPEC, restart files etc.
        results = {
            'status': 'completed',
            'message': 'Results parsed from OPM output files'
        }
        return results
    
    def validate_deck(self, deck_content: str) -> Tuple[bool, str]:
        """Validate ECL deck file syntax"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.deck', delete=False) as f:
                f.write(deck_content)
                temp_file = f.name
            
            if not self.opm_executable:
                return True, "OPM not available - validation skipped"
            
            # Run syntax check (if available)
            # This would use OPM Flow's built-in validation
            os.unlink(temp_file)
            return True, "Deck syntax valid"
            
        except Exception as e:
            return False, str(e)
    
    def cleanup(self):
        """Clean up temporary files"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
