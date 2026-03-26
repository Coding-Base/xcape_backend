"""
Simulation Engine
Main coordinator for simulation runs
"""

import logging
from typing import Dict, List, Optional
from datetime import datetime
import json
import numpy as np

from .opm_wrapper import OPMFlowWrapper

logger = logging.getLogger(__name__)


class SimulationEngine:
    """Main simulation engine for executing reserves matching"""
    
    def __init__(self):
        """Initialize simulation engine"""
        self.opm = OPMFlowWrapper()
        self.simulations = {}
    
    def create_simulation(
        self,
        simulation_id: str,
        parameters: Dict,
        dataset_id: Optional[str] = None
    ) -> Dict:
        """
        Create and initialize a simulation
        
        Args:
            simulation_id: Unique simulation identifier
            parameters: Reservoir parameters (pressure, porosity, permeability, saturation)
            dataset_id: Associated dataset for comparison
            
        Returns:
            Simulation configuration
        """
        config = {
            'id': simulation_id,
            'created_at': datetime.now().isoformat(),
            'parameters': parameters,
            'dataset_id': dataset_id,
            'status': 'initialized',
            'progress': 0,
            'results': {}
        }
        self.simulations[simulation_id] = config
        logger.info(f"Simulation {simulation_id} initialized")
        return config
    
    def run_forward_model(
        self,
        simulation_id: str,
        parameters: Dict,
        progress_callback = None
    ) -> Dict:
        """
        Execute forward model simulation
        
        Args:
            simulation_id: Simulation ID
            parameters: Reservoir parameters
            progress_callback: Function to call for progress updates
            
        Returns:
            Simulation results
        """
        logger.info(f"Starting forward model for simulation {simulation_id}")
        
        # Create ECL deck (simplified version)
        deck = self._create_deck(parameters)
        
        # Update progress
        if progress_callback:
            progress_callback(simulation_id, 10, "Deck created")
        
        # Run OPM Flow
        success, results = self.opm.run_simulation(deck, parameters)
        
        if not success:
            logger.error(f"Forward model failed: {results}")
            if progress_callback:
                progress_callback(simulation_id, 0, f"Failed: {results.get('error')}")
            return {'status': 'failed', 'error': results.get('error')}
        
        # Update progress
        if progress_callback:
            progress_callback(simulation_id, 50, "Simulation completed")
        
        # Store results
        self.simulations[simulation_id]['results'] = results
        self.simulations[simulation_id]['status'] = 'completed'
        
        if progress_callback:
            progress_callback(simulation_id, 100, "Forward model completed")
        
        return {
            'status': 'completed',
            'results': results,
            'message': 'Forward model executed successfully'
        }
    
    def _create_deck(self, parameters: Dict) -> str:
        """
        Create ECL deck from parameters
        This is a simplified version - full implementation would be more complex
        """
        deck = f"""
-- XCAPE Simulation Deck
TITLE
  XCAPE Reservoir Simulation
/

RUNSPEC
  DIMENS
    50 50 10 /
  OIL
  WATER
  GAS
  DISGAS
  VAPOIL
  AQUIFER
  FULLIMP
/

TABDIMS
  1  1  15 20 1 10 100 20 1  1  1  1 /
/

START
  01 JAN 2000 /

WELSPECS
  'PROD1' 'G1' 25 25 0 OIL /
  'INJ1'  'G2' 10 10 0 WAT /
/

COMPDAT
  'PROD1' 25 25 1 10 OPEN 0 -1.0 0.0 0.5 /
  'INJ1'  10 10 1 10 OPEN 0 -1.0 0.0 0.5 /
/

WCONPROD
  'PROD1' OPEN ORAT 500 4*  10 /
/

WCONINJE
  'INJ1' WAT OPEN RATE 750 4* 100 /
/

PROPS
  SWOF
    0.2 0.000 1.0 4.0
    1.0 1.0 0.0 0.001
  /
  
  SGOF
    0.0 0.0 1.0 0.0
    1.0 1.0 0.0 0.0
  /
/

SOLUTION
  EQUIL
    2500 150 2700 0.0 0 0 1 1 0 0 0 0 0 0 0 /
/

SUMMARY
  WOPR
  WWPR
  WGPR
  WBHP
  FOPT
  FOPR
  FWPT
  FWPR
/

SCHEDULE
  TSTEP
    1 2 3 5 8 13 21 34 55 89 144 233 377 610 987 1597 2584
  /
/
        """
        return deck
    
    def get_simulation(self, simulation_id: str) -> Optional[Dict]:
        """Get simulation by ID"""
        return self.simulations.get(simulation_id)
    
    def get_all_simulations(self) -> List[Dict]:
        """Get all simulations"""
        return list(self.simulations.values())
    
    def cancel_simulation(self, simulation_id: str) -> bool:
        """Cancel a running simulation"""
        if simulation_id in self.simulations:
            self.simulations[simulation_id]['status'] = 'cancelled'
            logger.info(f"Simulation {simulation_id} cancelled")
            return True
        return False
