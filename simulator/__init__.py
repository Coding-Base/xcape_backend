"""
Simulator module for XCAPE
Handles OPM Flow integration, baseline matching, and EnKF algorithms
"""

from .engine import SimulationEngine
from .baseline_matcher import BaselineMatcher
from .enkf_filter import EnKFFilter
from .forecast_generator import ForecastGenerator

__all__ = [
    'SimulationEngine',
    'BaselineMatcher',
    'EnKFFilter',
    'ForecastGenerator',
]
