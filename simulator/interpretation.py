"""
Simulation Interpretation Service
Provides expert-level analysis of simulation results for reservoir engineers
"""

import logging
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class ReservoirInterpretationEngine:
    """
    Expert interpretation engine for simulation results
    Analyzes well performance, pressure dynamics, and production trends
    for meaningful recommendations to reservoir engineers
    """
    
    # Engineering thresholds based on industry standards
    PRESSURE_DECLINE_THRESHOLD = 5.0  # % per month
    PRODUCTION_DECLINE_THRESHOLD = 3.0  # % per day
    PRESSURE_THRESHOLD_HIGH = 2500  # bar
    PRESSURE_THRESHOLD_NORMAL = 1500  # bar
    PRESSURE_THRESHOLD_LOW = 500  # bar
    BS_W_THRESHOLD = 30  # % water saturation concern threshold
    GOR_THRESHOLD = 500  # scf/bbl - gas oil ratio
    
    def __init__(self):
        """Initialize interpretation engine"""
        self.logger = logging.getLogger(__name__)
    
    def interpret_simulation(self, simulation_data: Dict) -> Dict:
        """
        Generate comprehensive interpretation of simulation results
        
        Args:
            simulation_data: Dict containing:
                - results_data: simulation output
                - initial_pressure: bar
                - porosity: %
                - permeability: mD
                - water_saturation: %
                - match_quality: float (0-100)
                - matching_type: baseline/enkf
        
        Returns:
            Dict with comprehensive interpretation including:
            - executive_summary
            - well_performance_analysis
            - production_trends
            - pressure_dynamics
            - water_saturation_analysis
            - forecast_interpretation
            - risk_assessment
            - recommendations
            - metrics (detailed list of analyzed metrics)
        """
        try:
            results = simulation_data.get('results_data', {})
            initial_pressure = simulation_data.get('initial_pressure', 1500)
            porosity = simulation_data.get('porosity', 20)
            permeability = simulation_data.get('permeability', 100)
            water_sat = simulation_data.get('water_saturation', 25)
            match_quality = simulation_data.get('match_quality', 0)
            # If observed time-series data is provided in simulation_data, prefer those
            observed = simulation_data.get('observed_data') or simulation_data.get('dataset_production') or {}
            # observed may contain arrays for 'oil','water','gas','pressure'
            obs_oil = np.array(observed.get('oil', []), dtype=float) if observed else np.array([])
            obs_water = np.array(observed.get('water', []), dtype=float) if observed else np.array([])
            obs_gas = np.array(observed.get('gas', []), dtype=float) if observed else np.array([])
            obs_pressure = np.array(observed.get('pressure', []), dtype=float) if observed else np.array([])
            
            # Extract key metrics (prefer time-series summary if available)
            if obs_oil.size > 0:
                oil_rate = float(np.mean(obs_oil[-5:]))  # average of last 5 points
            else:
                oil_rate = float(results.get('oil_predicted', 0))

            if obs_water.size > 0:
                water_rate = float(np.mean(obs_water[-5:]))
            else:
                water_rate = float(results.get('water_predicted', 0))

            if obs_gas.size > 0:
                gas_rate = float(np.mean(obs_gas[-5:]))
            else:
                gas_rate = float(results.get('gas_predicted', 0))

            if obs_pressure.size > 0:
                current_pressure = float(np.mean(obs_pressure[-5:]))
            else:
                current_pressure = float(results.get('pressure_predicted', initial_pressure))
            
            # Calculate derived metrics
            pressure_decline = self._calculate_pressure_decline(initial_pressure, current_pressure)
            gas_oil_ratio = self._calculate_gor(gas_rate, oil_rate) if oil_rate > 0 else 0
            water_cut = self._calculate_water_cut(oil_rate, water_rate)
            productivity_index = self._calculate_productivity_index(oil_rate, pressure_decline, initial_pressure)

            # If observed time-series available, compute observed trends and discrepancies
            observed_discrepancies = {}
            try:
                if obs_oil.size > 2 and obs_pressure.size > 2:
                    # compute observed decline slope (linear fit) for oil and pressure
                    t_oil = np.arange(obs_oil.size)
                    slope_oil, intercept = np.polyfit(t_oil, obs_oil, 1)
                    pct_decline_oil = -100.0 * (slope_oil / np.mean(obs_oil)) if np.mean(obs_oil) != 0 else 0.0
                    observed_discrepancies['observed_oil_decline_pct_per_step'] = pct_decline_oil

                if obs_pressure.size > 2:
                    t_p = np.arange(obs_pressure.size)
                    slope_p, intercept = np.polyfit(t_p, obs_pressure, 1)
                    pct_decline_pressure = -100.0 * (slope_p / np.mean(obs_pressure)) if np.mean(obs_pressure) != 0 else 0.0
                    observed_discrepancies['observed_pressure_decline_pct_per_step'] = pct_decline_pressure

                # Compute simple RMSE between observed last-window and predicted scalar if both exist
                if obs_oil.size > 0 and results.get('oil_predicted') is not None:
                    pred_scalar = float(results.get('oil_predicted'))
                    rmse_oil = np.sqrt(np.mean((obs_oil - pred_scalar) ** 2))
                    observed_discrepancies['rmse_oil_vs_pred'] = float(rmse_oil)
                if obs_pressure.size > 0 and results.get('pressure_predicted') is not None:
                    pred_p = float(results.get('pressure_predicted'))
                    rmse_p = np.sqrt(np.mean((obs_pressure - pred_p) ** 2))
                    observed_discrepancies['rmse_pressure_vs_pred'] = float(rmse_p)
            except Exception:
                # Non-fatal: keep observed_discrepancies partial
                pass
            
            # Analyze well performance
            well_performance = self._analyze_well_performance(
                oil_rate, water_rate, gas_rate, water_cut, gas_oil_ratio, porosity, permeability
            )
            
            # Analyze pressure dynamics
            pressure_analysis = self._analyze_pressure_dynamics(
                initial_pressure, current_pressure, pressure_decline, permeability, porosity
            )
            
            # Analyze water saturation and production
            water_analysis = self._analyze_water_saturation(
                water_cut, water_sat, initial_pressure, current_pressure
            )
            
            # Analyze production trends
            production_trends = self._analyze_production_trends(
                oil_rate, water_rate, gas_rate, match_quality
            )
            
            # Generate forecast interpretation
            forecast = self._interpret_forecast(
                oil_rate, pressure_decline, water_cut, productivity_index
            )
            
            # Risk assessment
            risks = self._assess_risks(
                pressure_decline, water_cut, gas_oil_ratio, match_quality
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                well_performance, pressure_analysis, water_analysis, 
                forecast, risks, pressure_decline, water_cut, productivity_index
            )
            
            # Build metrics list for detailed view
            detailed_metrics = self._build_detailed_metrics(
                oil_rate, water_rate, gas_rate, initial_pressure, current_pressure,
                pressure_decline, water_cut, gas_oil_ratio, productivity_index,
                match_quality, porosity, permeability
            )
            
            # Executive summary
            executive_summary = self._generate_executive_summary(
                well_performance, pressure_analysis, water_analysis, 
                forecast, risks, match_quality
            )
            
            return {
                'executive_summary': executive_summary,
                'well_performance': well_performance,
                'pressure_dynamics': pressure_analysis,
                'water_saturation': water_analysis,
                'production_trends': production_trends,
                'forecast_interpretation': forecast,
                'risk_assessment': risks,
                'recommendations': recommendations,
                'metrics': detailed_metrics,
                'observed_discrepancies': observed_discrepancies,
                'interpretation_timestamp': datetime.now().isoformat(),
            }
        
        except Exception as e:
            logger.error(f"Interpretation error: {str(e)}")
            return {
                'error': str(e),
                'executive_summary': 'Unable to interpret simulation results'
            }
    
    def _calculate_pressure_decline(self, initial: float, current: float) -> float:
        """Calculate absolute and relative pressure changes"""
        if initial == 0:
            return 0
        return ((initial - current) / initial) * 100
    
    def _calculate_gor(self, gas_rate: float, oil_rate: float) -> float:
        """Calculate Gas-Oil Ratio"""
        if oil_rate <= 0:
            return 0
        return gas_rate / oil_rate
    
    def _calculate_water_cut(self, oil_rate: float, water_rate: float) -> float:
        """Calculate water cut percentage"""
        total = oil_rate + water_rate
        if total <= 0:
            return 0
        return (water_rate / total) * 100
    
    def _calculate_productivity_index(self, oil_rate: float, pressure_decline: float, initial_pressure: float) -> float:
        """
        Calculate productivity index (PI)
        PI = oil production rate / pressure drawdown
        Units: bbl/day/psi
        """
        if pressure_decline <= 0 or initial_pressure <= 0:
            return 0
        pressure_drawdown = (pressure_decline / 100) * initial_pressure
        if pressure_drawdown <= 0:
            return 0
        return oil_rate / pressure_drawdown * 14.5038  # Convert bar to psi
    
    def _analyze_well_performance(self, oil_rate: float, water_rate: float, gas_rate: float,
                                  water_cut: float, gor: float, porosity: float, 
                                  permeability: float) -> Dict:
        """Comprehensive well performance analysis"""
        performance = {
            'status': 'excellent' if oil_rate > 50000 else 'good' if oil_rate > 20000 else 'moderate',
            'oil_production_rate': f"{oil_rate:,.0f} bbl/day",
            'water_production_rate': f"{water_rate:,.0f} bbl/day",
            'gas_production_rate': f"{gas_rate:,.0f} scf/day",
            'assessment': []
        }
        
        # Oil production assessment
        if oil_rate > 75000:
            performance['assessment'].append({
                'metric': 'Oil Production',
                'status': 'EXCELLENT',
                'value': f"{oil_rate:,.0f} bbl/day",
                'insight': 'Exceptionally high oil production rate indicates excellent well connectivity and reservoir quality'
            })
        elif oil_rate > 40000:
            performance['assessment'].append({
                'metric': 'Oil Production',
                'status': 'GOOD',
                'value': f"{oil_rate:,.0f} bbl/day",
                'insight': 'Good oil production rate consistent with effective reservoir drainage'
            })
        elif oil_rate > 15000:
            performance['assessment'].append({
                'metric': 'Oil Production',
                'status': 'MODERATE',
                'value': f"{oil_rate:,.0f} bbl/day",
                'insight': 'Moderate production - may indicate wellbore restrictions or limited pressure drawdown'
            })
        else:
            performance['assessment'].append({
                'metric': 'Oil Production',
                'status': 'LOW',
                'value': f"{oil_rate:,.0f} bbl/day",
                'insight': 'Low oil production suggests significant flow restrictions or depleted reservoir section'
            })
        
        # Water cut assessment
        if water_cut < 5:
            performance['assessment'].append({
                'metric': 'Water Cut',
                'status': 'EXCELLENT',
                'value': f"{water_cut:.1f}%",
                'insight': 'Very low water production indicates early production life or good aquifer containment'
            })
        elif water_cut < 20:
            performance['assessment'].append({
                'metric': 'Water Cut',
                'status': 'GOOD',
                'value': f"{water_cut:.1f}%",
                'insight': 'Acceptable water cut for mature well. Monitor for future water breakthrough'
            })
        elif water_cut < 50:
            performance['assessment'].append({
                'metric': 'Water Cut',
                'status': 'MODERATE',
                'value': f"{water_cut:.1f}%",
                'insight': f'Increasing water production ({water_cut:.1f}%). Consider water management or production optimization'
            })
        else:
            performance['assessment'].append({
                'metric': 'Water Cut',
                'status': 'HIGH',
                'value': f"{water_cut:.1f}%",
                'insight': 'High water cut indicates advanced aquifer encroachment. Intervention may be cost-prohibitive'
            })
        
        # GOR assessment
        gor_status = 'normal'
        gor_insight = f"Gas-oil ratio of {gor:.1f} scf/bbl is within expected range"
        if gor > self.GOR_THRESHOLD:
            gor_status = 'HIGH'
            gor_insight = f"Gas-oil ratio ({gor:.1f}) exceeds typical threshold. Gas-cap advance or gas liberation expected"
        elif gor > self.GOR_THRESHOLD * 0.7:
            gor_status = 'ELEVATED'
            gor_insight = f"GOR approaching threshold ({gor:.1f}). Expect increasing gas production"
        
        performance['assessment'].append({
            'metric': 'Gas-Oil Ratio',
            'status': gor_status,
            'value': f"{gor:.1f} scf/bbl",
            'insight': gor_insight
        })
        
        # Reservoir quality assessment based on permeability
        if permeability > 500:
            performance['assessment'].append({
                'metric': 'Reservoir Quality',
                'status': 'EXCELLENT',
                'value': f"{permeability:.0f} mD",
                'insight': 'High permeability reservoir - excellent fluid flow potential'
            })
        elif permeability > 100:
            performance['assessment'].append({
                'metric': 'Reservoir Quality',
                'status': 'GOOD',
                'value': f"{permeability:.0f} mD",
                'insight': 'Good permeability typical of productive reservoirs'
            })
        else:
            performance['assessment'].append({
                'metric': 'Reservoir Quality',
                'status': 'FAIR',
                'value': f"{permeability:.0f} mD",
                'insight': 'Lower permeability may limit production capacity'
            })
        
        return performance
    
    def _analyze_pressure_dynamics(self, initial: float, current: float, decline_pct: float,
                                   permeability: float, porosity: float) -> Dict:
        """Analyze pressure behavior and depletion regime"""
        analysis = {
            'initial_pressure': f"{initial:.1f} bar",
            'current_pressure': f"{current:.1f} bar",
            'absolute_decline': f"{initial - current:.1f} bar",
            'relative_decline': f"{decline_pct:.2f}%",
            'assessment': []
        }
        
        # Pressure regime classification
        if current > self.PRESSURE_THRESHOLD_HIGH:
            regime = 'HIGH PRESSURE'
            regime_insight = 'Well in high-pressure regime. Expect strong natural drive and sustained production'
        elif current > self.PRESSURE_THRESHOLD_NORMAL:
            regime = 'NORMAL PRESSURE'
            regime_insight = 'Well in normal pressure depletion regime. Sustainable production anticipated'
        elif current > self.PRESSURE_THRESHOLD_LOW:
            regime = 'LOW PRESSURE'
            regime_insight = 'Significant pressure depletion. Secondary recovery methods may be required'
        else:
            regime = 'DEPLETED'
            regime_insight = 'Well approaching economic limit. Intervention strategies needed'
        
        analysis['assessment'].append({
            'metric': 'Pressure Regime',
            'status': regime,
            'value': f"{current:.0f} bar",
            'insight': regime_insight
        })
        
        # Depletion rate assessment
        if decline_pct < 1.0:
            analysis['assessment'].append({
                'metric': 'Depletion Rate',
                'status': 'SLOW',
                'value': f"{decline_pct:.2f}% decline",
                'insight': 'Minimal pressure decline indicates strong aquifer support or weak depletion'
            })
        elif decline_pct < 10.0:
            analysis['assessment'].append({
                'metric': 'Depletion Rate',
                'status': 'SUSTAINABLE',
                'value': f"{decline_pct:.2f}% decline",
                'insight': 'Moderate pressure depletion consistent with primary depletion mechanisms'
            })
        elif decline_pct < 30.0:
            analysis['assessment'].append({
                'metric': 'Depletion Rate',
                'status': 'SIGNIFICANT',
                'value': f"{decline_pct:.2f}% decline",
                'insight': 'Substantial pressure loss requiring production management and monitoring'
            })
        else:
            analysis['assessment'].append({
                'metric': 'Depletion Rate',
                'status': 'SEVERE',
                'value': f"{decline_pct:.2f}% decline",
                'insight': 'Severe pressure depletion - emergency intervention may be necessary'
            })
        
        # Permeability impact
        if permeability > 200:
            analysis['assessment'].append({
                'metric': 'Flow Capacity',
                'status': 'HIGH',
                'value': f"{permeability:.0f} mD",
                'insight': 'High permeability maintains good pressure communication and fluid mobility'
            })
        elif permeability > 50:
            analysis['assessment'].append({
                'metric': 'Flow Capacity',
                'status': 'MODERATE',
                'value': f"{permeability:.0f} mD",
                'insight': 'Moderate permeability limits pressure support and production capacity'
            })
        else:
            analysis['assessment'].append({
                'metric': 'Flow Capacity',
                'status': 'LIMITED',
                'value': f"{permeability:.0f} mD",
                'insight': 'Low permeability restricts lateral pressure support and recovery'
            })
        
        return analysis
    
    def _analyze_water_saturation(self, water_cut: float, initial_water_sat: float,
                                  initial_pressure: float, current_pressure: float) -> Dict:
        """Analyze water contact and saturation changes"""
        analysis = {
            'initial_water_saturation': f"{initial_water_sat:.1f}%",
            'current_water_cut': f"{water_cut:.1f}%",
            'assessment': []
        }
        
        # Water saturation interpretation
        if initial_water_sat < 15:
            saturation_type = 'OIL-WET'
            saturation_insight = 'Low initial water saturation typical of oil-wet reservoirs with good oil recovery potential'
        elif initial_water_sat < 35:
            saturation_type = 'MODERATELY OIL-WET'
            saturation_insight = 'Moderate water saturation with oil-wetting tendency favorable for oil production'
        else:
            saturation_type = 'WATER-DOMINANT'
            saturation_insight = 'High water saturation may impact oil recovery and production quality'
        
        analysis['assessment'].append({
            'metric': 'Water Saturation Type',
            'status': saturation_type,
            'value': f"{initial_water_sat:.1f}%",
            'insight': saturation_insight
        })
        
        # Water saturation change rate
        saturation_change = initial_water_sat - (initial_water_sat * (1 - water_cut/100))
        if saturation_change < 5:
            change_status = 'STABLE'
            change_insight = 'Stable water saturation indicates good separation or strong aquifer support'
        elif saturation_change < 15:
            change_status = 'MODERATE ADVANCE'
            change_insight = f'Water saturation increasing by ~{saturation_change:.1f}%. Monitor water encroachment'
        else:
            change_status = 'RAPID ADVANCE'
            change_insight = f'Rapid water saturation increase ({saturation_change:.1f}%). Strong aquifer encroachment'
        
        analysis['assessment'].append({
            'metric': 'Water Saturation Change',
            'status': change_status,
            'value': f"{saturation_change:.1f}% increase",
            'insight': change_insight
        })
        
        # Aquifer strength assessment
        pressure_maintained = (current_pressure / initial_pressure) > 0.9
        if pressure_maintained:
            aquifer_status = 'STRONG'
            aquifer_insight = 'Strong aquifer support indicated by maintained pressure - excellent bottom-hole conditions'
        else:
            aquifer_status = 'MODERATE TO WEAK'
            aquifer_insight = 'Weak aquifer support - pressure depletion will accelerate if production continues'
        
        analysis['assessment'].append({
            'metric': 'Aquifer Support',
            'status': aquifer_status,
            'value': f"{(current_pressure/initial_pressure)*100:.1f}% remaining",
            'insight': aquifer_insight
        })
        
        return analysis
    
    def _analyze_production_trends(self, oil_rate: float, water_rate: float, gas_rate: float,
                                   match_quality: float) -> Dict:
        """Analyze production trends and match quality"""
        trends = {
            'combined_production': f"{oil_rate + water_rate:,.0f} bbl/day equivalent",
            'liquid_production': f"{oil_rate + water_rate:,.0f} bbl/day",
            'gas_production': f"{gas_rate:,.0f} scf/day",
            'assessment': []
        }
        
        # Match quality assessment
        if match_quality > 85:
            match_status = 'EXCELLENT'
            match_insight = f'Match quality of {match_quality:.1f}% indicates model accurately reproduces observed behavior'
        elif match_quality > 70:
            match_status = 'GOOD'
            match_insight = f'Match quality of {match_quality:.1f}% acceptable for forecasting and optimization'
        elif match_quality > 55:
            match_status = 'FAIR'
            match_insight = f'Match quality of {match_quality:.1f}% suggests model refinement recommended'
        else:
            match_status = 'POOR'
            match_insight = f'Low match quality ({match_quality:.1f}%) - consider parameter adjustment or data quality'
        
        trends['assessment'].append({
            'metric': 'History Match Quality',
            'status': match_status,
            'value': f"{match_quality:.1f}%",
            'insight': match_insight
        })
        
        # Production trend assessment
        total_liquid = oil_rate + water_rate
        if total_liquid > 100000:
            production_trend = 'HIGH VOLUME'
            trend_insight = 'High-rate production well. Carefully manage to avoid mechanical issues or unrealistic decline'
        elif total_liquid > 50000:
            production_trend = 'MODERATE-HIGH'
            trend_insight = 'Good production rates suitable for long-term economic operation'
        elif total_liquid > 15000:
            production_trend = 'MODERATE'
            trend_insight = 'Stable, predictable production rates'
        else:
            production_trend = 'LOW'
            trend_insight = 'Low production rates approaching uneconomic thresholds'
        
        trends['assessment'].append({
            'metric': 'Production Trend',
            'status': production_trend,
            'value': f"{total_liquid:,.0f} bbl/day",
            'insight': trend_insight
        })
        
        return trends
    
    def _interpret_forecast(self, current_oil_rate: float, pressure_decline: float,
                           water_cut: float, productivity_index: float) -> Dict:
        """Interpret production forecast and future well behavior"""
        forecast = {
            'assessment': [],
            'production_outlook': '',
            'timeframe': '6-12 months'
        }
        
        # Production decline forecast
        if pressure_decline < 5:
            decline_rate = 1.5  # % per year
            decline_outlook = 'SLOW'
            decline_insight = 'Low pressure decline suggests gradual production loss - well remains productive'
        elif pressure_decline < 15:
            decline_rate = 5.0  # % per year
            decline_outlook = 'MODERATE'
            decline_insight = 'Moderate decline expected - typical for primary depletion wells'
        else:
            decline_rate = 12.0  # % per year
            decline_outlook = 'RAPID'
            decline_insight = 'Rapid decline anticipated - urgent intervention may be required'
        
        forecast['assessment'].append({
            'metric': 'Production Decline Rate',
            'status': decline_outlook,
            'value': f"~{decline_rate:.1f}% per year",
            'insight': decline_insight,
            'forecast_oil': f"{current_oil_rate * (1 - decline_rate/100):,.0f} bbl/day (in 1 year)"
        })
        
        # Water breakthrough forecast
        if water_cut < 10:
            water_forecast = 'EARLY'
            water_insight = 'Water breakthrough likely within 6-18 months; prepare contingency plans'
        elif water_cut < 30:
            water_forecast = 'PROGRESSING'
            water_insight = 'Water production will continue increasing; monitor for economic limit'
        elif water_cut < 70:
            water_forecast = 'ADVANCED'
            water_insight = 'Advanced water cut; focus on cost optimization and well integrity'
        else:
            water_forecast = 'MATURE'
            water_insight = 'Well likely approaching economic abandonment threshold'
        
        forecast['assessment'].append({
            'metric': 'Water Encroachment Stage',
            'status': water_forecast,
            'value': f"{water_cut:.1f}% current water cut",
            'insight': water_insight
        })
        
        # Well lifetime estimate
        if current_oil_rate > 50000 and pressure_decline < 10:
            well_life = '10-15 years additional production'
            life_insight = 'Excellent well with extended economic life ahead'
        elif current_oil_rate > 20000 or pressure_decline < 20:
            well_life = '3-7 years additional production'
            life_insight = 'Well has moderate remaining productive life'
        else:
            well_life = '1-2 years remaining'
            life_insight = 'Limited time to optimize before economic abandonment'
        
        forecast['assessment'].append({
            'metric': 'Well Economic Life',
            'status': 'ESTIMATED',
            'value': well_life,
            'insight': life_insight
        })
        
        # Overall production outlook
        if decline_rate < 3 and water_cut < 20:
            outlook = 'STABLE - Production expected to remain relatively stable with gradual decline'
        elif decline_rate < 8 and water_cut < 50:
            outlook = 'DECLINING - Expected gradual production loss requiring active management'
        else:
            outlook = 'CRITICAL - Rapid decline anticipated; strategic intervention recommended'
        
        forecast['production_outlook'] = outlook
        
        return forecast
    
    def _assess_risks(self, pressure_decline: float, water_cut: float, gor: float,
                      match_quality: float) -> Dict:
        """Comprehensive risk assessment"""
        risks = {
            'overall_risk_level': 'LOW',
            'risk_factors': []
        }
        
        risk_score = 0
        
        # Pressure-related risks
        if pressure_decline > 30:
            risks['risk_factors'].append({
                'category': 'RESERVOIR PRESSURE',
                'level': 'HIGH',
                'factor': 'Severe pressure depletion',
                'impact': 'Critical impact on well productivity; secondary recovery may be necessary',
                'mitigation': 'Consider pressure maintenance via water/gas injection'
            })
            risk_score += 25
        elif pressure_decline > 15:
            risks['risk_factors'].append({
                'category': 'RESERVOIR PRESSURE',
                'level': 'MEDIUM',
                'factor': 'Significant pressure decline',
                'impact': 'Moderate reduction in well productivity',
                'mitigation': 'Monitor pressure trends; plan production strategy accordingly'
            })
            risk_score += 10
        
        # Water-related risks
        if water_cut > 50:
            risks['risk_factors'].append({
                'category': 'WATER PRODUCTION',
                'level': 'HIGH',
                'factor': 'Very high water cut',
                'impact': 'Economic challenges; excessive water separation costs',
                'mitigation': 'Evaluate water shutdown or selective perforation techniques'
            })
            risk_score += 25
        elif water_cut > 30:
            risks['risk_factors'].append({
                'category': 'WATER PRODUCTION',
                'level': 'MEDIUM',
                'factor': 'Increasing water production',
                'impact': 'Growing operational and cost concerns',
                'mitigation': 'Implement water management strategy; monitor trends closely'
            })
            risk_score += 10
        
        # Gas-related risks
        if gor > 1000:
            risks['risk_factors'].append({
                'category': 'GAS PRODUCTION',
                'level': 'HIGH',
                'factor': 'Very high gas-oil ratio',
                'impact': 'Gas handling and treatment challenges; potential separation issues',
                'mitigation': 'Upgrade processing facilities; consider gas export or compression'
            })
            risk_score += 15
        elif gor > 500:
            risks['risk_factors'].append({
                'category': 'GAS PRODUCTION',
                'level': 'MEDIUM',
                'factor': 'Elevated gas production',
                'impact': 'Increasing gas handling requirements',
                'mitigation': 'Evaluate processing capacity; plan facility upgrades if necessary'
            })
            risk_score += 8
        
        # Model uncertainty risks
        if match_quality < 60:
            risks['risk_factors'].append({
                'category': 'MODEL UNCERTAINTY',
                'level': 'HIGH',
                'factor': 'Poor history match quality',
                'impact': 'Low confidence in predictions; unreliable for major decisions',
                'mitigation': 'Refine model parameters; integrate additional data; perform sensitivity analysis'
            })
            risk_score += 20
        elif match_quality < 75:
            risks['risk_factors'].append({
                'category': 'MODEL UNCERTAINTY',
                'level': 'MEDIUM',
                'factor': 'Moderate match quality',
                'impact': 'Some uncertainty in predictions',
                'mitigation': 'Validate assumptions; use probabilistic approaches; perform sensitivity analysis'
            })
            risk_score += 8
        
        # Determine overall risk level
        if risk_score >= 50:
            risks['overall_risk_level'] = 'HIGH'
        elif risk_score >= 25:
            risks['overall_risk_level'] = 'MEDIUM'
        else:
            risks['overall_risk_level'] = 'LOW'
        
        risks['risk_score'] = risk_score
        
        return risks
    
    def _generate_recommendations(self, well_perf: Dict, pressure_analysis: Dict,
                                 water_analysis: Dict, forecast: Dict, risks: Dict,
                                 pressure_decline: float, water_cut: float,
                                 productivity_index: float) -> List[Dict]:
        """Generate actionable recommendations for reservoir engineers"""
        recommendations = []
        
        # Production optimization recommendations
        if pressure_decline > 15:
            recommendations.append({
                'priority': 'HIGH',
                'category': 'PRODUCTION OPTIMIZATION',
                'title': 'Implement Pressure Maintenance Strategy',
                'description': 'Significant pressure decline detected. Consider water or gas injection to maintain reservoir pressure and sustain production rates.',
                'expected_benefit': 'Extend well economic life by 5-10 years; maintain production rates',
                'implementation_effort': 'HIGH - requires capital investment and operational changes'
            })
        
        if water_cut > 20 and water_cut < 50:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'WATER MANAGEMENT',
                'title': 'Implement Water Shutdown Program',
                'description': 'Water production is increasing and approaching operational concern levels. Evaluate selective performation or water shutoff options.',
                'expected_benefit': 'Reduce water handling costs; extend oil production period',
                'implementation_effort': 'MEDIUM - well intervention required'
            })
        
        if productivity_index > 0:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'WELL ENHANCEMENT',
                'title': 'Optimize Well Geometry and Completion',
                'description': 'Evaluate wellbore skin damage and completion design. Stimulation may improve flow capacity.',
                'expected_benefit': f'Increase production by 10-30%; improve productivity index',
                'implementation_effort': 'MEDIUM - well intervention required'
            })
        
        # Operational recommendations
        if pressure_decline < 5:
            recommendations.append({
                'priority': 'LOW',
                'category': 'OPERATIONS',
                'title': 'Maintain Current Production Rate',
                'description': 'Low pressure decline indicates strong reservoir support or weak depletion. Current production strategy appears sustainable.',
                'expected_benefit': 'Maintain stable economics and well performance',
                'implementation_effort': 'LOW - continue monitoring'
            })
        
        # Monitoring and surveillance
        recommendations.append({
            'priority': 'HIGH',
            'category': 'MONITORING & SURVEILLANCE',
            'title': 'Establish Comprehensive Monitoring Program',
            'description': 'Implement regular monitoring of well performance, pressure trends, and fluid properties. Monthly production and quarterly pressure surveys recommended.',
            'expected_benefit': 'Early detection of problems; optimize intervention timing',
            'implementation_effort': 'LOW - standard operational practice'
        })
        
        # Model refinement
        if risks.get('risk_score', 0) > 25:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'MODELING & FORECASTING',
                'title': 'Refine Simulation Model',
                'description': 'Improve history match quality by integrating additional data, refining grid resolution, or adjusting petrophysical parameters.',
                'expected_benefit': 'Increase forecast confidence; better decision support',
                'implementation_effort': 'MEDIUM - requires detailed model study'
            })
        
        # Economic analysis
        if water_cut > 30 or pressure_decline > 20:
            recommendations.append({
                'priority': 'MEDIUM',
                'category': 'ECONOMICS',
                'title': 'Perform Comprehensive Economic Analysis',
                'description': 'Conduct detailed NPV analysis under various production scenarios and intervention strategies.',
                'expected_benefit': 'Optimize capital allocation; maximize value creation',
                'implementation_effort': 'MEDIUM - cross-functional study'
            })
        
        return recommendations
    
    def _build_detailed_metrics(self, oil_rate: float, water_rate: float, gas_rate: float,
                               initial_pressure: float, current_pressure: float,
                               pressure_decline: float, water_cut: float, gor: float,
                               productivity_index: float, match_quality: float,
                               porosity: float, permeability: float) -> List[Dict]:
        """Build detailed metrics list for visualization"""
        metrics = [
            {
                'group': 'PRODUCTION',
                'items': [
                    {
                        'name': 'Oil Production Rate',
                        'value': f"{oil_rate:,.0f}",
                        'unit': 'bbl/day',
                        'benchmark': '50,000',
                        'status': 'excellent' if oil_rate > 50000 else 'good' if oil_rate > 20000 else 'moderate'
                    },
                    {
                        'name': 'Water Production Rate',
                        'value': f"{water_rate:,.0f}",
                        'unit': 'bbl/day',
                        'benchmark': '<10,000',
                        'status': 'good' if water_rate < 10000 else 'moderate' if water_rate < 30000 else 'poor'
                    },
                    {
                        'name': 'Gas Production Rate',
                        'value': f"{gas_rate:,.0f}",
                        'unit': 'scf/day',
                        'benchmark': '700,000',
                        'status': 'good' if 500000 < gas_rate < 1000000 else 'moderate'
                    },
                    {
                        'name': 'Water Cut',
                        'value': f"{water_cut:.2f}",
                        'unit': '%',
                        'benchmark': '<20%',
                        'status': 'excellent' if water_cut < 5 else 'good' if water_cut < 20 else 'fair' if water_cut < 50 else 'poor'
                    },
                    {
                        'name': 'Gas-Oil Ratio (GOR)',
                        'value': f"{gor:.2f}",
                        'unit': 'scf/bbl',
                        'benchmark': '<500',
                        'status': 'good' if gor < 500 else 'elevated' if gor < 1000 else 'high'
                    }
                ]
            },
            {
                'group': 'RESERVOIR CONDITIONS',
                'items': [
                    {
                        'name': 'Initial Reservoir Pressure',
                        'value': f"{initial_pressure:.1f}",
                        'unit': 'bar',
                        'benchmark': '1,500',
                        'status': 'normal'
                    },
                    {
                        'name': 'Current Reservoir Pressure',
                        'value': f"{current_pressure:.1f}",
                        'unit': 'bar',
                        'benchmark': f"{initial_pressure * 0.85:.0f}",
                        'status': 'good' if current_pressure > initial_pressure * 0.85 else 'declining'
                    },
                    {
                        'name': 'Absolute Pressure Decline',
                        'value': f"{initial_pressure - current_pressure:.1f}",
                        'unit': 'bar',
                        'benchmark': '<150',
                        'status': 'low' if (initial_pressure - current_pressure) < 150 else 'moderate' if (initial_pressure - current_pressure) < 400 else 'high'
                    },
                    {
                        'name': 'Relative Pressure Decline (%)',
                        'value': f"{pressure_decline:.2f}",
                        'unit': '%',
                        'benchmark': '<10%',
                        'status': 'excellent' if pressure_decline < 5 else 'good' if pressure_decline < 15 else 'significant' if pressure_decline < 30 else 'severe'
                    },
                    {
                        'name': 'Porosity',
                        'value': f"{porosity:.1f}",
                        'unit': '%',
                        'benchmark': '20',
                        'status': 'good' if 15 < porosity < 25 else 'fair'
                    },
                    {
                        'name': 'Permeability',
                        'value': f"{permeability:.0f}",
                        'unit': 'mD',
                        'benchmark': '100',
                        'status': 'excellent' if permeability > 200 else 'good' if permeability > 100 else 'fair' if permeability > 50 else 'limited'
                    }
                ]
            },
            {
                'group': 'WELL PRODUCTIVITY',
                'items': [
                    {
                        'name': 'Productivity Index',
                        'value': f"{productivity_index:.2f}",
                        'unit': 'bbl/day/psi',
                        'benchmark': '>0.5',
                        'status': 'excellent' if productivity_index > 1.5 else 'good' if productivity_index > 0.5 else 'fair' if productivity_index > 0.1 else 'low'
                    },
                    {
                        'name': 'Pressure Drawdown Sensitivity',
                        'value': f"{oil_rate / max(1, initial_pressure - current_pressure / 100):.1f}",
                        'unit': 'bbl/day/bar',
                        'benchmark': 'Variable',
                        'status': 'good'
                    }
                ]
            },
            {
                'group': 'MODEL QUALITY',
                'items': [
                    {
                        'name': 'History Match Quality',
                        'value': f"{match_quality:.1f}",
                        'unit': '%',
                        'benchmark': '>80%',
                        'status': 'excellent' if match_quality > 85 else 'good' if match_quality > 70 else 'fair' if match_quality > 55 else 'poor'
                    }
                ]
            }
        ]
        return metrics
    
    def _generate_executive_summary(self, well_perf: Dict, pressure_analysis: Dict,
                                   water_analysis: Dict, forecast: Dict, risks: Dict,
                                   match_quality: float) -> str:
        """Generate concise executive summary for quick understanding"""
        summary_points = []
        
        # Add key insights
        if 'assessment' in well_perf and len(well_perf['assessment']) > 0:
            oil_status = well_perf['assessment'][0].get('status', 'UNKNOWN')
            summary_points.append(f"Well performing at {oil_status} production level")
        
        if 'assessment' in pressure_analysis and len(pressure_analysis['assessment']) > 0:
            pressure_status = pressure_analysis['assessment'][0].get('insight', '')
            summary_points.append(pressure_status)
        
        if 'assessment' in water_analysis and len(water_analysis['assessment']) > 1:
            water_status = water_analysis['assessment'][1].get('insight', '')
            summary_points.append(water_status)
        
        if forecast.get('production_outlook'):
            summary_points.append(forecast['production_outlook'])
        
        overall_risk = risks.get('overall_risk_level', 'MODERATE')
        summary_points.append(f"Overall risk assessment: {overall_risk}")
        
        if match_quality < 65:
            summary_points.append(f"Note: Model match quality ({match_quality:.1f}%) suggests results should be interpreted with caution.")
        
        return ' | '.join(summary_points)


# Singleton instance
interpreter = ReservoirInterpretationEngine()


def interpret_simulation_results(simulation_data: Dict) -> Dict:
    """
    Public API for simulation interpretation
    
    Args:
        simulation_data: Dictionary with simulation results and parameters
        
    Returns:
        Comprehensive interpretation dictionary
    """
    return interpreter.interpret_simulation(simulation_data)
