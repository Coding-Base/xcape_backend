"""
Data parsing utilities for production data CSV files
"""
import csv
import json
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)


def parse_production_csv(file_path: str) -> Dict[str, Any]:
    """
    Parse production data CSV file and convert to dictionary format.
    
    Expected CSV columns:
    - Days: Time step (days)
    - Oil_bbl: Oil production (bbl/day)
    - Water_bbl: Water production (bbl/day)
    - Gas_scf: Gas production (scf/day)
    - Pressure_psi: Reservoir pressure (psi)
    - Cumulative_Oil_bbl: Cumulative oil (bbl)
    - Cumulative_Water_bbl: Cumulative water (bbl)
    - Cumulative_Gas_scf: Cumulative gas (scf)
    
    Args:
        file_path: Path to CSV file
        
    Returns:
        Dictionary with parsed data and metadata
    """
    data = {
        'days': [],
        'Oil_bbl': [],
        'Water_bbl': [],
        'Gas_scf': [],
        'Pressure_psi': [],
        'Cumulative_Oil_bbl': [],
        'Cumulative_Water_bbl': [],
        'Cumulative_Gas_scf': []
    }
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            if not reader.fieldnames:
                raise ValueError("CSV file is empty or has no header")
            
            # Map possible column name variations
            column_mapping = {
                'days': ['Days', 'days', 'Day', 'day'],
                'Oil_bbl': ['Oil_bbl', 'oil', 'Oil', 'OilRate'],
                'Water_bbl': ['Water_bbl', 'water', 'Water', 'WaterRate'],
                'Gas_scf': ['Gas_scf', 'gas', 'Gas', 'GasRate'],
                'Pressure_psi': ['Pressure_psi', 'pressure', 'Pressure', 'Pres'],
                'Cumulative_Oil_bbl': ['Cumulative_Oil_bbl', 'CumulativeOil', 'CumOil'],
                'Cumulative_Water_bbl': ['Cumulative_Water_bbl', 'CumulativeWater', 'CumWater'],
                'Cumulative_Gas_scf': ['Cumulative_Gas_scf', 'CumulativeGas', 'CumGas']
            }
            
            # Find actual column names in CSV
            actual_columns = {}
            for standard_name, aliases in column_mapping.items():
                for alias in aliases:
                    if alias in reader.fieldnames:
                        actual_columns[standard_name] = alias
                        break
            
            logger.info(f"CSV columns found: {list(reader.fieldnames)}")
            logger.info(f"Mapped to: {actual_columns}")
            
            # Parse rows
            row_count = 0
            for row in reader:
                row_count += 1
                try:
                    for output_key, input_key in actual_columns.items():
                        value = row.get(input_key, '').strip()
                        if value:
                            data[output_key].append(float(value))
                        else:
                            # Use zero for missing values
                            data[output_key].append(0.0)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error parsing row {row_count}: {e}")
                    continue
            
            if row_count == 0:
                raise ValueError("No data rows found in CSV")
            
            # Validate data
            data_lengths = {k: len(v) for k, v in data.items() if v}
            if len(set(data_lengths.values())) > 1:
                logger.warning(f"Data columns have different lengths: {data_lengths}")
            
            metadata = {
                'rows': row_count,
                'days': len(data['days']),
                'oil_total': sum(data['Oil_bbl']) if data['Oil_bbl'] else 0,
                'water_total': sum(data['Water_bbl']) if data['Water_bbl'] else 0,
                'gas_total': sum(data['Gas_scf']) if data['Gas_scf'] else 0,
                'initial_pressure': data['Pressure_psi'][0] if data['Pressure_psi'] else None,
                'final_pressure': data['Pressure_psi'][-1] if data['Pressure_psi'] else None,
            }
            
            logger.info(f"Successfully parsed {row_count} rows of production data")
            logger.info(f"Metadata: {metadata}")
            
            return {
                'status': 'success',
                'data': data,
                'metadata': metadata
            }
            
    except Exception as e:
        logger.error(f"Error parsing CSV file {file_path}: {e}")
        return {
            'status': 'error',
            'error': str(e),
            'data': None
        }


def validate_production_data(data: Dict[str, Any]) -> tuple[bool, str]:
    """
    Validate parsed production data for use in simulation.
    
    Args:
        data: Dictionary with parsed production data
        
    Returns:
        Tuple of (is_valid, message)
    """
    required_fields = ['Oil_bbl', 'Water_bbl', 'Gas_scf', 'Pressure_psi']
    
    # Check all required fields exist
    for field in required_fields:
        if field not in data or not data[field]:
            return False, f"Missing required field: {field}"
    
    # Check all fields have same length
    lengths = {k: len(v) for k, v in data.items() if k in required_fields}
    if len(set(lengths.values())) > 1:
        return False, f"Data fields have inconsistent lengths: {lengths}"
    
    # Check minimum data points
    num_points = len(data['Oil_bbl'])
    if num_points < 5:
        return False, f"Insufficient data points: {num_points} (minimum 5 required)"
    
    if num_points > 10000:
        return False, f"Too many data points: {num_points} (maximum 10000)"
    
    # Check for valid numeric values
    all_fields = [v for k, v in data.items() if k in required_fields]
    for field_values in all_fields:
        if any(v < 0 for v in field_values if isinstance(v, (int, float))):
            return False, "Found negative production values"
    
    return True, "Data validation passed"
