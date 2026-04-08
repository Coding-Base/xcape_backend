"""
Quick utility to load SPE10 production data into XCAPE for demonstration.

Usage:
    python setup_spe10_data.py
"""
import os
import sys
import django
import csv

# Setup Django
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'XCAPE.settings')
django.setup()

from django.contrib.auth import get_user_model
from simulations.models import Dataset

User = get_user_model()

def load_spe10_production_data():
    """Load SPE10 production CSV into database"""
    
    csv_file = os.path.join(
        os.path.dirname(__file__),
        '..', '..', '..',
        'test_data', 'spe10_case1_production.csv'
    )
    
    print(f"[*] Loading SPE10 production data from: {csv_file}")
    
    if not os.path.exists(csv_file):
        print(f"[ERROR] File not found: {csv_file}")
        print(f"[HINT] Generate it first: python generate_spe10.py")
        return False
    
    # Parse CSV
    data = {
        'Days': [],
        'Oil_bbl': [],
        'Water_bbl': [],
        'Gas_scf': [],
        'Pressure_psi': [],
        'Cumulative_Oil_bbl': [],
        'Cumulative_Water_bbl': [],
        'Cumulative_Gas_scf': []
    }
    
    row_count = 0
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                data['Days'].append(float(row['Days']))
                data['Oil_bbl'].append(float(row['Oil_bbl']))
                data['Water_bbl'].append(float(row['Water_bbl']))
                data['Gas_scf'].append(float(row['Gas_scf']))
                data['Pressure_psi'].append(float(row['Pressure_psi']))
                data['Cumulative_Oil_bbl'].append(float(row.get('Cumulative_Oil_bbl', 0)))
                data['Cumulative_Water_bbl'].append(float(row.get('Cumulative_Water_bbl', 0)))
                data['Cumulative_Gas_scf'].append(float(row.get('Cumulative_Gas_scf', 0)))
                row_count += 1
            except (ValueError, KeyError) as e:
                print(f"[!] Skipped row: {e}")
                continue
    
    if row_count == 0:
        print("[ERROR] No valid data rows found in CSV")
        return False
    
    print(f"[OK] Parsed {row_count} rows of production data")
    print(f"[OK] Days: {data['Days'][0]:.0f} to {data['Days'][-1]:.0f}")
    print(f"[OK] Oil: {data['Oil_bbl'][0]:.0f} to {data['Oil_bbl'][-1]:.0f} bbl/day")
    print(f"[OK] Pressure: {data['Pressure_psi'][0]:.0f} to {data['Pressure_psi'][-1]:.0f} psi")
    
    # Get or create admin user
    admin_user = User.objects.filter(is_superuser=True).first()
    if not admin_user:
        print("[ERROR] No admin user found. Create one first: python manage.py createsuperuser")
        return False
    
    # Create or update Dataset
    dataset, created = Dataset.objects.update_or_create(
        user=admin_user,
        name='SPE10 Case 1 - Real Benchmark',
        defaults={
            'description': 'Real SPE10 Case 1 benchmark permeability field with 365 days of synthetic production history',
            'filename': 'spe10_case1_production.csv',
            'file_size': os.path.getsize(csv_file),
            'production_data': data
        }
    )
    
    action = "CREATED" if created else "UPDATED"
    print(f"[*] Dataset {action}: {dataset.name}")
    print(f"[OK] Production data linked to user: {admin_user.username}")
    print(f"\n[READY] To use this data:")
    print(f"  1. Login to XCAPE as {admin_user.username}")
    print(f"  2. Go to Simulator > Configuration")
    print(f"  3. Run EnKF + Forecasts")
    print(f"  4. Check Match Quality - should be >70% for real data")
    
    return True


if __name__ == '__main__':
    try:
        success = load_spe10_production_data()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
