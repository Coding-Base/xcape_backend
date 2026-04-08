"""
Quick verification that SPE10 data is properly set up in XCAPE.

Usage:
    python manage.py shell < verify_spe10.py
    
Or standalone:
    python verify_spe10_standalone.py
"""
import os
import sys

def verify_spe10_setup():
    """Verify SPE10 data setup"""
    
    print("\n" + "="*60)
    print("SPE10 DATA VERIFICATION")
    print("="*60)
    
    checks = {
        'files': [],
        'database': [],
        'data': []
    }
    
    # Check 1: Files exist
    print("\n[1/3] Checking files...")
    files_to_check = [
        ('Permeability field', r'..\..\test_data\perm_case1\perm_case1.dat'),
        ('Production data', r'..\..\test_data\spe10_case1_production.csv'),
        ('Documentation', r'..\..\test_data\SPE10_README.md'),
    ]
    
    for name, path in files_to_check:
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f"  [OK] {name}: {size:,} bytes")
            checks['files'].append((name, True))
        else:
            print(f"  [FAIL] {name}: NOT FOUND")
            checks['files'].append((name, False))
    
    # Check 2: Database
    print("\n[2/3] Checking database...")
    try:
        import django
        os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'XCAPE.settings')
        django.setup()
        
        from django.contrib.auth import get_user_model
        from simulations.models import Dataset
        
        User = get_user_model()
        
        # Check admin user
        admin_count = User.objects.filter(is_superuser=True).count()
        print(f"  [OK] Admin users: {admin_count}")
        checks['database'].append(('Admin users', admin_count > 0))
        
        # Check SPE10 dataset
        spé10_datasets = Dataset.objects.filter(name__icontains='SPE10')
        print(f"  [OK] SPE10 datasets in DB: {spé10_datasets.count()}")
        
        if spé10_datasets.exists():
            for ds in spé10_datasets:
                prod_data = ds.production_data
                if prod_data:
                    print(f"    - {ds.name}")
                    print(f"      Oil records: {len(prod_data.get('Oil_bbl', []))}")
                    print(f"      Water records: {len(prod_data.get('Water_bbl', []))}")
                    print(f"      Gas records: {len(prod_data.get('Gas_scf', []))}")
                    checks['database'].append(('SPE10 dataset', True))
                else:
                    print(f"    [WARN] {ds.name} has no production_data")
                    checks['database'].append(('SPE10 dataset', False))
        else:
            print(f"  [INFO] No SPE10 datasets found (run setup_spe10_data.py)")
            checks['database'].append(('SPE10 dataset', False))
            
    except ImportError:
        print("  [SKIP] Django not available (run in manage.py shell)")
        checks['database'].append(('Database check', None))
    except Exception as e:
        print(f"  [ERROR] Database error: {e}")
        checks['database'].append(('Database check', False))
    
    # Check 3: Data validity
    print("\n[3/3] Checking data validity...")
    try:
        import csv
        
        csv_file = r'..\..\test_data\spe10_case1_production.csv'
        if os.path.exists(csv_file):
            with open(csv_file, 'r') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
            if rows:
                print(f"  [OK] CSV has {len(rows)} records")
                checks['data'].append(('CSV records', True))
                
                # Sample first and last
                first = rows[0]
                last = rows[-1]
                
                print(f"  [OK] First record: Day {first['Days']}, Oil {first['Oil_bbl']} bbl/day")
                print(f"  [OK] Last record: Day {last['Days']}, Oil {last['Oil_bbl']} bbl/day")
                
                # Check for required columns
                required = ['Oil_bbl', 'Water_bbl', 'Gas_scf', 'Pressure_psi']
                missing = [c for c in required if reader.fieldnames is None or c not in reader.fieldnames]
                if not missing:
                    print(f"  [OK] All required columns present")
                    checks['data'].append(('Required columns', True))
                else:
                    print(f"  [FAIL] Missing columns: {missing}")
                    checks['data'].append(('Required columns', False))
            else:
                print(f"  [FAIL] CSV is empty")
                checks['data'].append(('CSV records', False))
        else:
            print(f"  [FAIL] CSV file not found")
            checks['data'].append(('CSV file', False))
            
    except Exception as e:
        print(f"  [ERROR] Data check failed: {e}")
        checks['data'].append(('Data validation', False))
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    total_ok = sum(1 for v in 
                   checks['files'] + checks['database'] + checks['data'] 
                   if v[1] is True)
    total_checks = sum(1 for v in 
                       checks['files'] + checks['database'] + checks['data'] 
                       if v[1] is not None)
    
    print(f"Passed: {total_ok}/{total_checks} checks")
    
    if total_ok == total_checks and total_checks > 0:
        print("\n✓ SPE10 DATA READY FOR USE")
        print("  Go to XCAPE Simulator > Execution > Run EnKF + Forecasts")
        return True
    else:
        print("\n✗ SPE10 DATA SETUP INCOMPLETE")
        print("  Next steps:")
        print("  1. Ensure perm_case1.dat and production CSV exist")
        print("  2. Run: python setup_spe10_data.py")
        print("  3. Verify database has admin user")
        return False


if __name__ == '__main__':
    if 'django.conf' in sys.modules:
        # Running inside manage.py shell
        verify_spe10_setup()
    else:
        print("Run this inside manage.py shell:")
        print("  cd backend\\XCAPE")
        print("  python manage.py shell")
        print("  exec(open('verify_spe10.py').read())")
