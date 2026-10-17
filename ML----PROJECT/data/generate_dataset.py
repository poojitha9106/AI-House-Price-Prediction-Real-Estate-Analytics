"""
=============================================================================
HOUSE PRICE PREDICTION - INDIAN REAL ESTATE DATASET GENERATOR
=============================================================================
Purpose:
  Generates a realistic Indian real-estate dataset (`house_data.csv`) featuring
  real Indian Metropolitan Cities, Tier-2 Towns, and Villages, dual land area
  measurements in Square Yards (Gaj) and Square Feet (sq.ft), room counts, 
  parking, property condition, and price in Indian Rupees (₹).

Indian Locations Included:
  - Metros: Mumbai (Bandra West), Bengaluru (Indiranagar), Delhi (Vasant Kunj), 
            Gurugram (Golf Course Rd), Hyderabad (Gachibowli), Pune (Baner)
  - Tier-2/3 Cities: Jaipur (Vaishali Nagar), Ahmedabad (Bodakdev), Kochi (Edappally), 
                     Chandigarh (Sector 17), Lucknow (Gomti Nagar)
  - Towns & Villages: Palampur Village (HP), Mandya Village (KA), Rampur (UP), 
                      Baramati Town (MH), Wayanad Village (KL), Anakapalle (AP)
=============================================================================
"""

import os
import numpy as np
import pandas as pd

INDIAN_LOCATIONS = [
    # Metro Cities & Prime Zones
    'Mumbai - Bandra West',
    'Bengaluru - Indiranagar',
    'Delhi - Vasant Kunj',
    'Gurugram - Golf Course Rd',
    'Hyderabad - Gachibowli',
    'Pune - Baner',
    
    # Tier-2/3 Cities
    'Jaipur - Vaishali Nagar',
    'Ahmedabad - Bodakdev',
    'Kochi - Edappally',
    'Chandigarh - Sector 17',
    'Lucknow - Gomti Nagar',
    
    # Towns & Rural Villages
    'Baramati Town (MH)',
    'Rampur (UP)',
    'Palampur Village (HP)',
    'Mandya Village (KA)',
    'Wayanad Village (KL)',
    'Anakapalle (AP)'
]

LOCATION_WEIGHTS = [
    0.08, 0.10, 0.08, 0.08, 0.10, 0.08,  # Metros (52%)
    0.07, 0.06, 0.05, 0.05, 0.05,        # Tier-2/3 Cities (28%)
    0.05, 0.04, 0.04, 0.03, 0.02, 0.02   # Towns & Villages (20%)
]

# Real Estate Land Rate per Square Yard (Gaj) in INR
LOCATION_SQYARD_RATES = {
    # Metros (High Premium)
    'Mumbai - Bandra West': 320000,
    'Delhi - Vasant Kunj': 180000,
    'Gurugram - Golf Course Rd': 160000,
    'Bengaluru - Indiranagar': 135000,
    'Hyderabad - Gachibowli': 95000,
    'Pune - Baner': 82000,
    
    # Tier-2/3 Cities
    'Chandigarh - Sector 17': 75000,
    'Ahmedabad - Bodakdev': 62000,
    'Jaipur - Vaishali Nagar': 48000,
    'Kochi - Edappally': 42000,
    'Lucknow - Gomti Nagar': 38000,
    
    # Towns & Villages
    'Baramati Town (MH)': 25000,
    'Rampur (UP)': 18000,
    'Mandya Village (KA)': 14000,
    'Anakapalle (AP)': 12000,
    'Palampur Village (HP)': 10000,
    'Wayanad Village (KL)': 9000
}

PARKING_TYPES = ['None', 'Two-Wheeler Only', 'Covered Car Park', 'Private Multi-Car Garage']
PARKING_WEIGHTS = [0.15, 0.25, 0.40, 0.20]

CONDITIONS = ['Fixer-Upper / Old', 'Standard / Semi-Furnished', 'Good / Well Maintained', 'Fully Furnished / Renovated', 'Luxury Villa / Ultra-Modern']
CONDITION_WEIGHTS = [0.10, 0.30, 0.35, 0.18, 0.07]

CONDITION_MULTIPLIERS = {
    'Fixer-Upper / Old': 0.72,
    'Standard / Semi-Furnished': 0.88,
    'Good / Well Maintained': 1.05,
    'Fully Furnished / Renovated': 1.25,
    'Luxury Villa / Ultra-Modern': 1.55
}

PARKING_BONUS_INR = {
    'None': 0,
    'Two-Wheeler Only': 50000,
    'Covered Car Park': 250000,
    'Private Multi-Car Garage': 600000
}

def generate_indian_house_dataset(n_samples=2500, random_state=42):
    """
    Generates realistic Indian house prices based on city/village,
    square yards, bedrooms (BHK), bathrooms, floors, age, parking, and condition.
    """
    np.random.seed(random_state)
    
    # 1. Sample Locations
    loc_arr = np.random.choice(INDIAN_LOCATIONS, size=n_samples, p=LOCATION_WEIGHTS)
    
    # 2. Area in Square Yards (Gaj) - typically ranges from 60 Sq.Yards (small house) to 750 Sq.Yards (bungalow)
    sqyards_arr = np.random.normal(loc=220, scale=95, size=n_samples).astype(int)
    sqyards_arr = np.clip(sqyards_arr, 50, 850)
    
    # Square Feet = Square Yards * 9
    sqft_arr = sqyards_arr * 9
    
    # 3. Bedrooms / BHK (1 BHK to 5 BHK)
    bedrooms_arr = np.random.choice([1, 2, 3, 4, 5], size=n_samples, p=[0.10, 0.35, 0.38, 0.13, 0.04])
    
    # 4. Bathrooms
    bathrooms_arr = []
    for b in bedrooms_arr:
        b_count = np.random.choice([b*0.5, b*0.75, b*1.0], p=[0.2, 0.6, 0.2])
        bathrooms_arr.append(max(1.0, min(6.0, round(b_count, 1))))
    bathrooms_arr = np.array(bathrooms_arr)
    
    # 5. Floors (1 to 4 stories)
    floors_arr = np.random.choice([1, 2, 3, 4], size=n_samples, p=[0.45, 0.40, 0.10, 0.05])
    
    # 6. Year Built (1960 to 2026)
    year_built_arr = np.random.randint(1960, 2026, size=n_samples)
    
    # 7. Parking & Condition
    parking_arr = np.random.choice(PARKING_TYPES, size=n_samples, p=PARKING_WEIGHTS)
    condition_arr = np.random.choice(CONDITIONS, size=n_samples, p=CONDITION_WEIGHTS)
    
    # 8. Compute Price in Indian Rupees (₹)
    prices = []
    current_year = 2026
    
    for i in range(n_samples):
        loc = loc_arr[i]
        sqyd = sqyards_arr[i]
        bed = bedrooms_arr[i]
        bath = bathrooms_arr[i]
        flr = floors_arr[i]
        yr = year_built_arr[i]
        park = parking_arr[i]
        cond = condition_arr[i]
        
        # Base land & building value
        rate_per_sqyd = LOCATION_SQYARD_RATES[loc]
        land_val = sqyd * rate_per_sqyd
        
        # Rooms amenity value
        rooms_val = (bed * 180000) + (bath * 250000) + (flr * 150000)
        
        # Age depreciation / modern construction bonus
        age = current_year - yr
        if yr > 2018:
            age_factor = 1.08  # New modern construction
        elif yr > 2005:
            age_factor = 1.00
        elif yr > 1990:
            age_factor = 0.90
        else:
            age_factor = 0.82
            
        subtotal = (land_val + rooms_val + PARKING_BONUS_INR[park]) * CONDITION_MULTIPLIERS[cond] * age_factor
        
        # Add market fluctuation noise (+/- 3.5%)
        noise = np.random.normal(loc=0, scale=subtotal * 0.035)
        price_inr = round(max(350000, subtotal + noise), -3)  # Rounded to thousands
        prices.append(price_inr)
        
    df = pd.DataFrame({
        'city_village': loc_arr,
        'area_sqyards': sqyards_arr,
        'area_sqft': sqft_arr,
        'bedrooms': bedrooms_arr,
        'bathrooms': bathrooms_arr,
        'floors': floors_arr,
        'year_built': year_built_arr,
        'parking': parking_arr,
        'condition': condition_arr,
        'price': prices
    })
    
    # Introduce small missing value rate (~1.5%) for imputer validation
    nan_indices_area = np.random.choice(n_samples, size=int(n_samples * 0.015), replace=False)
    nan_indices_bath = np.random.choice(n_samples, size=int(n_samples * 0.015), replace=False)
    nan_indices_park = np.random.choice(n_samples, size=int(n_samples * 0.015), replace=False)
    
    df.loc[nan_indices_area, 'area_sqyards'] = np.nan
    df.loc[nan_indices_bath, 'bathrooms'] = np.nan
    df.loc[nan_indices_park, 'parking'] = np.nan
    
    return df

def inspect_dataset(df):
    """
    Prints inspection summary for Indian house dataset.
    """
    print("=" * 70)
    print("INDIAN REAL ESTATE DATASET SUMMARY (house_data.csv)")
    print("=" * 70)
    print(f"Total Rows: {len(df)}")
    print(f"Total Columns: {len(df.columns)}")
    print("\nColumns & Data Types:")
    print(df.dtypes)
    
    print("\nSample Indian City / Village Records:")
    print(df[['city_village', 'area_sqyards', 'bedrooms', 'bathrooms', 'price']].head(8))
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    print("=" * 70)

if __name__ == '__main__':
    data_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(data_dir, exist_ok=True)
    
    file_path = os.path.join(data_dir, "house_data.csv")
    alias_path = os.path.join(data_dir, "house_prices.csv")
    
    print("Generating dataset with real Indian cities, villages & Square Yards...")
    df = generate_indian_house_dataset(n_samples=2500)
    df.to_csv(file_path, index=False)
    df.to_csv(alias_path, index=False)
    
    print(f"Dataset successfully saved to {file_path}")
    inspect_dataset(df)
