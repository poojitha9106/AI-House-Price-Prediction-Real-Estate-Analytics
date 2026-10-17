"""
=============================================================================
HOUSE PRICE PREDICTION - PREDICTOR UTILITY & VALIDATION MODULE
=============================================================================
Purpose:
  Provides input validation for Indian cities/villages, loads the trained 
  Scikit-Learn model pipeline (.pkl), executes inference, formats 
  predictions in Indian Rupee (₹) currency with Lakhs/Crores notations,
  and computes Explainable AI (XAI) feature contributions using the trained model.
=============================================================================
"""

import os
import joblib
import pandas as pd
import numpy as np

# Valid allowed Indian locations (Cities & Villages)
VALID_LOCATIONS = [
    'Mumbai - Bandra West',
    'Bengaluru - Indiranagar',
    'Delhi - Vasant Kunj',
    'Gurugram - Golf Course Rd',
    'Hyderabad - Gachibowli',
    'Pune - Baner',
    'Jaipur - Vaishali Nagar',
    'Ahmedabad - Bodakdev',
    'Kochi - Edappally',
    'Chandigarh - Sector 17',
    'Lucknow - Gomti Nagar',
    'Baramati Town (MH)',
    'Rampur (UP)',
    'Palampur Village (HP)',
    'Mandya Village (KA)',
    'Wayanad Village (KL)',
    'Anakapalle (AP)'
]

VALID_PARKING = ['None', 'Two-Wheeler Only', 'Covered Car Park', 'Private Multi-Car Garage']
VALID_CONDITIONS = [
    'Fixer-Upper / Old',
    'Standard / Semi-Furnished',
    'Good / Well Maintained',
    'Fully Furnished / Renovated',
    'Luxury Villa / Ultra-Modern'
]

# Baseline representative reference house in India (used for marginal feature attribution)
BASELINE_FEATURES = {
    'city_village': 'Jaipur - Vaishali Nagar',  # Mid-tier reference city
    'area_sqyards': 200.0,
    'area_sqft': 1800.0,
    'bedrooms': 3,
    'bathrooms': 2.0,
    'floors': 2,
    'year_built': 2010,
    'parking': 'Covered Car Park',
    'condition': 'Good / Well Maintained'
}

FEATURE_FRIENDLY_NAMES = {
    'city_village': 'Location',
    'area_sqyards': 'Plot Area',
    'condition': 'Property Condition',
    'year_built': 'Construction Year',
    'parking': 'Parking Facility',
    'bedrooms': 'Bedrooms (BHK)',
    'bathrooms': 'Bathrooms',
    'floors': 'Total Floors'
}

MODEL_PIPELINE = None

def get_model():
    """
    Lazy loader for the saved Scikit-Learn model pipeline (.pkl file).
    """
    global MODEL_PIPELINE
    if MODEL_PIPELINE is None:
        model_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "models", "house_price_pipeline.pkl"
        )
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}. Please run src/train_model.py first.")
        MODEL_PIPELINE = joblib.load(model_path)
    return MODEL_PIPELINE

def format_inr(number):
    """
    Formats a numeric value into standard Indian Rupee currency format (e.g. ₹45,25,000).
    """
    val = int(round(number))
    s = str(abs(val))
    if len(s) <= 3:
        formatted = s
    else:
        last_three = s[-3:]
        other_digits = s[:-3]
        res = ""
        for i, d in enumerate(reversed(other_digits)):
            if i > 0 and i % 2 == 0:
                res = "," + res
            res = d + res
        formatted = f"{res},{last_three}"
    
    prefix = "-₹" if val < 0 else "₹"
    return f"{prefix}{formatted}"

def format_inr_words(number):
    """
    Formats numeric INR value into Indian notation (Lakhs / Crores).
    Example: 4525000 -> "₹45.25 Lakhs", 12500000 -> "₹1.25 Crore"
    """
    val = float(abs(number))
    sign = "-" if number < 0 else ""
    if val >= 10000000:
        cr = val / 10000000
        return f"{sign}₹{cr:.2f} Crore"
    elif val >= 100000:
        lakh = val / 100000
        return f"{sign}₹{lakh:.2f} Lakhs"
    else:
        return f"{sign}{format_inr(val)}"

def validate_input(data):
    """
    Validates user input fields for correct data types and allowed boundaries.
    """
    errors = []
    
    # 1. City / Village Location Validation
    city_village = str(data.get('city_village') or data.get('location', '')).strip()
    if city_village not in VALID_LOCATIONS:
        match = next((loc for loc in VALID_LOCATIONS if loc.lower() == city_village.lower()), None)
        if match:
            city_village = match
        else:
            errors.append(f"Location must be one of available Indian cities/villages: {', '.join(VALID_LOCATIONS[:5])}...")
        
    # 2. Area Validation (Square Yards / Square Feet)
    area_sqyards = None
    area_sqft = None
    
    raw_sqyards = data.get('area_sqyards')
    raw_sqft = data.get('area_sqft')
    
    if raw_sqyards is not None and str(raw_sqyards).strip() != '':
        try:
            area_sqyards = float(raw_sqyards)
            if area_sqyards < 20 or area_sqyards > 5000:
                errors.append("Area in Square Yards must be between 20 and 5,000 Gaj.")
            area_sqft = round(area_sqyards * 9, 1)
        except (ValueError, TypeError):
            errors.append("Area in Square Yards must be a valid number.")
    elif raw_sqft is not None and str(raw_sqft).strip() != '':
        try:
            area_sqft = float(raw_sqft)
            if area_sqft < 180 or area_sqft > 45000:
                errors.append("Area in Square Feet must be between 180 and 45,000 sq ft.")
            area_sqyards = round(area_sqft / 9, 1)
        except (ValueError, TypeError):
            errors.append("Area in Square Feet must be a valid number.")
    else:
        errors.append("Please provide either Area in Square Yards or Area in Square Feet.")
        
    # 3. Bedrooms Validation
    try:
        bedrooms = int(data.get('bedrooms', 0))
        if bedrooms < 1 or bedrooms > 10:
            errors.append("Number of bedrooms must be an integer between 1 and 10.")
    except (ValueError, TypeError):
        errors.append("Number of bedrooms must be a valid integer.")
        bedrooms = None
        
    # 4. Bathrooms Validation
    try:
        bathrooms = float(data.get('bathrooms', 0))
        if bathrooms < 0.5 or bathrooms > 10.0:
            errors.append("Number of bathrooms must be between 0.5 and 10.0.")
    except (ValueError, TypeError):
        errors.append("Number of bathrooms must be a valid number.")
        bathrooms = None
        
    # 5. Floors Validation
    try:
        floors = int(data.get('floors', 0))
        if floors < 1 or floors > 5:
            errors.append("Number of floors must be an integer between 1 and 5.")
    except (ValueError, TypeError):
        errors.append("Number of floors must be a valid integer.")
        floors = None
        
    # 6. Year Built Validation
    try:
        year_built = int(data.get('year_built', 0))
        if year_built < 1950 or year_built > 2026:
            errors.append("Year built must be between 1950 and 2026.")
    except (ValueError, TypeError):
        errors.append("Year built must be a valid 4-digit year.")
        year_built = None
        
    # 7. Parking Validation
    parking = str(data.get('parking', '')).strip()
    if parking not in VALID_PARKING:
        match_park = next((p for p in VALID_PARKING if p.lower() == parking.lower()), None)
        if match_park:
            parking = match_park
        else:
            errors.append(f"Parking must be one of: {', '.join(VALID_PARKING)}")
        
    # 8. House Condition Validation
    condition = str(data.get('condition', '')).strip()
    if condition not in VALID_CONDITIONS:
        match_cond = next((c for c in VALID_CONDITIONS if c.lower() == condition.lower()), None)
        if match_cond:
            condition = match_cond
        else:
            errors.append(f"House condition must be one of: {', '.join(VALID_CONDITIONS)}")
        
    if errors:
        raise ValueError("; ".join(errors))
        
    return {
        'city_village': city_village,
        'area_sqyards': area_sqyards,
        'area_sqft': area_sqft,
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'floors': floors,
        'year_built': year_built,
        'parking': parking,
        'condition': condition
    }

def explain_prediction(clean_data, predicted_price):
    """
    Computes Explainable AI (XAI) feature attribution directly using the trained model
    by calculating counterfactual marginal deviations from baseline real-estate attributes.
    """
    pipeline = get_model()
    
    features_to_check = ['city_village', 'area_sqyards', 'condition', 'year_built', 'parking', 'bedrooms', 'bathrooms', 'floors']
    contributions = []
    
    for feat in features_to_check:
        current_val = clean_data[feat]
        base_val = BASELINE_FEATURES[feat]
        
        # Create counterfactual sample with baseline replacement
        cf_data = clean_data.copy()
        cf_data[feat] = base_val
        if feat == 'area_sqyards':
            cf_data['area_sqft'] = BASELINE_FEATURES['area_sqft']
            
        cf_pred = float(pipeline.predict(pd.DataFrame([cf_data]))[0])
        delta = predicted_price - cf_pred
        
        # Only include non-negligible shifts (> 15,000 INR)
        if abs(delta) >= 15000:
            contributions.append({
                'feature': feat,
                'title': FEATURE_FRIENDLY_NAMES[feat],
                'current_value': str(current_val),
                'delta': delta,
                'impact_formatted': f"+{format_inr_words(delta)}" if delta > 0 else f"-{format_inr_words(abs(delta))}",
                'delta_inr': f"+{format_inr(delta)}" if delta > 0 else f"-{format_inr(abs(delta))}"
            })
            
    # Separate into increasing and decreasing factors
    increasing = [c for c in contributions if c['delta'] > 0]
    increasing.sort(key=lambda x: x['delta'], reverse=True)
    
    decreasing = [c for c in contributions if c['delta'] < 0]
    decreasing.sort(key=lambda x: x['delta']) # most negative first
    
    # Add human-friendly contextual explanations
    for item in increasing:
        f = item['feature']
        val = item['current_value']
        if f == 'city_village':
            item['reason'] = f"Premium location in {val} commands higher land circle rates and heavy buyer demand."
        elif f == 'area_sqyards':
            item['reason'] = f"Generous plot size of {val} Gaj ({clean_data['area_sqft']} sqft) adds substantial land valuation."
        elif f == 'condition':
            item['reason'] = f"{val} condition eliminates buyer renovation expense and commands top market premium."
        elif f == 'year_built':
            item['reason'] = f"Recent construction ({val}) enjoys modern building codes, fresh fixtures, and minimal depreciation."
        elif f == 'parking':
            item['reason'] = f"{val} is a prime convenience feature in Indian urban/suburban centers."
        elif f == 'bedrooms':
            item['reason'] = f"{val} BHK layout caters to larger families, elevating demand."
        else:
            item['reason'] = f"{item['title']} adds positive value to the property."

    for item in decreasing:
        f = item['feature']
        val = item['current_value']
        if f == 'city_village':
            item['reason'] = f"{val} has lower rural or tier-3 circle rates compared to metropolitan commercial zones."
        elif f == 'area_sqyards':
            item['reason'] = f"Compact plot size of {val} Gaj ({clean_data['area_sqft']} sqft) reduces overall built-up footprint."
        elif f == 'condition':
            item['reason'] = f"{val} condition requires buyer renovation expenditure, discounting the market price."
        elif f == 'year_built':
            item['reason'] = f"Built in {val} incurs structural age depreciation over decades."
        elif f == 'parking':
            item['reason'] = f"{val} parking limits vehicle security compared to covered car garages."
        elif f == 'bedrooms':
            item['reason'] = f"Compact {val} BHK layout reduces multi-person family capacity."
        else:
            item['reason'] = f"{item['title']} reduces the property market valuation."

    # Generate short natural-language narrative
    summary_parts = []
    if increasing:
        top_pos = increasing[0]
        pos_desc = f"{top_pos['title']} ({top_pos['current_value']}) is the strongest positive driver adding approx {top_pos['impact_formatted']}"
        if len(increasing) > 1:
            pos_desc += f", followed by {increasing[1]['title']} ({increasing[1]['impact_formatted']})"
        summary_parts.append(pos_desc + ".")
        
    if decreasing:
        top_neg = decreasing[0]
        neg_desc = f"{top_neg['title']} ({top_neg['current_value']}) pulls the estimate down by approx {top_neg['impact_formatted']}"
        summary_parts.append(neg_desc + ".")
        
    if not summary_parts:
        natural_explanation = "The property specifications closely align with standard baseline market averages across all major features."
    else:
        natural_explanation = " ".join(summary_parts)
        
    return {
        'natural_explanation': natural_explanation,
        'increasing_factors': increasing,
        'decreasing_factors': decreasing
    }

def predict_house_price(raw_input_data):
    """
    Validates input, constructs feature DataFrame, executes model inference,
    formats valuation in INR, and computes AI prediction explanations.
    """
    # 1. Validate inputs
    clean_data = validate_input(raw_input_data)
    
    # 2. Convert to 1-row Pandas DataFrame matching model expected columns
    input_df = pd.DataFrame([clean_data])
    
    # 3. Load model pipeline and predict
    pipeline = get_model()
    predicted_price = float(pipeline.predict(input_df)[0])
    
    # 4. Format price & metrics
    price_rounded = round(max(250000, predicted_price), -3)
    price_inr_formatted = format_inr(price_rounded)
    price_words = format_inr_words(price_rounded)
    
    price_per_sqyd = round(price_rounded / clean_data['area_sqyards'], 0)
    price_per_sqft = round(price_rounded / clean_data['area_sqft'], 0)
    
    # 5. Valuation Range (+/- 3.5%)
    lower_bound = round(price_rounded * 0.965, -3)
    upper_bound = round(price_rounded * 1.035, -3)
    
    # 6. Compute AI Explainability (XAI)
    explanation = explain_prediction(clean_data, price_rounded)
    
    return {
        'predicted_price': price_rounded,
        'price_formatted': price_inr_formatted,
        'price_words': price_words,
        'price_per_sqyard': price_per_sqyd,
        'price_per_sqyard_formatted': f"{format_inr(price_per_sqyd)}/sq.yd (Gaj)",
        'price_per_sqft': price_per_sqft,
        'price_per_sqft_formatted': f"{format_inr(price_per_sqft)}/sq.ft",
        'valuation_range': {
            'min': lower_bound,
            'max': upper_bound,
            'range_formatted': f"{format_inr(lower_bound)} - {format_inr(upper_bound)}",
            'range_words': f"{format_inr_words(lower_bound)} - {format_inr_words(upper_bound)}"
        },
        'input_summary': clean_data,
        'explanation': explanation
    }

if __name__ == '__main__':
    sample = {
        'city_village': 'Hyderabad - Gachibowli',
        'area_sqyards': 250,
        'bedrooms': 3,
        'bathrooms': 3.0,
        'floors': 2,
        'year_built': 2021,
        'parking': 'Covered Car Park',
        'condition': 'Fully Furnished / Renovated'
    }
    result = predict_house_price(sample)
    p_words = result['price_words'].replace('₹', 'Rs. ')
    p_fmt = result['price_formatted'].replace('₹', 'Rs. ')
    print(f"Sample Prediction Test: {p_words} ({p_fmt})")
    print(f"Rate: {result['price_per_sqyard_formatted'].replace('₹', 'Rs. ')} | {result['price_per_sqft_formatted'].replace('₹', 'Rs. ')}")
    print("AI Explanation:", result['explanation']['natural_explanation'].replace('₹', 'Rs. '))
