import flask
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

# Load artifacts
print("Loading model and preprocessors...")
try:
    model = joblib.load('best_model.pkl')
    scaler = joblib.load('scaler.pkl')
    label_encoders = joblib.load('label_encoders.pkl')
except FileNotFoundError:
    print("Error: Artifacts not found. Please run 'model_training.py' first.")
    exit(1)

# Define feature order (must match training order)
# Based on the CSV structure (excluding dropout)
FEATURE_ORDER = [
    'jee_main_score', 'jee_advanced_score', 'mock_test_score_avg', 'school_board', 
    'class_12_percent', 'attempt_count', 'coaching_institute', 'daily_study_hours', 
    'family_income', 'parent_education', 'location_type', 'peer_pressure_level', 
    'mental_health_issues', 'admission_taken'
]

NUMERICAL_FEATURES = [
    'jee_main_score', 'jee_advanced_score', 'mock_test_score_avg', 
    'class_12_percent', 'attempt_count', 'daily_study_hours'
]

CATEGORICAL_FEATURES = [
    'school_board', 'coaching_institute', 'family_income', 'parent_education', 
    'location_type', 'peer_pressure_level', 'mental_health_issues', 'admission_taken'
]

def get_clean_categorical_value(val, classes):
    for c in classes:
        # Match Nan or empty values to nan in encoder classes
        if pd.isna(c) and (val == 'None' or val == 'nan' or val is None or val == ''):
            return c
        if str(c) == str(val):
            return c
    return val

def is_class_match(val, classes):
    for c in classes:
        if (pd.isna(c) and pd.isna(val)) or c == val:
            return True
    return False

# A neutral/baseline student profile to compare actual inputs against (used in XAI calculations)
BASELINE_STUDENT = {
    'jee_main_score': 75.0,
    'jee_advanced_score': 55.0,
    'mock_test_score_avg': 65.0,
    'school_board': 'CBSE',
    'class_12_percent': 75.0,
    'attempt_count': 1.0,
    'coaching_institute': 'FIITJEE',
    'daily_study_hours': 5.0,
    'family_income': 'Mid',
    'parent_education': 'Graduate',
    'location_type': 'Semi-Urban',
    'peer_pressure_level': 'Medium',
    'mental_health_issues': 'No',
    'admission_taken': 'No'
}

def explain_prediction_xai(model, scaler, label_encoders, input_dict, base_dropout_prob):
    # Model-agnostic local feature attribution (Kernel-SHAP / LIME style)
    # We substitute each feature's value with a baseline student's value one-by-one, 
    # run inference, and measure the difference in dropout probability.
    explanations = []
    
    # Run a perturbation loop
    for feature in FEATURE_ORDER:
        # Create a modified profile starting from the actual input_dict
        modified_dict = input_dict.copy()
        
        # Substitute the baseline value for THIS feature
        modified_dict[feature] = BASELINE_STUDENT[feature]
        
        # Process and scale the modified dict
        try:
            modified_df = pd.DataFrame([modified_dict])
            
            # Encode categorical
            for col in CATEGORICAL_FEATURES:
                le = label_encoders[col]
                cleaned_val = get_clean_categorical_value(modified_dict[col], le.classes_)
                if is_class_match(cleaned_val, le.classes_):
                    modified_df[col] = le.transform([cleaned_val])
                else:
                    # Fallback to zero code
                    modified_df[col] = 0
            
            # Scale numerical
            modified_df[NUMERICAL_FEATURES] = scaler.transform(modified_df[NUMERICAL_FEATURES])
            
            # Run model inference to get perturbed probability
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(modified_df)[0]
                perturbed_dropout_prob = float(probs[1] if len(probs) > 1 else 0)
            else:
                pred = model.predict(modified_df)[0]
                perturbed_dropout_prob = 1.0 if pred == 1 else 0.0
                
            # The contribution of the student's ACTUAL value is:
            # P(actual) - P(baseline_substituted)
            # A positive contribution means this feature increases dropout risk.
            # A negative contribution means this feature decreases dropout risk.
            contribution = base_dropout_prob - (perturbed_dropout_prob * 100)
            
            # Format the output value to represent the clean user-facing name
            display_name = feature.replace('_', ' ').replace('avg', '').title()
            
            # Clean up value display
            actual_val = input_dict[feature]
            if feature in NUMERICAL_FEATURES:
                if feature == 'daily_study_hours':
                    actual_display = f"{float(actual_val):.1f} hrs"
                elif feature in ['jee_main_score', 'jee_advanced_score', 'mock_test_score_avg', 'class_12_percent']:
                    actual_display = f"{float(actual_val):.1f}%"
                else:
                    actual_display = str(int(float(actual_val)))
            else:
                actual_display = str(actual_val)
                
            explanations.append({
                "feature": feature,
                "display_name": display_name,
                "actual_value": actual_display,
                "contribution": round(contribution, 2)
            })
            
        except Exception as e:
            # Silently skip if there's any perturbation failure
            print(f"Perturbation failed for {feature}: {e}")
            continue
            
    # Sort contributions by absolute impact (highest impact first)
    explanations.sort(key=lambda x: abs(x["contribution"]), reverse=True)
    return explanations

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    probability = None
    input_data = {}
    
    # Prepare options for dropdowns, mapping nan to 'None' for user readability
    dropdown_options = {col: [str(x) if not pd.isna(x) else 'None' for x in le.classes_] for col, le in label_encoders.items()}

    if request.method == 'POST':
        try:
            # Collect input data
            processed_input = []
            
            # Create a DataFrame for easy processing
            input_dict = {}
            
            for feature in FEATURE_ORDER:
                val = request.form.get(feature)
                input_dict[feature] = val
                
                # Check for empty input
                if not val:
                    return render_template('index.html', error=f"Missing value for {feature}", options=dropdown_options)

            # Process Data
            input_df = pd.DataFrame([input_dict])
            
            # Encode Categorical
            for col in CATEGORICAL_FEATURES:
                le = label_encoders[col]
                cleaned_val = get_clean_categorical_value(input_dict[col], le.classes_)
                
                if is_class_match(cleaned_val, le.classes_):
                    input_df[col] = le.transform([cleaned_val])
                else:
                    return render_template('index.html', error=f"Invalid value for {col}", options=dropdown_options)

            # Scale Numerical
            input_df[NUMERICAL_FEATURES] = scaler.transform(input_df[NUMERICAL_FEATURES])
            
            # Predict
            pred = model.predict(input_df)[0]
            # Try to get probability if supported
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(input_df)[0]
                probability = probs[1] if len(probs) > 1 else 0
                probability = round(probability * 100, 2)
            
            prediction = "Dropout" if pred == 1 else "Not Dropout"
            
            # Pass back input to repopulate form
            input_data = input_dict

        except Exception as e:
            return render_template('index.html', error=str(e), options=dropdown_options)

    return render_template('index.html', prediction=prediction, probability=probability, options=dropdown_options, input_data=input_data)

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        # Check if request is JSON or form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        # Collect and validate input data
        input_dict = {}
        for feature in FEATURE_ORDER:
            val = data.get(feature)
            if val is None or val == '':
                # Map standard names for customer readability
                display_name = feature.replace('_', ' ').replace('avg', '').title()
                return flask.jsonify({"error": f"Missing value for '{display_name}'"}), 400
            
            # Convert numeric features to appropriate float/int types
            if feature in NUMERICAL_FEATURES:
                try:
                    input_dict[feature] = float(val)
                except ValueError:
                    return flask.jsonify({"error": f"Invalid number value for '{feature.replace('_', ' ').title()}'"}), 400
            else:
                input_dict[feature] = str(val)

        # Process Data into DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Encode Categorical
        for col in CATEGORICAL_FEATURES:
            le = label_encoders[col]
            cleaned_val = get_clean_categorical_value(input_dict[col], le.classes_)
            
            if is_class_match(cleaned_val, le.classes_):
                input_df[col] = le.transform([cleaned_val])
            else:
                return flask.jsonify({"error": f"Invalid category value '{input_dict[col]}' for '{col.replace('_', ' ').title()}'"}), 400

        # Scale Numerical
        input_df[NUMERICAL_FEATURES] = scaler.transform(input_df[NUMERICAL_FEATURES])
        
        # Predict
        pred = int(model.predict(input_df)[0])
        probability = 0.0
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(input_df)[0]
            probability = float(probs[1] if len(probs) > 1 else 0)
            probability = round(probability * 100, 2)
        
        prediction = "Dropout" if pred == 1 else "Not Dropout"
        
        # Calculate XAI Local Feature Attributions
        xai_explanations = explain_prediction_xai(model, scaler, label_encoders, input_dict, probability)
        
        return flask.jsonify({
            "success": True,
            "prediction": prediction,
            "probability": probability,
            "input_data": input_dict,
            "xai_explanations": xai_explanations
        })

    except Exception as e:
        return flask.jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
