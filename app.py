import flask
from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os
import lime
import lime.lime_tabular

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

# Initialize LIME Explainer
print("Initializing LIME Explainer...")
try:
    # Load training dataset for LIME references
    df_train = pd.read_csv(os.path.join('dataset', 'JEE_Dropout_After_Class_12.csv'))
    
    # Process categorical columns exactly like training
    df_encoded = df_train.copy()
    
    def get_clean_categorical_value_local(val, classes):
        for c in classes:
            if pd.isna(c) and (val == 'None' or val == 'nan' or val is None or val == '' or pd.isna(val)):
                return c
            if str(c) == str(val):
                return c
        return val

    for col in label_encoders.keys():
        le = label_encoders[col]
        df_encoded[col] = df_encoded[col].apply(lambda x: get_clean_categorical_value_local(x, le.classes_))
        df_encoded[col] = le.transform(df_encoded[col])
        
    # Scale numerical columns
    df_encoded[scaler.feature_names_in_] = scaler.transform(df_encoded[scaler.feature_names_in_])
    
    # Train features in matching order
    X_train = df_encoded[df_encoded.columns.drop('dropout')].values
    
    # Categorical features indices
    categorical_features_indices = [list(df_encoded.columns.drop('dropout')).index(col) for col in label_encoders.keys()]
    
    # Initialize LIME
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=list(df_encoded.columns.drop('dropout')),
        class_names=['Stayed', 'Dropout'],
        categorical_features=categorical_features_indices,
        categorical_names={list(df_encoded.columns.drop('dropout')).index(col): list(le.classes_) for col, le in label_encoders.items()},
        kernel_width=3,
        verbose=False,
        mode='classification'
    )
    print("LIME Explainer initialized successfully!")
except Exception as e:
    print(f"LIME Explainer initialization failed: {e}")
    explainer = None

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

def explain_prediction_lime(input_df):
    if explainer is None:
        return []
    
    try:
        # LIME expects a 1D numpy array representing the preprocessed features
        row = input_df.iloc[0].values
        
        # Generate local explanation using model's predict_proba
        exp = explainer.explain_instance(
            data_row=row,
            predict_fn=model.predict_proba,
            num_features=6
        )
        
        explanations = []
        for feature_exp, weight in exp.as_list():
            # Find which feature name this explanation represents
            feature_name = None
            for feat in FEATURE_ORDER:
                if feat in feature_exp:
                    feature_name = feat
                    break
                    
            if feature_name is None:
                continue
                
            display_name = feature_name.replace('_', ' ').replace('avg', '').title()
            
            # The contribution represents the weight impact on class 1 (Dropout) in percentage
            contribution = round(float(weight) * 100, 2)
            
            # Extract and unprocess actual value to display cleanly in frontend
            actual_val = input_df[feature_name].iloc[0]
            if feature_name in NUMERICAL_FEATURES:
                feat_idx = NUMERICAL_FEATURES.index(feature_name)
                # Unscale: (scaled_val * std) + mean
                unscaled_val = (actual_val * scaler.scale_[feat_idx]) + scaler.mean_[feat_idx]
                if feature_name == 'daily_study_hours':
                    actual_display = f"{unscaled_val:.1f} hrs"
                elif feature_name in ['jee_main_score', 'jee_advanced_score', 'mock_test_score_avg', 'class_12_percent']:
                    actual_display = f"{unscaled_val:.1f}%"
                else:
                    actual_display = str(int(round(unscaled_val)))
            else:
                le = label_encoders[feature_name]
                unencoded_val = le.inverse_transform([int(actual_val)])[0]
                # Map nan to 'None'
                actual_display = 'None' if pd.isna(unencoded_val) else str(unencoded_val)
                
            explanations.append({
                "feature": feature_name,
                "display_name": display_name,
                "actual_value": actual_display,
                "contribution": contribution
            })
            
        # Sort contributions by absolute impact
        explanations.sort(key=lambda x: abs(x["contribution"]), reverse=True)
        return explanations
        
    except Exception as e:
        print(f"LIME explanation generation failed: {e}")
        return []

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
        
        # Calculate XAI Local Feature Attributions using LIME
        xai_explanations = explain_prediction_lime(input_df)
        
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
