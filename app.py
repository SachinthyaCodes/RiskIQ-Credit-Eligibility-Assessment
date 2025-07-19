from flask import Flask, render_template, request, jsonify, flash, redirect, url_for, send_file
import pandas as pd
import numpy as np
import pickle
import os
from werkzeug.utils import secure_filename
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import warnings
from dotenv import load_dotenv
from ai_feedback_service import AIFeedbackService
from form_validator import FormValidator
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')

# Initialize services
validator = FormValidator()

# Initialize AI Feedback Service
ai_feedback_service = AIFeedbackService()

# Configuration
current_dir = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(current_dir, 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables for model components
risk_ensemble = None
preprocessor = None
optimal_threshold = 0.30  # More aggressive threshold to catch more risky applications

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load the trained models and preprocessor"""
    global risk_ensemble, preprocessor
    
    try:
        # Get the absolute path to the models directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        models_dir = os.path.join(current_dir, 'models')
        
        # Load saved models
        risk_ensemble = joblib.load(os.path.join(models_dir, 'risk_ensemble_model.pkl'))
        preprocessor = joblib.load(os.path.join(models_dir, 'preprocessor.pkl'))
        
        print("Models loaded successfully!")
        print(f"   Models directory: {models_dir}")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        print("Pre-trained models not found. Please train and save models first.")
        return False

def get_risk_level(probability):
    """Convert probability to risk level"""
    if probability >= 0.8:
        return "VERY HIGH", "", "danger"
    elif probability >= 0.6:
        return "HIGH", "", "warning"
    elif probability >= 0.4:
        return "MEDIUM", "", "info"
    elif probability >= 0.2:
        return "LOW", "", "success"
    else:
        return "VERY LOW", "", "success"

def get_recommendation(probability, threshold=0.30):
    """Get approval recommendation based on probability"""
    if probability >= 0.8:
        return {
            'decision': 'REJECTED',
            'reason': 'Very high default risk detected',
            'color': 'danger',
            'icon': ''
        }
    elif probability >= 0.6:
        return {
            'decision': 'REJECTED',
            'reason': 'High default risk detected',
            'color': 'danger',
            'icon': ''
        }
    elif probability >= threshold:
        return {
            'decision': 'MANUAL REVIEW',
            'reason': 'Moderate risk - requires additional documentation',
            'color': 'warning',
            'icon': ''
        }
    elif probability >= 0.15:
        return {
            'decision': 'APPROVED',
            'reason': 'Approved with enhanced monitoring',
            'color': 'info',
            'icon': ''
        }
    else:
        return {
            'decision': 'APPROVED',
            'reason': 'Standard approval with good terms',
            'color': 'success',
            'icon': ''
        }

def analyze_application_factors(input_data, risk_probability):
    """
    Analyze factors contributing to approval/rejection decision
    Provide customer-friendly explanations
    """
    
    # Calculate key metrics
    income = input_data['AMT_INCOME_TOTAL']
    age = input_data['AGE_YEARS']
    children = input_data['CNT_CHILDREN']
    employed_years = input_data['EMPLOYED_YEARS']
    family_members = input_data['CNT_FAM_MEMBERS']
    
    analysis = {
        'risk_factors': [],
        'positive_factors': [],
        'recommendations': [],
        'improvement_tips': []
    }
    
    # Risk factor analysis
    if income < 100000:
        analysis['risk_factors'].append({
            'factor': 'Low Income',
            'description': f'Your income of ${income:,.0f} is below our preferred threshold of $100,000',
            'impact': 'High',
            'icon': ''
        })
    
    if age < 25:
        analysis['risk_factors'].append({
            'factor': 'Young Age',
            'description': f'At {age} years old, you have limited credit history',
            'impact': 'Medium',
            'icon': ''
        })
    
    if employed_years <= 1:
        analysis['risk_factors'].append({
            'factor': 'Limited Employment History',
            'description': f'Only {employed_years} year(s) of employment shows limited stability',
            'impact': 'High',
            'icon': ''
        })
    
    if children >= 3:
        analysis['risk_factors'].append({
            'factor': 'High Financial Responsibility',
            'description': f'Supporting {children} children increases financial obligations',
            'impact': 'Medium',
            'icon': ''
        })
    
    if input_data['FLAG_OWN_REALTY'] == 'N':
        analysis['risk_factors'].append({
            'factor': 'No Property Ownership',
            'description': 'Not owning property reduces collateral security',
            'impact': 'Low',
            'icon': ''
        })
    
    if input_data['FLAG_OWN_CAR'] == 'N':
        analysis['risk_factors'].append({
            'factor': 'No Vehicle Ownership',
            'description': 'Not owning a vehicle may indicate limited assets',
            'impact': 'Low',
            'icon': ''
        })
    
    # Positive factors
    if income >= 200000:
        analysis['positive_factors'].append({
            'factor': 'High Income',
            'description': f'Your income of ${income:,.0f} shows strong earning capacity',
            'impact': 'High',
            'icon': ''
        })
    elif income >= 100000:
        analysis['positive_factors'].append({
            'factor': 'Good Income',
            'description': f'Your income of ${income:,.0f} meets our requirements',
            'impact': 'Medium',
            'icon': ''
        })
    
    if age >= 35:
        analysis['positive_factors'].append({
            'factor': 'Mature Age',
            'description': f'At {age} years old, you have established financial maturity',
            'impact': 'Medium',
            'icon': ''
        })
    
    if employed_years >= 5:
        analysis['positive_factors'].append({
            'factor': 'Stable Employment',
            'description': f'{employed_years} years of employment shows excellent stability',
            'impact': 'High',
            'icon': ''
        })
    elif employed_years >= 2:
        analysis['positive_factors'].append({
            'factor': 'Good Employment History',
            'description': f'{employed_years} years of employment shows stability',
            'impact': 'Medium',
            'icon': ''
        })
    
    if input_data['FLAG_OWN_REALTY'] == 'Y':
        analysis['positive_factors'].append({
            'factor': 'Property Owner',
            'description': 'Property ownership provides financial security',
            'impact': 'Medium',
            'icon': ''
        })
    
    if input_data['NAME_EDUCATION_TYPE'] == 'Higher education':
        analysis['positive_factors'].append({
            'factor': 'Higher Education',
            'description': 'Higher education correlates with financial responsibility',
            'impact': 'Medium',
            'icon': ''
        })
    
    if input_data['NAME_FAMILY_STATUS'] == 'Married':
        analysis['positive_factors'].append({
            'factor': 'Married',
            'description': 'Married status often indicates financial stability',
            'impact': 'Low',
            'icon': ''
        })
    
    # Generate recommendations based on risk factors
    if risk_probability > 0.6:
        analysis['recommendations'] = [
            "Unfortunately, your application has been rejected due to high risk factors.",
            "However, we encourage you to improve your financial profile and reapply in the future."
        ]
        
        # Improvement tips for rejected applications
        if income < 100000:
            analysis['improvement_tips'].append("Increase your income through career advancement or additional income sources")
        
        if employed_years <= 1:
            analysis['improvement_tips'].append("Build a longer employment history (minimum 2-3 years recommended)")
        
        if age < 25:
            analysis['improvement_tips'].append("Consider having a co-signer with established credit history")
        
        if input_data['FLAG_OWN_REALTY'] == 'N':
            analysis['improvement_tips'].append("Consider building assets like property ownership")
            
        analysis['improvement_tips'].append("You can reapply after 6 months with improved financial standing")
    
    elif risk_probability > 0.3:
        analysis['recommendations'] = [
            "Your application requires manual review.",
            "We need additional documentation to process your application."
        ]
        analysis['improvement_tips'] = [
            "Please provide additional income verification",
            "Bank statements for the last 3 months",
            "Employment verification letter",
            "Any additional asset documentation"
        ]
    
    elif risk_probability > 0.15:
        analysis['recommendations'] = [
            "Congratulations! Your application has been approved.",
            "You will receive enhanced monitoring for the first 6 months."
        ]
        analysis['improvement_tips'] = [
            "Maintain regular payments to build excellent credit history",
            "Consider setting up automatic payments",
            "Monitor your credit utilization ratio"
        ]
    
    else:
        analysis['recommendations'] = [
            "Congratulations! Your application has been approved with excellent terms.",
            "You qualify for our premium credit card benefits."
        ]
        analysis['improvement_tips'] = [
            "You qualify for our lowest interest rates",
            "Premium rewards and cashback programs available",
            "Higher credit limits available upon request"
        ]
    
    return analysis

def add_engineered_features(data_dict):
    """Add engineered features to match the trained model"""
    # Create a copy to avoid modifying original
    enhanced_data = data_dict.copy()
    
    # Calculate engineered features
    enhanced_data['INCOME_PER_PERSON'] = enhanced_data['AMT_INCOME_TOTAL'] / enhanced_data['CNT_FAM_MEMBERS']
    enhanced_data['INCOME_PER_CHILD'] = enhanced_data['AMT_INCOME_TOTAL'] / (enhanced_data['CNT_CHILDREN'] + 1)
    enhanced_data['EMPLOYMENT_RATIO'] = enhanced_data['EMPLOYED_YEARS'] / (enhanced_data['AGE_YEARS'] + 1)
    
    # Risk categories
    enhanced_data['HIGH_CHILDREN'] = 1 if enhanced_data['CNT_CHILDREN'] >= 3 else 0
    enhanced_data['LOW_INCOME'] = 1 if enhanced_data['AMT_INCOME_TOTAL'] < 100000 else 0
    enhanced_data['YOUNG_AGE'] = 1 if enhanced_data['AGE_YEARS'] < 25 else 0
    enhanced_data['NO_EMPLOYMENT'] = 1 if enhanced_data['EMPLOYED_YEARS'] <= 1 else 0
    
    return enhanced_data

def predict_single(data_dict):
    """Predict for a single application"""
    global risk_ensemble, preprocessor
    
    if risk_ensemble is None or preprocessor is None:
        return None, "Models not loaded"
    
    try:
        # Add engineered features
        enhanced_data = add_engineered_features(data_dict)
        
        # Convert to DataFrame
        df = pd.DataFrame([enhanced_data])
        
        # Transform the data
        X_transformed = preprocessor.transform(df)
        
        # Get prediction probabilities
        probabilities = risk_ensemble.predict_proba(X_transformed)[0]
        probability = probabilities[1]  # Class 1 (bad/risky)
        
        # Debug information
        print(f"Debug - Raw probabilities: Class 0 (Good): {probabilities[0]:.3f}, Class 1 (Bad): {probabilities[1]:.3f}")
        print(f"Debug - Using probability for Class 1 (Bad): {probability:.3f}")
        print(f"Debug - Threshold: {optimal_threshold}")
        
        # Get risk level and recommendation
        risk_level, risk_icon, risk_color = get_risk_level(probability)
        recommendation = get_recommendation(probability, optimal_threshold)
        
        # Get detailed analysis for customer explanation
        detailed_analysis = analyze_application_factors(data_dict, probability)
        
        print(f"Debug - Risk level: {risk_level}")
        print(f"Debug - Recommendation: {recommendation['decision']}")
        
        result = {
            'risk_probability': round(probability * 100, 1),
            'risk_level': risk_level,
            'risk_icon': risk_icon,
            'risk_color': risk_color,
            'recommendation': recommendation,
            'confidence': round((1 - abs(probability - 0.5) * 2) * 100, 1),
            'detailed_analysis': detailed_analysis,
            'debug_info': {
                'class_0_prob': round(probabilities[0] * 100, 1),
                'class_1_prob': round(probabilities[1] * 100, 1),
                'threshold_used': optimal_threshold,
                'engineered_features': {
                    'income_per_person': round(enhanced_data['INCOME_PER_PERSON'], 2),
                    'income_per_child': round(enhanced_data['INCOME_PER_CHILD'], 2),
                    'employment_ratio': round(enhanced_data['EMPLOYMENT_RATIO'], 3),
                    'high_children': enhanced_data['HIGH_CHILDREN'],
                    'low_income': enhanced_data['LOW_INCOME'],
                    'young_age': enhanced_data['YOUNG_AGE'],
                    'no_employment': enhanced_data['NO_EMPLOYMENT']
                }
            }
        }
        
        return result, None
        
    except Exception as e:
        return None, f"Prediction error: {str(e)}"

def predict_batch(csv_file):
    """Predict for batch CSV file"""
    global risk_ensemble, preprocessor
    
    if risk_ensemble is None or preprocessor is None:
        return None, "Models not loaded"
    
    try:
        # Read CSV
        df = pd.read_csv(csv_file)
        
        # Check required columns
        required_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 
                        'CNT_CHILDREN', 'AMT_INCOME_TOTAL', 'NAME_INCOME_TYPE',
                        'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
                        'CNT_FAM_MEMBERS', 'OCCUPATION_TYPE', 'AGE_YEARS', 'EMPLOYED_YEARS']
        
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return None, f"Missing columns: {missing_cols}"
        
        # Add engineered features to the dataframe
        df['INCOME_PER_PERSON'] = df['AMT_INCOME_TOTAL'] / df['CNT_FAM_MEMBERS']
        df['INCOME_PER_CHILD'] = df['AMT_INCOME_TOTAL'] / (df['CNT_CHILDREN'] + 1)
        df['EMPLOYMENT_RATIO'] = df['EMPLOYED_YEARS'] / (df['AGE_YEARS'] + 1)
        
        # Risk categories
        df['HIGH_CHILDREN'] = (df['CNT_CHILDREN'] >= 3).astype(int)
        df['LOW_INCOME'] = (df['AMT_INCOME_TOTAL'] < 100000).astype(int)
        df['YOUNG_AGE'] = (df['AGE_YEARS'] < 25).astype(int)
        df['NO_EMPLOYMENT'] = (df['EMPLOYED_YEARS'] <= 1).astype(int)
        
        # All features for the model
        all_features = required_cols + ['INCOME_PER_PERSON', 'INCOME_PER_CHILD', 'EMPLOYMENT_RATIO',
                                       'HIGH_CHILDREN', 'LOW_INCOME', 'YOUNG_AGE', 'NO_EMPLOYMENT']
        
        # Transform data
        X_transformed = preprocessor.transform(df[all_features])
        
        # Get predictions
        probabilities = risk_ensemble.predict_proba(X_transformed)[:, 1]
        
        # Add results to dataframe
        df['Risk_Probability'] = (probabilities * 100).round(1)
        df['Risk_Level'] = [get_risk_level(p)[0] for p in probabilities]
        df['Decision'] = [get_recommendation(p, optimal_threshold)['decision'] for p in probabilities]
        df['Recommendation'] = [get_recommendation(p, optimal_threshold)['reason'] for p in probabilities]
        
        return df, None
        
    except Exception as e:
        return None, f"Batch prediction error: {str(e)}"

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/single_prediction', methods=['GET'])
def single_prediction():
    """Single prediction form page"""
    return render_template('single_prediction_new.html')

@app.route('/batch_prediction', methods=['GET'])
def batch_prediction():
    """Batch prediction page"""
    return render_template('batch_prediction.html')

@app.route('/predict_single', methods=['POST'])
def predict_single_route():
    """Handle single prediction"""
    try:
        # Server-side validation first
        form_data = {
            'gender': request.form.get('gender'),
            'own_car': request.form.get('own_car'),
            'own_realty': request.form.get('own_realty'),
            'children': request.form.get('children', 0),
            'income': request.form.get('income', 0),
            'income_type': request.form.get('income_type'),
            'education_type': request.form.get('education_type'),
            'family_status': request.form.get('family_status'),
            'housing_type': request.form.get('housing_type'),
            'family_members': request.form.get('family_members', 1),
            'occupation_type': request.form.get('occupation_type'),
            'age': request.form.get('age', 18),
            'employment_days': request.form.get('employment_days', 0)
        }
        
        # Validate form data
        is_valid, errors, warnings = validator.validate_credit_application(form_data)
        
        if not is_valid:
            for error in errors:
                flash(error, 'danger')
            return redirect(url_for('single_prediction'))
        
        # Show warnings if any
        for warning in warnings:
            flash(warning, 'warning')
        
        # Convert to model format
        data = {
            'CODE_GENDER': form_data['gender'],
            'FLAG_OWN_CAR': form_data['own_car'],
            'FLAG_OWN_REALTY': form_data['own_realty'],
            'CNT_CHILDREN': int(form_data['children']),
            'AMT_INCOME_TOTAL': float(form_data['income']),
            'NAME_INCOME_TYPE': form_data['income_type'],
            'NAME_EDUCATION_TYPE': form_data['education_type'],
            'NAME_FAMILY_STATUS': form_data['family_status'],
            'NAME_HOUSING_TYPE': form_data['housing_type'],
            'CNT_FAM_MEMBERS': int(form_data['family_members']),
            'OCCUPATION_TYPE': form_data['occupation_type'],
            'AGE_YEARS': int(form_data['age']),
            'EMPLOYED_YEARS': int(form_data['employment_days'])
        }
        
        # Debug: Print received data
        print("Debug - Validated form data:")
        for key, value in data.items():
            print(f"  {key}: {value}")
        
        # Make prediction
        result, error = predict_single(data)
        
        if error:
            flash(f'Error: {error}', 'danger')
            return redirect(url_for('single_prediction'))
        
        # Generate AI-powered feedback
        try:
            ai_feedback = ai_feedback_service.generate_detailed_feedback(data, result)
            result['ai_feedback'] = ai_feedback
        except Exception as e:
            print(f"AI Feedback Error: {str(e)}")
            result['ai_feedback'] = None
        
        return render_template('result_single_clean.html', result=result, input_data=data)
        
    except Exception as e:
        print(f"Error processing form: {str(e)}")
        flash(f'Error processing form: {str(e)}', 'danger')
        return redirect(url_for('single_prediction'))

@app.route('/predict_batch', methods=['POST'])
def predict_batch_route():
    """Handle batch prediction"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected', 'danger')
            return redirect(url_for('batch_prediction'))
        
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected', 'danger')
            return redirect(url_for('batch_prediction'))
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Make predictions
            results_df, error = predict_batch(filepath)
            
            # Clean up uploaded file
            os.remove(filepath)
            
            if error:
                flash(f'Error: {error}', 'danger')
                return redirect(url_for('batch_prediction'))
            
            # Convert to records for template
            results = results_df.to_dict('records')
            
            return render_template('result_batch.html', results=results, 
                                 total_count=len(results),
                                 approved_count=len([r for r in results if r['Decision'] == 'APPROVED']),
                                 rejected_count=len([r for r in results if r['Decision'] == 'REJECTED']),
                                 review_count=len([r for r in results if r['Decision'] == 'MANUAL REVIEW']))
        
        else:
            flash('Invalid file type. Please upload a CSV file.', 'danger')
            return redirect(url_for('batch_prediction'))
            
    except Exception as e:
        flash(f'Error processing file: {str(e)}', 'danger')
        return redirect(url_for('batch_prediction'))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        result, error = predict_single(data)
        
        if error:
            return jsonify({'error': error}), 500
        
        return jsonify({'success': True, 'result': result})
        
    except Exception as e:
        return jsonify({'error': f'API error: {str(e)}'}), 500

@app.route('/download_template')
def download_template():
    """Download CSV template for batch prediction"""
    template_data = {
        'CODE_GENDER': ['M', 'F'],
        'FLAG_OWN_CAR': ['Y', 'N'],
        'FLAG_OWN_REALTY': ['Y', 'N'],
        'CNT_CHILDREN': [0, 2],
        'AMT_INCOME_TOTAL': [50000, 75000],
        'NAME_INCOME_TYPE': ['Working', 'Commercial associate'],
        'NAME_EDUCATION_TYPE': ['Higher education', 'Secondary / secondary special'],
        'NAME_FAMILY_STATUS': ['Married', 'Single / not married'],
        'NAME_HOUSING_TYPE': ['House / apartment', 'With parents'],
        'CNT_FAM_MEMBERS': [2, 4],
        'OCCUPATION_TYPE': ['Laborers', 'Core staff'],
        'AGE_YEARS': [30, 45],
        'EMPLOYED_YEARS': [5, 10]
    }
    
    template_df = pd.DataFrame(template_data)
    
    # Save to uploads folder temporarily with absolute path
    current_dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(current_dir, app.config['UPLOAD_FOLDER'], 'credit_application_template.csv')
    template_df.to_csv(template_path, index=False)
    
    return send_file(template_path, as_attachment=True, download_name='credit_application_template.csv')

@app.route('/api/validate_form', methods=['POST'])
def validate_form_api():
    """API endpoint for real-time form validation"""
    try:
        form_data = request.get_json()
        
        if not form_data:
            return jsonify({'error': 'No form data provided'}), 400
        
        # Validate the form data
        is_valid, errors, warnings = validator.validate_credit_application(form_data)
        
        return jsonify({
            'is_valid': is_valid,
            'errors': errors,
            'warnings': warnings,
            'summary': validator.format_validation_summary()
        })
        
    except Exception as e:
        return jsonify({'error': f'Validation API error: {str(e)}'}), 500

@app.route('/api/ai_feedback', methods=['POST'])
def get_ai_feedback():
    """API endpoint for getting detailed AI feedback"""
    try:
        data = request.get_json()
        applicant_data = data.get('applicant_data', {})
        prediction_result = data.get('prediction_result', {})
        
        if not applicant_data or not prediction_result:
            return jsonify({'error': 'Missing applicant_data or prediction_result'}), 400
        
        # Generate AI feedback
        ai_feedback = ai_feedback_service.generate_detailed_feedback(applicant_data, prediction_result)
        
        return jsonify({
            'success': True,
            'ai_feedback': ai_feedback
        })
        
    except Exception as e:
        return jsonify({'error': f'AI Feedback API error: {str(e)}'}), 500

@app.route('/feedback/<int:application_id>')
def detailed_feedback_page(application_id):
    """Dedicated page for detailed AI feedback"""
    # This could be used for accessing saved feedback later
    return render_template('detailed_feedback.html', application_id=application_id)

if __name__ == '__main__':
    print("Starting Credit Approval Web Application...")
    
    # Try to load models
    models_loaded = load_models()
    
    if not models_loaded:
        print("Warning: Models not loaded. Please train models first.")
        print("Tip: Run the notebook cells to train and save models.")
    
    # Production configuration
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    print(f"Web app will be available on port: {port}")
    app.run(debug=debug, host='0.0.0.0', port=port)
