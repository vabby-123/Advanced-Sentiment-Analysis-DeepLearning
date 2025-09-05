# Flask Web Application for Reddit Sentiment Analysis Deployment
# app.py - Updated for local deployment

from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import re
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# ============================================================================
# LOAD MODELS AND PREPROCESSORS
# ============================================================================

print("Loading models and preprocessors...")

try:
    # Define model paths
    models_dir = 'models'
    
    # Load traditional ML models and preprocessors
    tfidf_vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
    label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
    
    # Load individual models
    nb_model = joblib.load(os.path.join(models_dir, 'naive_bayes_model.pkl'))
    lr_model = joblib.load(os.path.join(models_dir, 'logistic_regression_model.pkl'))
    rf_model = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))
    
    # Load deep learning model and tokenizer
    lstm_model = load_model(os.path.join(models_dir, 'lstm_model.h5'))
    tokenizer = joblib.load(os.path.join(models_dir, 'tokenizer.pkl'))
    
    print("✅ All models loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    print("Make sure all model files are in the 'models/' directory")

# ============================================================================
# PREPROCESSING FUNCTION
# ============================================================================

def preprocess_text(text):
    """Text preprocessing function"""
    if pd.isna(text) or text == "":
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_traditional_model(text, model, model_name):
    """Predict using traditional ML models"""
    try:
        processed = preprocess_text(text)
        text_tfidf = tfidf_vectorizer.transform([processed])
        
        prediction = model.predict(text_tfidf)[0]
        probabilities = model.predict_proba(text_tfidf)[0]
        category = label_encoder.inverse_transform([prediction])[0]
        confidence = max(probabilities)
        
        # Create probability distribution
        prob_dict = {}
        for i, class_name in enumerate(label_encoder.classes_):
            prob_dict[class_name] = float(probabilities[i])
        
        return {
            'model': model_name,
            'category': category,
            'confidence': float(confidence),
            'probabilities': prob_dict,
            'success': True
        }
    
    except Exception as e:
        return {
            'model': model_name,
            'error': str(e),
            'success': False
        }

def predict_lstm_model(text):
    """Predict using LSTM model"""
    try:
        processed = preprocess_text(text)
        
        # Convert to sequence
        sequence = tokenizer.texts_to_sequences([processed])
        
        # Pad sequence
        max_len = 100  # Same as training
        padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
        
        # Predict
        prediction_prob = lstm_model.predict(padded, verbose=0)[0]
        prediction = np.argmax(prediction_prob)
        category = label_encoder.inverse_transform([prediction])[0]
        confidence = float(max(prediction_prob))
        
        # Create probability distribution
        prob_dict = {}
        for i, class_name in enumerate(label_encoder.classes_):
            prob_dict[class_name] = float(prediction_prob[i])
        
        return {
            'model': 'LSTM',
            'category': category,
            'confidence': confidence,
            'probabilities': prob_dict,
            'success': True
        }
    
    except Exception as e:
        return {
            'model': 'LSTM',
            'error': str(e),
            'success': False
        }

def get_all_predictions(text):
    """Get predictions from all models"""
    models = {
        'Naive Bayes': nb_model,
        'Logistic Regression': lr_model,
        'Random Forest': rf_model
    }
    
    results = {}
    
    # Get traditional model predictions
    for model_name, model in models.items():
        results[model_name] = predict_traditional_model(text, model, model_name)
    
    # Get LSTM prediction
    results['LSTM'] = predict_lstm_model(text)
    
    return results

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def home():
    """Home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        if len(text) > 1000:  # Limit text length
            return jsonify({'error': 'Text too long. Maximum 1000 characters allowed.'}), 400
        
        # Get model selection (default to all)
        selected_models = data.get('models', ['all'])
        
        if 'all' in selected_models:
            # Get predictions from all models
            results = get_all_predictions(text)
        else:
            # Get predictions from selected models only
            results = {}
            models = {
                'Naive Bayes': nb_model,
                'Logistic Regression': lr_model,
                'Random Forest': rf_model
            }
            
            for model_name in selected_models:
                if model_name in models:
                    results[model_name] = predict_traditional_model(text, models[model_name], model_name)
                elif model_name == 'LSTM':
                    results['LSTM'] = predict_lstm_model(text)
        
        return jsonify({
            'text': text,
            'predictions': results,
            'available_categories': list(label_encoder.classes_)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/models')
def get_models():
    """Get available models"""
    return jsonify({
        'traditional_models': ['Naive Bayes', 'Logistic Regression', 'Random Forest'],
        'deep_learning_models': ['LSTM'],
        'all_models': ['Naive Bayes', 'Logistic Regression', 'Random Forest', 'LSTM'],
        'categories': list(label_encoder.classes_)
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    try:
        # Test a simple prediction to ensure all models work
        test_text = "test"
        test_results = get_all_predictions(test_text)
        
        # Check if all models returned successfully
        all_success = all(result.get('success', False) for result in test_results.values())
        
        if all_success:
            return jsonify({
                'status': 'healthy', 
                'message': 'All models loaded and working correctly',
                'models_loaded': list(test_results.keys())
            })
        else:
            return jsonify({
                'status': 'degraded',
                'message': 'Some models have issues',
                'models_status': {model: result.get('success', False) for model, result in test_results.items()}
            }), 500
            
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'message': f'Health check failed: {str(e)}'
        }), 500

@app.route('/demo')
def demo():
    """Demo page with sample predictions"""
    sample_texts = [
        "This product is absolutely amazing! Best purchase ever!",
        "I'm really disappointed with the service.",
        "This is my car",
        "The weather is nice today",
        "I hate waiting in long queues"
    ]
    
    demo_results = []
    for text in sample_texts:
        try:
            predictions = get_all_predictions(text)
            demo_results.append({
                'text': text,
                'predictions': predictions
            })
        except Exception as e:
            demo_results.append({
                'text': text,
                'error': str(e)
            })
    
    return jsonify({'demo_results': demo_results})

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*50)
    print("🚀 REDDIT SENTIMENT ANALYSIS API")
    print("="*50)
    print("Available endpoints:")
    print("  - GET  /           : Web interface")
    print("  - POST /predict    : Get sentiment predictions")
    print("  - GET  /models     : Get available models info")
    print("  - GET  /demo       : Demo predictions")
    print("  - GET  /health     : Health check")
    print("="*50)
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("="*50)
    
    app.run(debug=True, host='0.0.0.0', port=5000)