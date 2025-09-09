# # Flask Web Application for Reddit Sentiment Analysis Deployment
# # app.py - Updated for local deployment

# from flask import Flask, render_template, request, jsonify
# import joblib
# import numpy as np
# import pandas as pd
# import re
# import os
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# import warnings
# warnings.filterwarnings('ignore')

# app = Flask(__name__)

# # ============================================================================
# # LOAD MODELS AND PREPROCESSORS
# # ============================================================================

# print("Loading models and preprocessors...")

# try:
#     # Define model paths
#     models_dir = 'models'
    
#     # Load traditional ML models and preprocessors
#     tfidf_vectorizer = joblib.load(os.path.join(models_dir, 'tfidf_vectorizer.pkl'))
#     label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))
    
#     # Load individual models
#     nb_model = joblib.load(os.path.join(models_dir, 'naive_bayes_model.pkl'))
#     lr_model = joblib.load(os.path.join(models_dir, 'logistic_regression_model.pkl'))
#     rf_model = joblib.load(os.path.join(models_dir, 'random_forest_model.pkl'))
    
#     # Load deep learning model and tokenizer
#     lstm_model = load_model(os.path.join(models_dir, 'lstm_model.h5'))
#     tokenizer = joblib.load(os.path.join(models_dir, 'tokenizer.pkl'))
    
#     print("✅ All models loaded successfully!")
    
# except Exception as e:
#     print(f"❌ Error loading models: {e}")
#     print("Make sure all model files are in the 'models/' directory")

# # ============================================================================
# # PREPROCESSING FUNCTION
# # ============================================================================

# def preprocess_text(text):
#     """Text preprocessing function"""
#     if pd.isna(text) or text == "":
#         return ""
    
#     # Convert to lowercase
#     text = text.lower()
    
#     # Remove URLs
#     text = re.sub(r'http\S+|www.\S+', '', text)
    
#     # Remove mentions and hashtags
#     text = re.sub(r'@\w+|#\w+', '', text)
    
#     # Remove extra whitespace
#     text = ' '.join(text.split())
    
#     return text

# # ============================================================================
# # PREDICTION FUNCTIONS
# # ============================================================================

# def predict_traditional_model(text, model, model_name):
#     """Predict using traditional ML models"""
#     try:
#         processed = preprocess_text(text)
#         text_tfidf = tfidf_vectorizer.transform([processed])
        
#         prediction = model.predict(text_tfidf)[0]
#         probabilities = model.predict_proba(text_tfidf)[0]
#         category = label_encoder.inverse_transform([prediction])[0]
#         confidence = max(probabilities)
        
#         # Create probability distribution
#         prob_dict = {}
#         for i, class_name in enumerate(label_encoder.classes_):
#             prob_dict[class_name] = float(probabilities[i])
        
#         return {
#             'model': model_name,
#             'category': category,
#             'confidence': float(confidence),
#             'probabilities': prob_dict,
#             'success': True
#         }
    
#     except Exception as e:
#         return {
#             'model': model_name,
#             'error': str(e),
#             'success': False
#         }

# def predict_lstm_model(text):
#     """Predict using LSTM model"""
#     try:
#         processed = preprocess_text(text)
        
#         # Convert to sequence
#         sequence = tokenizer.texts_to_sequences([processed])
        
#         # Pad sequence
#         max_len = 100  # Same as training
#         padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
        
#         # Predict
#         prediction_prob = lstm_model.predict(padded, verbose=0)[0]
#         prediction = np.argmax(prediction_prob)
#         category = label_encoder.inverse_transform([prediction])[0]
#         confidence = float(max(prediction_prob))
        
#         # Create probability distribution
#         prob_dict = {}
#         for i, class_name in enumerate(label_encoder.classes_):
#             prob_dict[class_name] = float(prediction_prob[i])
        
#         return {
#             'model': 'LSTM',
#             'category': category,
#             'confidence': confidence,
#             'probabilities': prob_dict,
#             'success': True
#         }
    
#     except Exception as e:
#         return {
#             'model': 'LSTM',
#             'error': str(e),
#             'success': False
#         }

# def get_all_predictions(text):
#     """Get predictions from all models"""
#     models = {
#         'Naive Bayes': nb_model,
#         'Logistic Regression': lr_model,
#         'Random Forest': rf_model
#     }
    
#     results = {}
    
#     # Get traditional model predictions
#     for model_name, model in models.items():
#         results[model_name] = predict_traditional_model(text, model, model_name)
    
#     # Get LSTM prediction
#     results['LSTM'] = predict_lstm_model(text)
    
#     return results

# # ============================================================================
# # FLASK ROUTES
# # ============================================================================

# @app.route('/')
# def home():
#     """Home page"""
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     """API endpoint for predictions"""
#     try:
#         data = request.get_json()
        
#         if not data or 'text' not in data:
#             return jsonify({'error': 'No text provided'}), 400
        
#         text = data['text'].strip()
        
#         if not text:
#             return jsonify({'error': 'Empty text provided'}), 400
        
#         if len(text) > 1000:  # Limit text length
#             return jsonify({'error': 'Text too long. Maximum 1000 characters allowed.'}), 400
        
#         # Get model selection (default to all)
#         selected_models = data.get('models', ['all'])
        
#         if 'all' in selected_models:
#             # Get predictions from all models
#             results = get_all_predictions(text)
#         else:
#             # Get predictions from selected models only
#             results = {}
#             models = {
#                 'Naive Bayes': nb_model,
#                 'Logistic Regression': lr_model,
#                 'Random Forest': rf_model
#             }
            
#             for model_name in selected_models:
#                 if model_name in models:
#                     results[model_name] = predict_traditional_model(text, models[model_name], model_name)
#                 elif model_name == 'LSTM':
#                     results['LSTM'] = predict_lstm_model(text)
        
#         return jsonify({
#             'text': text,
#             'predictions': results,
#             'available_categories': list(label_encoder.classes_)
#         })
    
#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/models')
# def get_models():
#     """Get available models"""
#     return jsonify({
#         'traditional_models': ['Naive Bayes', 'Logistic Regression', 'Random Forest'],
#         'deep_learning_models': ['LSTM'],
#         'all_models': ['Naive Bayes', 'Logistic Regression', 'Random Forest', 'LSTM'],
#         'categories': list(label_encoder.classes_)
#     })

# @app.route('/health')
# def health_check():
#     """Health check endpoint"""
#     try:
#         # Test a simple prediction to ensure all models work
#         test_text = "test"
#         test_results = get_all_predictions(test_text)
        
#         # Check if all models returned successfully
#         all_success = all(result.get('success', False) for result in test_results.values())
        
#         if all_success:
#             return jsonify({
#                 'status': 'healthy', 
#                 'message': 'All models loaded and working correctly',
#                 'models_loaded': list(test_results.keys())
#             })
#         else:
#             return jsonify({
#                 'status': 'degraded',
#                 'message': 'Some models have issues',
#                 'models_status': {model: result.get('success', False) for model, result in test_results.items()}
#             }), 500
            
#     except Exception as e:
#         return jsonify({
#             'status': 'unhealthy',
#             'message': f'Health check failed: {str(e)}'
#         }), 500

# @app.route('/demo')
# def demo():
#     """Demo page with sample predictions"""
#     sample_texts = [
#         "This product is absolutely amazing! Best purchase ever!",
#         "I'm really disappointed with the service.",
#         "This is my car",
#         "The weather is nice today",
#         "I hate waiting in long queues"
#     ]
    
#     demo_results = []
#     for text in sample_texts:
#         try:
#             predictions = get_all_predictions(text)
#             demo_results.append({
#                 'text': text,
#                 'predictions': predictions
#             })
#         except Exception as e:
#             demo_results.append({
#                 'text': text,
#                 'error': str(e)
#             })
    
#     return jsonify({'demo_results': demo_results})

# # ============================================================================
# # ERROR HANDLERS
# # ============================================================================

# @app.errorhandler(404)
# def not_found(error):
#     return jsonify({'error': 'Endpoint not found'}), 404

# @app.errorhandler(500)
# def internal_error(error):
#     return jsonify({'error': 'Internal server error'}), 500

# # ============================================================================
# # MAIN EXECUTION
# # ============================================================================

# if __name__ == '__main__':
#     print("\n" + "="*50)
#     print("🚀 REDDIT SENTIMENT ANALYSIS API")
#     print("="*50)
#     print("Available endpoints:")
#     print("  - GET  /           : Web interface")
#     print("  - POST /predict    : Get sentiment predictions")
#     print("  - GET  /models     : Get available models info")
#     print("  - GET  /demo       : Demo predictions")
#     print("  - GET  /health     : Health check")
#     print("="*50)
#     print("🌐 Open your browser and go to: http://localhost:5000")
#     print("="*50)
    
#     port = int(os.environ.get("PORT", 5000))
#     app.run(host="0.0.0.0", port=port, debug=False)
    
# streamlit_sentiment_app.py - Enhanced Reddit Sentiment Analysis with Streamlit

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import re
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import warnings
import datetime
import json
from collections import Counter
import time
from io import BytesIO
import base64

warnings.filterwarnings('ignore')


st.set_page_config(
    page_title="🎭 Advanced Sentiment Analyzer",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .sentiment-positive {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 10px;
        margin: 5px 0;
    }
    .sentiment-negative {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 10px;
        margin: 5px 0;
    }
    .sentiment-neutral {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 10px;
        margin: 5px 0;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# INITIALIZE SESSION STATE
# ============================================================================

if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []

if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False

if 'batch_results' not in st.session_state:
    st.session_state.batch_results = []

# ============================================================================
# MODEL LOADING FUNCTIONS
# ============================================================================

@st.cache_resource
def load_models():
    """Load all models and preprocessors with caching"""
    try:
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
        
        return {
            'tfidf_vectorizer': tfidf_vectorizer,
            'label_encoder': label_encoder,
            'nb_model': nb_model,
            'lr_model': lr_model,
            'rf_model': rf_model,
            'lstm_model': lstm_model,
            'tokenizer': tokenizer
        }
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None

# Load models
with st.spinner("🔄 Loading AI models..."):
    models = load_models()
    if models:
        st.session_state.models_loaded = True
        st.success("✅ All models loaded successfully!")
    else:
        st.error("❌ Failed to load models. Please check the models directory.")
        st.stop()

# ============================================================================
# PREPROCESSING FUNCTIONS
# ============================================================================

def preprocess_text(text, advanced_cleaning=False):
    """Enhanced text preprocessing function"""
    if pd.isna(text) or text == "":
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Remove mentions and hashtags
    text = re.sub(r'@\w+|#\w+', '', text)
    
    if advanced_cleaning:
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove special characters except basic punctuation
        text = re.sub(r'[^a-zA-Z\s.!?]', '', text)
        
        # Remove extra punctuation
        text = re.sub(r'[.!?]{2,}', '.', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text

def get_text_statistics(text):
    """Get comprehensive text statistics"""
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    
    return {
        'character_count': len(text),
        'word_count': len(words),
        'sentence_count': len([s for s in sentences if s.strip()]),
        'avg_word_length': np.mean([len(word) for word in words]) if words else 0,
        'unique_words': len(set(words)),
        'lexical_diversity': len(set(words)) / len(words) if words else 0
    }

# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_traditional_model(text, model, model_name, confidence_threshold=0.5):
    """Enhanced prediction function for traditional ML models"""
    try:
        processed = preprocess_text(text, advanced_cleaning=st.session_state.get('advanced_cleaning', False))
        text_tfidf = models['tfidf_vectorizer'].transform([processed])
        
        prediction = model.predict(text_tfidf)[0]
        probabilities = model.predict_proba(text_tfidf)[0]
        category = models['label_encoder'].inverse_transform([prediction])[0]
        confidence = max(probabilities)
        
        # Apply confidence threshold
        if confidence < confidence_threshold:
            category = "Uncertain"
        
        # Create probability distribution
        prob_dict = {}
        for i, class_name in enumerate(models['label_encoder'].classes_):
            prob_dict[class_name] = float(probabilities[i])
        
        return {
            'model': model_name,
            'category': category,
            'confidence': float(confidence),
            'probabilities': prob_dict,
            'success': True,
            'timestamp': datetime.datetime.now()
        }
    
    except Exception as e:
        return {
            'model': model_name,
            'error': str(e),
            'success': False
        }

def predict_lstm_model(text, confidence_threshold=0.5):
    """Enhanced LSTM prediction function"""
    try:
        processed = preprocess_text(text, advanced_cleaning=st.session_state.get('advanced_cleaning', False))
        
        # Convert to sequence
        sequence = models['tokenizer'].texts_to_sequences([processed])
        
        # Pad sequence
        max_len = 100
        padded = pad_sequences(sequence, maxlen=max_len, padding='post', truncating='post')
        
        # Predict
        prediction_prob = models['lstm_model'].predict(padded, verbose=0)[0]
        prediction = np.argmax(prediction_prob)
        category = models['label_encoder'].inverse_transform([prediction])[0]
        confidence = float(max(prediction_prob))
        
        # Apply confidence threshold
        if confidence < confidence_threshold:
            category = "Uncertain"
        
        # Create probability distribution
        prob_dict = {}
        for i, class_name in enumerate(models['label_encoder'].classes_):
            prob_dict[class_name] = float(prediction_prob[i])
        
        return {
            'model': 'LSTM',
            'category': category,
            'confidence': confidence,
            'probabilities': prob_dict,
            'success': True,
            'timestamp': datetime.datetime.now()
        }
    
    except Exception as e:
        return {
            'model': 'LSTM',
            'error': str(e),
            'success': False
        }

def get_ensemble_prediction(predictions):
    """Get ensemble prediction from all models"""
    successful_predictions = [p for p in predictions.values() if p.get('success', False)]
    
    if not successful_predictions:
        return None
    
    # Weighted voting (LSTM gets higher weight)
    weights = {'LSTM': 0.4, 'Logistic Regression': 0.25, 'Random Forest': 0.2, 'Naive Bayes': 0.15}
    
    categories = list(models['label_encoder'].classes_)
    ensemble_probs = {cat: 0.0 for cat in categories}
    
    total_weight = 0
    for pred in successful_predictions:
        model_name = pred['model']
        weight = weights.get(model_name, 0.1)
        total_weight += weight
        
        for cat in categories:
            ensemble_probs[cat] += pred['probabilities'].get(cat, 0) * weight
    
    # Normalize probabilities
    for cat in categories:
        ensemble_probs[cat] /= total_weight
    
    # Get final prediction
    final_category = max(ensemble_probs, key=ensemble_probs.get)
    final_confidence = ensemble_probs[final_category]
    
    return {
        'model': 'Ensemble',
        'category': final_category,
        'confidence': final_confidence,
        'probabilities': ensemble_probs,
        'success': True,
        'timestamp': datetime.datetime.now()
    }

def get_all_predictions(text, selected_models, confidence_threshold=0.5):
    """Get predictions from selected models"""
    results = {}
    
    model_mapping = {
        'Naive Bayes': models['nb_model'],
        'Logistic Regression': models['lr_model'],
        'Random Forest': models['rf_model']
    }
    
    # Get traditional model predictions
    for model_name in selected_models:
        if model_name in model_mapping:
            results[model_name] = predict_traditional_model(
                text, model_mapping[model_name], model_name, confidence_threshold
            )
    
    # Get LSTM prediction
    if 'LSTM' in selected_models:
        results['LSTM'] = predict_lstm_model(text, confidence_threshold)
    
    # Get ensemble prediction if multiple models selected
    if len([r for r in results.values() if r.get('success', False)]) > 1:
        results['Ensemble'] = get_ensemble_prediction(results)
    
    return results

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_confidence_chart(predictions):
    """Create confidence comparison chart"""
    successful_preds = {k: v for k, v in predictions.items() if v.get('success', False)}
    
    if not successful_preds:
        return None
    
    models = list(successful_preds.keys())
    confidences = [successful_preds[model]['confidence'] for model in models]
    categories = [successful_preds[model]['category'] for model in models]
    
    # Color mapping for sentiments
    color_map = {
        'positive': '#28a745',
        'negative': '#dc3545',
        'neutral': '#ffc107',
        'Uncertain': '#6c757d'
    }
    
    colors = [color_map.get(cat.lower(), '#17a2b8') for cat in categories]
    
    fig = go.Figure(data=[
        go.Bar(
            x=models,
            y=confidences,
            text=[f"{cat}<br>{conf:.3f}" for cat, conf in zip(categories, confidences)],
            textposition='auto',
            marker_color=colors,
            hovertemplate='<b>%{x}</b><br>Confidence: %{y:.3f}<br>Prediction: %{text}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Model Confidence Comparison",
        xaxis_title="Models",
        yaxis_title="Confidence Score",
        yaxis=dict(range=[0, 1]),
        height=400,
        showlegend=False
    )
    
    return fig

def create_probability_heatmap(predictions):
    """Create probability distribution heatmap"""
    successful_preds = {k: v for k, v in predictions.items() if v.get('success', False)}
    
    if not successful_preds:
        return None
    
    # Prepare data for heatmap
    categories = list(models['label_encoder'].classes_)
    model_names = list(successful_preds.keys())
    
    prob_matrix = []
    for model in model_names:
        probs = [successful_preds[model]['probabilities'].get(cat, 0) for cat in categories]
        prob_matrix.append(probs)
    
    fig = go.Figure(data=go.Heatmap(
        z=prob_matrix,
        x=categories,
        y=model_names,
        colorscale='RdYlBu_r',
        text=[[f"{val:.3f}" for val in row] for row in prob_matrix],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title="Probability Distribution Across Models",
        xaxis_title="Sentiment Categories",
        yaxis_title="Models",
        height=400
    )
    
    return fig

def create_sentiment_pie_chart(prediction):
    """Create pie chart for probability distribution"""
    if not prediction or not prediction.get('success', False):
        return None
    
    probabilities = prediction['probabilities']
    
    fig = go.Figure(data=[go.Pie(
        labels=list(probabilities.keys()),
        values=list(probabilities.values()),
        hole=0.4,
        textinfo='label+percent',
        textposition='auto',
        marker_colors=['#28a745', '#dc3545', '#ffc107'][:len(probabilities)]
    )])
    
    fig.update_layout(
        title=f"Sentiment Distribution - {prediction['model']}",
        height=400,
        showlegend=True
    )
    
    return fig

def create_history_trend_chart():
    """Create trend chart from analysis history"""
    if not st.session_state.analysis_history:
        return None
    
    df = pd.DataFrame(st.session_state.analysis_history)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Count sentiments over time
    sentiment_counts = df.groupby(['timestamp', 'category']).size().unstack(fill_value=0)
    
    fig = go.Figure()
    
    for category in sentiment_counts.columns:
        fig.add_trace(go.Scatter(
            x=sentiment_counts.index,
            y=sentiment_counts[category],
            mode='lines+markers',
            name=category.title(),
            line=dict(width=3)
        ))
    
    fig.update_layout(
        title="Sentiment Analysis History",
        xaxis_title="Time",
        yaxis_title="Count",
        height=400,
        hovermode='x unified'
    )
    
    return fig

def generate_wordcloud(text):
    """Generate word cloud from text"""
    if not text.strip():
        return None
    
    try:
        wordcloud = WordCloud(
            width=800, 
            height=400, 
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate(text)
        
        # Convert to base64 for displaying in Streamlit
        img = BytesIO()
        wordcloud.to_image().save(img, format='PNG')
        img.seek(0)
        
        return img
    except Exception as e:
        st.error(f"Error generating word cloud: {e}")
        return None

# ============================================================================
# STREAMLIT APP MAIN INTERFACE
# ============================================================================

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🎭 Advanced Sentiment Analyzer</h1>', unsafe_allow_html=True)
    st.markdown("### Powered by Multiple AI Models | Real-time Analysis | Advanced Features")
    
    # Sidebar configuration
    st.sidebar.title("🎛️ Configuration")
    
    # Model selection
    st.sidebar.subheader("Select Models")
    available_models = ['Naive Bayes', 'Logistic Regression', 'Random Forest', 'LSTM']
    selected_models = st.sidebar.multiselect(
        "Choose models for analysis:",
        available_models,
        default=['Logistic Regression', 'LSTM']
    )
    
    # Advanced settings
    st.sidebar.subheader("Advanced Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Predictions below this threshold will be marked as 'Uncertain'"
    )
    
    st.session_state.advanced_cleaning = st.sidebar.checkbox(
        "Advanced Text Cleaning",
        value=False,
        help="Apply additional preprocessing steps"
    )
    
    real_time_mode = st.sidebar.checkbox(
        "Real-time Analysis",
        value=False,
        help="Analyze text as you type"
    )
    
    # Main content tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔍 Single Analysis", 
        "📊 Batch Analysis", 
        "📈 Analytics Dashboard", 
        "🕒 History", 
        "ℹ️ Model Info"
    ])
    
    # ========================================================================
    # TAB 1: SINGLE ANALYSIS
    # ========================================================================
    
    with tab1:
        st.subheader("🔍 Single Text Analysis")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Text input options
            input_method = st.radio(
                "Choose input method:",
                ["Type text", "Upload file"],
                horizontal=True
            )
            
            if input_method == "Type text":
                text_input = st.text_area(
                    "Enter text for sentiment analysis:",
                    height=150,
                    placeholder="Type your text here... (e.g., 'I love this product!', 'This service is terrible', etc.)"
                )
            else:
                uploaded_file = st.file_uploader(
                    "Upload a text file:",
                    type=['txt', 'csv'],
                    help="Upload a .txt file or .csv file with text content"
                )
                text_input = ""
                if uploaded_file:
                    if uploaded_file.type == "text/plain":
                        text_input = str(uploaded_file.read(), "utf-8")
                    elif uploaded_file.type == "text/csv":
                        df = pd.read_csv(uploaded_file)
                        if 'text' in df.columns:
                            text_input = ' '.join(df['text'].astype(str))
                        else:
                            st.error("CSV file must have a 'text' column")
            
            # Real-time analysis
            if real_time_mode and text_input and len(text_input) > 10:
                with st.spinner("Analyzing..."):
                    time.sleep(0.5)  # Small delay to avoid too frequent updates
                    predictions = get_all_predictions(text_input, selected_models, confidence_threshold)
                    
                    # Display real-time results
                    for model_name, pred in predictions.items():
                        if pred.get('success', False):
                            sentiment_class = f"sentiment-{pred['category'].lower()}"
                            st.markdown(f"""
                            <div class="{sentiment_class}">
                                <strong>{model_name}:</strong> {pred['category'].title()} 
                                (Confidence: {pred['confidence']:.3f})
                            </div>
                            """, unsafe_allow_html=True)
        
        with col2:
            # Quick analysis buttons
            st.subheader("Quick Examples")
            sample_texts = [
                "This product is absolutely amazing! Best purchase ever!",
                "I'm really disappointed with the service quality.",
                "The weather is okay today, nothing special.",
                "I absolutely hate waiting in long queues!",
                "This movie was fantastic, highly recommend it!"
            ]
            
            for i, sample in enumerate(sample_texts):
                if st.button(f"Example {i+1}", key=f"sample_{i}"):
                    text_input = sample
                    st.rerun()
        
        # Analysis button and results
        if st.button("🚀 Analyze Sentiment", type="primary", disabled=not text_input or not selected_models):
            if text_input and selected_models:
                with st.spinner("🤖 AI models are analyzing..."):
                    # Get text statistics
                    text_stats = get_text_statistics(text_input)
                    
                    # Get predictions
                    predictions = get_all_predictions(text_input, selected_models, confidence_threshold)
                    
                    # Store in history
                    for model_name, pred in predictions.items():
                        if pred.get('success', False):
                            history_entry = {
                                'text': text_input[:100] + "..." if len(text_input) > 100 else text_input,
                                'category': pred['category'],
                                'confidence': pred['confidence'],
                                'model': model_name,
                                'timestamp': datetime.datetime.now()
                            }
                            st.session_state.analysis_history.append(history_entry)
                
                # Display results
                st.success("✅ Analysis completed!")
                
                # Text statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Words", text_stats['word_count'])
                with col2:
                    st.metric("Characters", text_stats['character_count'])
                with col3:
                    st.metric("Sentences", text_stats['sentence_count'])
                with col4:
                    st.metric("Unique Words", text_stats['unique_words'])
                
                # Model predictions
                st.subheader("📊 Prediction Results")
                
                successful_predictions = {k: v for k, v in predictions.items() if v.get('success', False)}
                
                for model_name, pred in successful_predictions.items():
                    with st.expander(f"🤖 {model_name} Results", expanded=True):
                        col1, col2 = st.columns([1, 2])
                        
                        with col1:
                            # Sentiment display with styling
                            sentiment = pred['category'].lower()
                            confidence = pred['confidence']
                            
                            if sentiment == 'positive':
                                st.success(f"😊 **POSITIVE** (Confidence: {confidence:.3f})")
                            elif sentiment == 'negative':
                                st.error(f"😞 **NEGATIVE** (Confidence: {confidence:.3f})")
                            elif sentiment == 'neutral':
                                st.warning(f"😐 **NEUTRAL** (Confidence: {confidence:.3f})")
                            else:
                                st.info(f"🤔 **{sentiment.upper()}** (Confidence: {confidence:.3f})")
                        
                        with col2:
                            # Probability distribution
                            prob_df = pd.DataFrame(
                                list(pred['probabilities'].items()),
                                columns=['Sentiment', 'Probability']
                            )
                            prob_df['Probability'] = prob_df['Probability'].round(3)
                            st.dataframe(prob_df, use_container_width=True)
                
                # Visualizations
                if len(successful_predictions) > 1:
                    st.subheader("📈 Visual Analysis")
                    
                    viz_col1, viz_col2 = st.columns(2)
                    
                    with viz_col1:
                        confidence_chart = create_confidence_chart(predictions)
                        if confidence_chart:
                            st.plotly_chart(confidence_chart, use_container_width=True)
                    
                    with viz_col2:
                        heatmap_chart = create_probability_heatmap(predictions)
                        if heatmap_chart:
                            st.plotly_chart(heatmap_chart, use_container_width=True)
                
                # Word Cloud
                st.subheader("☁️ Word Cloud")
                wordcloud_img = generate_wordcloud(text_input)
                if wordcloud_img:
                    st.image(wordcloud_img, use_container_width=True)
    
    # ========================================================================
    # TAB 2: BATCH ANALYSIS
    # ========================================================================
    
    with tab2:
        st.subheader("📊 Batch Text Analysis")
        
        # File upload for batch processing
        col1, col2 = st.columns([2, 1])
        
        with col1:
            batch_file = st.file_uploader(
                "Upload CSV file for batch analysis:",
                type=['csv'],
                help="CSV file should have a 'text' column containing the texts to analyze"
            )
            
            if batch_file:
                try:
                    batch_df = pd.read_csv(batch_file)
                    st.write("📋 Data Preview:")
                    st.dataframe(batch_df.head(), use_container_width=True)
                    
                    if 'text' not in batch_df.columns:
                        st.error("❌ CSV file must contain a 'text' column")
                    else:
                        st.success(f"✅ Found {len(batch_df)} texts to analyze")
                        
                        if st.button("🚀 Start Batch Analysis", type="primary"):
                            # Progress bar
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            batch_results = []
                            total_texts = len(batch_df)
                            
                            for idx, row in batch_df.iterrows():
                                text = str(row['text'])
                                
                                # Update progress
                                progress = (idx + 1) / total_texts
                                progress_bar.progress(progress)
                                status_text.text(f"Analyzing text {idx + 1} of {total_texts}")
                                
                                # Get predictions
                                predictions = get_all_predictions(text, selected_models, confidence_threshold)
                                
                                # Store results
                                for model_name, pred in predictions.items():
                                    if pred.get('success', False):
                                        result = {
                                            'text_id': idx,
                                            'text_preview': text[:50] + "..." if len(text) > 50 else text,
                                            'model': model_name,
                                            'sentiment': pred['category'],
                                            'confidence': pred['confidence']
                                        }
                                        batch_results.append(result)
                            
                            # Store results in session state
                            st.session_state.batch_results = batch_results
                            
                            # Clear progress indicators
                            progress_bar.empty()
                            status_text.empty()
                            
                            st.success("✅ Batch analysis completed!")
                
                except Exception as e:
                    st.error(f"❌ Error reading file: {e}")
        
        with col2:
            st.subheader("📝 Sample CSV Format")
            sample_data = pd.DataFrame({
                'text': [
                    'This product is amazing!',
                    'I hate this service',
                    'It\'s okay, nothing special',
                    'Absolutely love it!',
                    'Could be better'
                ]
            })
            st.dataframe(sample_data, use_container_width=True)
            
            # Download sample CSV
            csv_sample = sample_data.to_csv(index=False)
            st.download_button(
                label="📥 Download Sample CSV",
                data=csv_sample,
                file_name="sample_sentiment_data.csv",
                mime="text/csv"
            )
        
        # Display batch results
        if st.session_state.batch_results:
            st.subheader("📊 Batch Analysis Results")
            
            # Convert to DataFrame
            results_df = pd.DataFrame(st.session_state.batch_results)
            
            # Summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_analyzed = len(results_df['text_id'].unique())
                st.metric("Total Texts", total_analyzed)
            
            with col2:
                avg_confidence = results_df['confidence'].mean()
                st.metric("Avg Confidence", f"{avg_confidence:.3f}")
            
            with col3:
                positive_pct = (results_df['sentiment'] == 'positive').mean() * 100
                st.metric("Positive %", f"{positive_pct:.1f}%")
            
            with col4:
                negative_pct = (results_df['sentiment'] == 'negative').mean() * 100
                st.metric("Negative %", f"{negative_pct:.1f}%")
            
            # Detailed results table
            st.dataframe(results_df, use_container_width=True)
            
            # Download results
            csv_results = results_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Results CSV",
                data=csv_results,
                file_name=f"sentiment_analysis_results_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
            # Batch visualization
            if len(results_df) > 0:
                st.subheader("📈 Batch Analysis Visualization")
                
                # Sentiment distribution
                sentiment_counts = results_df['sentiment'].value_counts()
                fig_pie = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title="Overall Sentiment Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
                
                # Model comparison
                if len(results_df['model'].unique()) > 1:
                    model_comparison = results_df.groupby(['model', 'sentiment']).size().unstack(fill_value=0)
                    fig_bar = px.bar(
                        model_comparison,
                        title="Sentiment Distribution by Model",
                        barmode='group'
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
    
    # ========================================================================
    # TAB 3: ANALYTICS DASHBOARD
    # ========================================================================
    
    with tab3:
        st.subheader("📈 Analytics Dashboard")
        
        if st.session_state.analysis_history:
            history_df = pd.DataFrame(st.session_state.analysis_history)
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_analyses = len(history_df)
                st.metric("Total Analyses", total_analyses)
            
            with col2:
                avg_confidence = history_df['confidence'].mean()
                st.metric("Average Confidence", f"{avg_confidence:.3f}")
            
            with col3:
                most_common_sentiment = history_df['category'].mode().iloc[0] if not history_df.empty else "N/A"
                st.metric("Most Common Sentiment", most_common_sentiment.title())
            
            with col4:
                high_confidence_pct = (history_df['confidence'] >= 0.8).mean() * 100
                st.metric("High Confidence %", f"{high_confidence_pct:.1f}%")
            
            # Trend analysis
            trend_chart = create_history_trend_chart()
            if trend_chart:
                st.plotly_chart(trend_chart, use_container_width=True)
            
            # Model performance comparison
            st.subheader("🤖 Model Performance Analysis")
            
            model_stats = history_df.groupby('model').agg({
                'confidence': ['mean', 'std', 'count'],
                'category': lambda x: x.mode().iloc[0] if len(x) > 0 else 'N/A'
            }).round(3)
            
            st.dataframe(model_stats, use_container_width=True)
            
            # Confidence distribution
            fig_hist = px.histogram(
                history_df,
                x='confidence',
                color='model',
                title="Confidence Score Distribution by Model",
                nbins=20
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Sentiment timeline
            if 'timestamp' in history_df.columns:
                history_df['timestamp'] = pd.to_datetime(history_df['timestamp'])
                daily_sentiment = history_df.groupby([
                    history_df['timestamp'].dt.date,
                    'category'
                ]).size().unstack(fill_value=0)
                
                if not daily_sentiment.empty:
                    fig_timeline = px.line(
                        daily_sentiment,
                        title="Daily Sentiment Trends",
                        labels={'index': 'Date', 'value': 'Count'}
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)
            
        else:
            st.info("📝 No analysis history available. Start analyzing some texts to see analytics!")
    
    # ========================================================================
    # TAB 4: HISTORY
    # ========================================================================
    
    with tab4:
        st.subheader("🕒 Analysis History")
        
        if st.session_state.analysis_history:
            # Filter options
            col1, col2, col3 = st.columns(3)
            
            history_df = pd.DataFrame(st.session_state.analysis_history)
            
            with col1:
                model_filter = st.selectbox(
                    "Filter by Model:",
                    ['All'] + list(history_df['model'].unique())
                )
            
            with col2:
                sentiment_filter = st.selectbox(
                    "Filter by Sentiment:",
                    ['All'] + list(history_df['category'].unique())
                )
            
            with col3:
                min_confidence = st.slider(
                    "Minimum Confidence:",
                    0.0, 1.0, 0.0, 0.05
                )
            
            # Apply filters
            filtered_df = history_df.copy()
            
            if model_filter != 'All':
                filtered_df = filtered_df[filtered_df['model'] == model_filter]
            
            if sentiment_filter != 'All':
                filtered_df = filtered_df[filtered_df['category'] == sentiment_filter]
            
            filtered_df = filtered_df[filtered_df['confidence'] >= min_confidence]
            
            # Display filtered results
            st.write(f"Showing {len(filtered_df)} of {len(history_df)} analyses")
            
            # Format for display
            display_df = filtered_df.copy()
            if 'timestamp' in display_df.columns:
                display_df['timestamp'] = pd.to_datetime(display_df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            st.dataframe(
                display_df.sort_values('timestamp', ascending=False) if 'timestamp' in display_df.columns else display_df,
                use_container_width=True
            )
            
            # Export options
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("📥 Export History as CSV"):
                    csv_history = filtered_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv_history,
                        file_name=f"sentiment_history_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            with col2:
                if st.button("🗑️ Clear History", type="secondary"):
                    if st.button("⚠️ Confirm Clear", type="secondary"):
                        st.session_state.analysis_history = []
                        st.success("History cleared!")
                        st.rerun()
        
        else:
            st.info("📝 No analysis history available yet. Start analyzing some texts!")
    
    # ========================================================================
    # TAB 5: MODEL INFO
    # ========================================================================
    
    with tab5:
        st.subheader("ℹ️ Model Information")
        
        # Model descriptions
        model_info = {
            "Naive Bayes": {
                "description": "A probabilistic classifier based on Bayes' theorem with strong independence assumptions.",
                "strengths": ["Fast training and prediction", "Works well with small datasets", "Good baseline model"],
                "use_cases": ["Text classification", "Spam detection", "Quick sentiment analysis"]
            },
            "Logistic Regression": {
                "description": "A linear model that uses logistic function to model binary or multiclass classification.",
                "strengths": ["Interpretable coefficients", "Fast and efficient", "No tuning of hyperparameters"],
                "use_cases": ["Binary classification", "Feature importance analysis", "Baseline for comparison"]
            },
            "Random Forest": {
                "description": "An ensemble method that combines multiple decision trees for robust predictions.",
                "strengths": ["Handles overfitting well", "Feature importance", "Works with mixed data types"],
                "use_cases": ["Complex datasets", "Feature selection", "Robust predictions"]
            },
            "LSTM": {
                "description": "A type of recurrent neural network designed to learn long-term dependencies in sequences.",
                "strengths": ["Captures sequential patterns", "Handles variable-length input", "State-of-the-art performance"],
                "use_cases": ["Sequential data", "Complex language patterns", "High-accuracy predictions"]
            }
        }
        
        for model_name, info in model_info.items():
            with st.expander(f"🤖 {model_name}", expanded=False):
                st.write(f"**Description:** {info['description']}")
                st.write("**Strengths:**")
                for strength in info['strengths']:
                    st.write(f"• {strength}")
                st.write("**Use Cases:**")
                for use_case in info['use_cases']:
                    st.write(f"• {use_case}")
        
        # System information
        st.subheader("🖥️ System Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Available Categories:**")
            if models and 'label_encoder' in models:
                categories = list(models['label_encoder'].classes_)
                for category in categories:
                    st.write(f"• {category.title()}")
        
        with col2:
            st.write("**Model Statistics:**")
            st.write(f"• Total Models: {len(available_models)}")
            st.write(f"• Selected Models: {len(selected_models)}")
            st.write(f"• Confidence Threshold: {confidence_threshold}")
            st.write(f"• Advanced Cleaning: {'Enabled' if st.session_state.get('advanced_cleaning', False) else 'Disabled'}")
        
        # Performance tips
        st.subheader("🎯 Performance Tips")
        tips = [
            "Use multiple models for more robust predictions",
            "Adjust confidence threshold based on your use case",
            "Enable advanced cleaning for noisy text data",
            "Check the ensemble prediction for best overall results",
            "Use batch analysis for processing multiple texts efficiently"
        ]
        
        for tip in tips:
            st.write(f"💡 {tip}")

if __name__ == "__main__":
    main()