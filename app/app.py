import streamlit as st
import pickle
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os

# Download NLTK data if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')

# Set page configuration
st.set_page_config(
    page_title="Spam Chat Detector",
    page_icon="🛡",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern UI
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main {
        background: #f8f9fa;
        padding: 0;
    }
    
    .block-container {
        padding: 2rem 1rem;
        max-width: 1200px;
    }
    
    /* Header Card */
    .header-card {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid #e9ecef;
    }
    
    .header-card h1 {
        color: #212529;
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .header-card p {
        color: #6c757d;
        font-size: 1rem;
        font-weight: 400;
    }
    
    /* Stats Cards Container */
    .stats-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 1.5rem;
        margin-bottom: 2rem;
    }
    
    .stat-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid #e9ecef;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .stat-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.12);
    }
    
    .stat-icon {
        width: 48px;
        height: 48px;
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 1rem;
        font-size: 24px;
    }
    
    .stat-card h3 {
        color: #6c757d;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .stat-card p {
        color: #212529;
        font-size: 1.75rem;
        font-weight: 700;
        margin: 0;
    }
    
    /* Main Card */
    .main-card {
        background: white;
        border-radius: 12px;
        padding: 2rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
        border: 1px solid #e9ecef;
        margin-bottom: 2rem;
    }
    
    /* Input Section */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e2e8f0;
        font-size: 16px;
        padding: 1rem;
        transition: border-color 0.3s ease;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Button Styles */
    .stButton button {
        background: #4c6ef5;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    .stButton button:hover {
        background: #3b5bdb;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(76, 110, 245, 0.25);
    }
    
    /* Result Cards */
    .result-card {
        border-radius: 16px;
        padding: 2rem;
        margin: 1.5rem 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .result-spam {
        background: #fff5f5;
        border: 2px solid #ff6b6b;
        color: #c92a2a;
    }
    
    .result-safe {
        background: #f3faf7;
        border: 2px solid #51cf66;
        color: #2b8a3e;
    }
    
    .result-icon {
        width: 64px;
        height: 64px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        margin: 0 auto 1rem;
        font-size: 32px;
    }
    
    .result-icon-spam {
        background: #ffe3e3;
    }
    
    .result-icon-safe {
        background: #d3f9e1;
    }
    
    .result-title {
        font-size: 1.75rem;
        font-weight: 700;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    
    .result-confidence {
        font-size: 1.25rem;
        text-align: center;
        opacity: 0.9;
    }
    
    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
    }
    
    .metric-label {
        color: #718096;
        font-size: 0.875rem;
        font-weight: 500;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        color: #1a202c;
        font-size: 1.5rem;
        font-weight: 700;
    }
    
    /* Progress Bar Custom */
    .progress-container {
        background: #e2e8f0;
        border-radius: 10px;
        height: 12px;
        overflow: hidden;
        margin: 0.5rem 0;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    .progress-ham {
        background: #51cf66;
    }
    
    .progress-spam {
        background: #ff6b6b;
    }
    
    /* Info Alert */
    .info-alert {
        background: #e7f5ff;
        border-left: 4px solid #4c6ef5;
        color: #1864ab;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin: 1rem 0;
        display: flex;
        align-items: center;
        gap: 1rem;
    }
    
    /* Example Buttons */
    .example-btn {
        background: white;
        border: 2px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        color: #4a5568;
        font-weight: 500;
        transition: all 0.3s ease;
        cursor: pointer;
        display: inline-block;
        margin: 0.5rem;
    }
    
    .example-btn:hover {
        border-color: #667eea;
        color: #667eea;
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.2);
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    </style>
""", unsafe_allow_html=True)

# Function to load model and vectorizer
@st.cache_resource
def load_models():
    """Load trained model and TF-IDF vectorizer"""
    try:
        # Get the absolute path to model directory
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_dir = os.path.join(os.path.dirname(current_dir), 'model')
        
        model_path = os.path.join(model_dir, 'model.pkl')
        tfidf_path = os.path.join(model_dir, 'tfidf.pkl')
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(tfidf_path, 'rb') as f:
            tfidf = pickle.load(f)
        
        return model, tfidf
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        st.info("Please run the training notebook first to generate the model files.")
        return None, None

# Text preprocessing function
def preprocess_text(text):
    """
    Preprocess text:
    1. Lowercase
    2. Remove punctuation
    3. Remove stopwords
    """
    # Lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Join tokens back to string
    return ' '.join(tokens)

# Prediction function
def predict_spam(text, model, vectorizer):
    """
    Predict if text is spam or not
    Returns: prediction, probability
    """
    # Preprocess text
    cleaned_text = preprocess_text(text)
    
    # Vectorize text
    text_vectorized = vectorizer.transform([cleaned_text])
    
    # Predict
    prediction = model.predict(text_vectorized)[0]
    probability = model.predict_proba(text_vectorized)[0]
    
    # Get probability for the predicted class
    if prediction == 'spam':
        confidence = probability[1] * 100  # Spam probability
    else:
        confidence = probability[0] * 100  # Ham probability
    
    return prediction, confidence, probability

# Main app
def main():
    # Load models
    model, tfidf = load_models()
    
    if model is None or tfidf is None:
        st.markdown("""
            <div class='main-card' style='text-align: center; padding: 3rem;'>
                <svg width="80" height="80" viewBox="0 0 24 24" style="margin: 0 auto 1rem;">
                    <circle cx="12" cy="12" r="11" fill="#4c6ef5" opacity="0.1" stroke="#4c6ef5" stroke-width="0.5"/>
                    <text x="12" y="16" font-size="20" font-weight="bold" fill="#4c6ef5" text-anchor="middle">!</text>
                </svg>
                <h2 style="color: #1a202c; margin-bottom: 1rem;">Model Not Found</h2>
                <p style="color: #718096;">Please run the training notebook first to generate the model files.</p>
            </div>
        """, unsafe_allow_html=True)
        st.stop()
    
    # Header Card
    st.markdown("""
        <div class='header-card'>
            <div style="display: flex; align-items: center; justify-content: space-between;">
                <div>
                    <h1 style="margin: 0;">Spam Chat Detector</h1>
                    <p style="margin: 0.5rem 0 0 0;">Detect spam messages with machine learning</p>
                </div>
                <div style="width: 60px; height: 60px; background: #4c6ef5; border-radius: 12px; display: flex; align-items: center; justify-content: center;">
                    <svg width="32" height="32" viewBox="0 0 24 24">
                        <circle cx="12" cy="8" r="3" fill="white"/>
                        <path d="M12 14c-4 0-6 2-6 4v2h12v-2c0-2-2-4-6-4z" fill="white"/>
                    </svg>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Stats Cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
            <div class='stat-card'>
                <div class='stat-icon' style='background: #e7f5ff;'>
                    <svg width="24" height="24" viewBox="0 0 24 24">
                        <circle cx="12" cy="12" r="10" fill="none" stroke="#4c6ef5" stroke-width="1.5"/>
                        <path d="M12 7v5l4 2.5" fill="none" stroke="#4c6ef5" stroke-width="1.5" stroke-linecap="round"/>
                    </svg>
                </div>
                <h3>ACCURACY</h3>
                <p>81%</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div class='stat-card'>
                <div class='stat-icon' style='background: #e7f5ff;'>
                    <svg width="24" height="24" viewBox="0 0 24 24">
                        <rect x="3" y="4" width="4" height="16" rx="1" fill="#4c6ef5"/>
                        <rect x="10" y="6" width="4" height="14" rx="1" fill="#4c6ef5"/>
                        <rect x="17" y="2" width="4" height="18" rx="1" fill="#4c6ef5"/>
                    </svg>
                </div>
                <h3>MODEL TYPE</h3>
                <p>Logistic</p>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
            <div class='stat-card'>
                <div class='stat-icon' style='background: #e7f5ff;'>
                    <svg width="24" height="24" viewBox="0 0 24 24">
                        <circle cx="12" cy="12" r="9" fill="none" stroke="#4c6ef5" stroke-width="1.5"/>
                        <path d="M12 7c1.5 0 2.7 1.2 2.7 2.7S13.5 12.4 12 12.4c-1.5 0-2.7-1.2-2.7-2.7S10.5 7 12 7z" fill="#4c6ef5"/>
                        <path d="M12 14c-2.4 0-4 1.3-4 3v2.5h8V17c0-1.7-1.6-3-4-3z" fill="#4c6ef5"/>
                    </svg>
                </div>
                <h3>F1-SCORE</h3>
                <p>82%</p>
            </div>
        """, unsafe_allow_html=True)
    
    # Info Alert
    st.markdown("""
        <div class='info-alert'>
            <svg width="20" height="20" viewBox="0 0 24 24">
                <circle cx="12" cy="12" r="10" fill="none" stroke="#4c6ef5" stroke-width="1.5"/>
                <circle cx="12" cy="7" r="1.2" fill="#4c6ef5"/>
                <path d="M12 10v6" stroke="#4c6ef5" stroke-width="1.5" stroke-linecap="round"/>
            </svg>
            <div>
                <strong>Tip:</strong> Try different message types to see how the model detects spam patterns
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Main Input Card
    st.markdown("<div class='main-card'>", unsafe_allow_html=True)
    
    st.markdown("""
        <div style='margin-bottom: 1.5rem;'>
            <h2 style='color: #212529; font-size: 1.25rem; font-weight: 600; margin-bottom: 0.5rem;'>
                Message Input
            </h2>
            <p style='color: #6c757d; font-size: 0.9rem;'>Type or paste the message you want to check</p>
        </div>
    """, unsafe_allow_html=True)
    
    user_input = st.text_area(
        "Message Content",
        height=150,
        placeholder="Example: Congratulations! You've won $1000. Click here to claim now!",
        label_visibility="collapsed"
    )
    
    # Example buttons
    st.markdown("<div style='margin: 1.5rem 0;'>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        spam_example = st.button("📧 Spam Example", use_container_width=True)
    
    with col2:
        normal_example = st.button("✉️ Normal Example", use_container_width=True)
    
    with col3:
        clear_btn = st.button("🗑️ Clear", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Handle example buttons
    if spam_example:
        user_input = "URGENT! You've won a $5000 prize. Click here to claim now!"
    if normal_example:
        user_input = "Hey, are we still meeting for lunch tomorrow?"
    if clear_btn:
        user_input = ""
    
    # Predict button
    detect_btn = st.button("🔍 Detect Spam", type="primary", use_container_width=True)
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if detect_btn:
        if user_input.strip() == "":
            st.markdown("""
                <div class='main-card' style='text-align: center; padding: 2rem;'>
                    <svg width="60" height="60" viewBox="0 0 24 24" style="margin: 0 auto 1rem;">
                        <circle cx="12" cy="12" r="10" fill="none" stroke="#4c6ef5" stroke-width="1.5"/>
                        <path d="M8 12h8M12 8v8" stroke="#4c6ef5" stroke-width="1.5" stroke-linecap="round"/>
                    </svg>
                    <p style="color: #4c6ef5; font-weight: 500;">Please enter a message to analyze</p>
                </div>
            """, unsafe_allow_html=True)
        else:
            with st.spinner("Analyzing message..."):
                # Make prediction
                prediction, confidence, probability = predict_spam(user_input, model, tfidf)
                
                # Display results
                if prediction == 'spam':
                    st.markdown(f"""
                        <div class='result-card result-spam'>
                            <div class='result-icon result-icon-spam'>
                                <svg width="32" height="32" viewBox="0 0 24 24">
                                    <circle cx="12" cy="12" r="9" fill="none" stroke="#4c6ef5" stroke-width="1.5"/>
                                    <path d="M8 12h8" stroke="#4c6ef5" stroke-width="2" stroke-linecap="round"/>
                                </svg>
                            </div>
                            <div class='result-title'>Spam Detected</div>
                            <div class='result-confidence' style='color: #4c6ef5; font-weight: 600;'>{confidence:.1f}% confidence</div>
                            <p style='margin-top: 1rem; color: #495057; font-size: 0.95rem;'>
                                This message appears to be spam. Exercise caution.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                else:
                    st.markdown(f"""
                        <div class='result-card result-safe'>
                            <div class='result-icon result-icon-safe'>
                                <svg width="32" height="32" viewBox="0 0 24 24">
                                    <circle cx="12" cy="12" r="9" fill="none" stroke="#4c6ef5" stroke-width="1.5"/>
                                    <path d="M9 12.5l2.5 2.5 4-4" stroke="#4c6ef5" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
                                </svg>
                            </div>
                            <div class='result-title'>Safe Message</div>
                            <div class='result-confidence' style='color: #4c6ef5; font-weight: 600;'>{confidence:.1f}% confidence</div>
                            <p style='margin-top: 1rem; color: #495057; font-size: 0.95rem;'>
                                This message appears to be legitimate and safe.
                            </p>
                        </div>
                    """, unsafe_allow_html=True)
                
                # Probability Details
                st.markdown("<div class='main-card' style='margin-top: 1.5rem;'>", unsafe_allow_html=True)
                st.markdown("""
                    <h3 style='color: #212529; font-size: 1.1rem; font-weight: 600; margin-bottom: 1.5rem;'>
                        Probability Analysis
                    </h3>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-label'>
                                <svg width="16" height="16" viewBox="0 0 24 24" style="vertical-align: middle; margin-right: 0.5rem;">
                                    <circle cx="12" cy="12" r="5" fill="#4c6ef5"/>
                                </svg>
                                Normal Message (Ham)
                            </div>
                            <div class='metric-value' style='color: #4c6ef5;'>{probability[0]*100:.1f}%</div>
                            <div class='progress-container'>
                                <div class='progress-bar progress-ham' style='width: {probability[0]*100}%;'></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                        <div class='metric-card'>
                            <div class='metric-label'>
                                <svg width="16" height="16" viewBox="0 0 24 24" style="vertical-align: middle; margin-right: 0.5rem;">
                                    <circle cx="12" cy="12" r="5" fill="#4c6ef5"/>
                                </svg>
                                Spam Message
                            </div>
                            <div class='metric-value' style='color: #4c6ef5;'>{probability[1]*100:.1f}%</div>
                            <div class='progress-container'>
                                <div class='progress-bar progress-spam' style='width: {probability[1]*100}%;'></div>
                            </div>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Preprocessed Text
                with st.expander("🔍 View Preprocessed Text"):
                    cleaned = preprocess_text(user_input)
                    st.markdown("""
                        <div class='metric-card'>
                            <div class='metric-label'>Original Message</div>
                            <p style='color: #1a202c; padding: 0.75rem; background: #f7fafc; border-radius: 8px; margin: 0.5rem 0;'>{}</p>
                            <div class='metric-label' style='margin-top: 1rem;'>After Processing</div>
                            <p style='color: #1a202c; padding: 0.75rem; background: #f7fafc; border-radius: 8px; margin: 0.5rem 0;'>{}</p>
                            <p style='color: #718096; font-size: 0.85rem; margin-top: 0.5rem;'>
                                <em>Processed: lowercase, punctuation removed, stopwords filtered</em>
                            </p>
                        </div>
                    """.format(user_input[:200] + "..." if len(user_input) > 200 else user_input, 
                              cleaned[:200] + "..." if len(cleaned) > 200 else cleaned), 
                    unsafe_allow_html=True)
    
    # Footer
    st.markdown("""
        <div style='margin-top: 3rem; padding: 2rem; background: white; border-radius: 12px; border: 1px solid #e9ecef;'>
            <div style='text-align: center;'>
                <div style='display: flex; justify-content: center; gap: 2rem; margin-bottom: 1rem; flex-wrap: wrap;'>
                    <div style='display: flex; align-items: center; gap: 0.5rem; color: #495057;'>
                        <svg width="18" height="18" viewBox="0 0 24 24">
                            <circle cx="12" cy="12" r="10" fill="none" stroke="#4c6ef5" stroke-width="1.5"/>
                            <path d="M12 7v5l3 1.5" fill="none" stroke="#4c6ef5" stroke-width="1" stroke-linecap="round"/>
                        </svg>
                        <span style='font-size: 0.9rem;'>ML Powered</span>
                    </div>
                    <div style='display: flex; align-items: center; gap: 0.5rem; color: #495057;'>
                        <svg width="18" height="18" viewBox="0 0 24 24">
                            <rect x="3" y="3" width="6" height="6" rx="1" fill="#4c6ef5"/>
                            <rect x="15" y="3" width="6" height="6" rx="1" fill="#4c6ef5"/>
                            <rect x="3" y="15" width="6" height="6" rx="1" fill="#4c6ef5"/>
                            <rect x="15" y="15" width="6" height="6" rx="1" fill="#4c6ef5"/>
                        </svg>
                        <span style='font-size: 0.9rem;'>Real-time</span>
                    </div>
                    <div style='display: flex; align-items: center; gap: 0.5rem; color: #495057;'>
                        <svg width="18" height="18" viewBox="0 0 24 24">
                            <path d="M12 2c-5.5 0-10 4.5-10 10s4.5 10 10 10 10-4.5 10-10-4.5-10-10-10z" fill="none" stroke="#4c6ef5" stroke-width="1.5"/>
                            <path d="M8 12h8M12 8v8" stroke="#4c6ef5" stroke-width="1" stroke-linecap="round"/>
                        </svg>
                        <span style='font-size: 0.9rem;'>Secure</span>
                    </div>
                </div>
                <p style='color: #6c757d; font-size: 0.85rem; margin: 0.5rem 0;'>
                    Built with Streamlit • Trained on Spam Detection Dataset
                </p>
                <p style='color: #adb5bd; font-size: 0.8rem; margin: 0;'>
                    © 2025 Spam Chat Detector
                </p>
            </div>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
