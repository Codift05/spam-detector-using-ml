# Spam Chat Detector - Machine Learning Classification System

A machine learning application for binary classification of chat messages as spam or legitimate content using Natural Language Processing (NLP) and statistical classification algorithms.

## Overview

This project implements a supervised learning approach to spam message detection. The system employs text preprocessing techniques combined with TF-IDF (Term Frequency-Inverse Document Frequency) feature extraction and logistic regression classification. The model achieves 81% accuracy on the test dataset with an F1-score of 82%.

## Key Features

- Binary classification for spam/legitimate message detection
- Comprehensive text preprocessing pipeline (normalization, punctuation removal, stopword filtering)
- TF-IDF vectorization for feature extraction (3000 features, bigrams enabled)
- Comparative analysis of multiple classification algorithms (Logistic Regression, Multinomial Naive Bayes)
- Web-based inference interface built with Streamlit
- Probabilistic confidence scores for predictions
- Performance visualization with confusion matrices and classification reports

## Project Structure

```
spam_chat_detector_using_ML/
│
├── data/
│   └── spam.csv                 # Message dataset (spam/ham labeled)
│
├── model/
│   ├── model.pkl                # Trained classification model
│   └── tfidf.pkl                # Fitted TF-IDF vectorizer
│
├── app/
│   └── app.py                   # Streamlit web application
│
├── notebook/
│   └── training.ipynb           # Jupyter notebook for model training
│
├── README.md                    # Project documentation
└── requirements.txt             # Python dependencies

```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:

```bash
git clone https://github.com/Codift05/spam-detector-using-ml.git
cd spam_chat_detector_using_ml
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. NLTK data is downloaded automatically on first run.

## Model Training

### Training Pipeline

Execute the training notebook to build and evaluate the classification model:

```bash
jupyter notebook notebook/training.ipynb
```

The training pipeline performs the following operations:

- Dataset loading and exploration
- Text normalization (lowercase conversion, punctuation removal, tokenization)
- Stopword filtering using NLTK corpus
- Feature extraction via TF-IDF vectorization
- Train-test split (80-20 ratio)
- Model training (Logistic Regression and Naive Bayes)
- Performance evaluation and comparison
- Model serialization and storage

### Training Output

Upon completion, the pipeline generates:

- Model performance metrics (accuracy, precision, recall, F1-score)
- Confusion matrices for each classification algorithm
- Detailed classification reports per class
- Serialized model and vectorizer files for inference

## Web Interface

### Running the Inference Application

Execute the Streamlit web application:

```bash
python -m streamlit run app/app.py
```

The application will launch at `http://localhost:8502` and provides:

- Text input interface for spam classification
- Real-time prediction with confidence scoring
- Probability distribution visualization
- Model preprocessing inspection (optional)

### Usage Example

Input:
```
URGENT! You've won a $5000 prize. Click here to claim now!
```

Expected Output:
```
Prediction: Spam
Confidence: 89.3%
Spam probability: 89.3%
Ham probability: 10.7%
```

## Technical Methodology

### Feature Extraction

Term Frequency-Inverse Document Frequency (TF-IDF) vectorization:
- Vocabulary size: 3000 features
- N-gram range: (1, 2) [unigrams and bigrams]
- Sublinear term frequency scaling: enabled
- Inverse document frequency weighting: enabled

### Text Normalization Pipeline

The preprocessing pipeline implements the following sequence:
1. Case normalization (conversion to lowercase)
2. Punctuation removal (preserving word boundaries)
3. Tokenization using NLTK punkt tokenizer
4. Stopword filtering (English corpus from NLTK)

### Classification Algorithms

Two baseline algorithms were evaluated:

1. **Logistic Regression**: Linear probabilistic classifier with L2 regularization (C=1.0)
2. **Multinomial Naive Bayes**: Probabilistic classifier based on Bayes' theorem

### Model Selection

Logistic Regression was selected as the production model based on superior performance metrics.

## Model Performance

### Test Set Metrics

- **Accuracy**: 81%
- **Precision**: 75%
- **Recall**: 90%
- **F1-Score**: 82%

### Evaluation Methodology

The model was evaluated using stratified k-fold cross-validation on an 80-20 train-test split of 105 labeled messages (52 spam, 53 legitimate).

## Dataset

### Data Specification

The training dataset (`data/spam.csv`) contains:

- **Total Samples**: 105 messages
- **Classes**: 2 (spam=1, ham=0)
- **Class Distribution**: 52 spam, 53 legitimate
- **Format**: CSV with text and label columns

### Data Format

```csv
text,label
"Congratulations! You've won $1000. Click to claim.",spam
"Hello, are we still meeting tomorrow?",ham
```

## Configuration and Customization

### Model Hyperparameters

To modify model parameters, edit `training.ipynb`:

```python
# Logistic Regression configuration
lr_model = LogisticRegression(max_iter=1000, C=1.0, random_state=42)

# TF-IDF configuration
tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
```

### Extending the Dataset

To add new labeled samples:

1. Append to `data/spam.csv` with format: `text,label`
2. Re-execute the training notebook
3. Restart the Streamlit application

## Troubleshooting

### Missing Model Files

**Issue**: Model files not found error during inference.

**Solution**: Execute the training notebook to generate serialized models (`model.pkl`, `tfidf.pkl`).

### NLTK Data Issues

**Issue**: LookupError for NLTK corpora or tokenizers.

**Solution**: Manually download required resources:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt_tab')
```

### Streamlit Startup Issues

**Issue**: Streamlit fails to start or connect.

**Solution**: Specify port explicitly and upgrade package:

```bash
pip install --upgrade streamlit
python -m streamlit run app/app.py --server.port 8502
```

## Future Work

Potential extensions and improvements:

- Expansion of training dataset for improved generalization
- Implementation of transformer-based models (BERT, DistilBERT)
- Multi-language support
- RESTful API interface for integration
- Batch prediction capabilities
- Model versioning and A/B testing framework
- Cloud deployment (Streamlit Cloud, Azure, AWS)

## Dependencies

```
numpy>=1.24.3
pandas>=2.0.3
scikit-learn>=1.3.0
nltk>=3.8.1
matplotlib>=3.7.2
seaborn>=0.12.2
streamlit>=1.26.0
```

## Methodology References

This project implements standard NLP and machine learning techniques:

- TF-IDF Feature Extraction: Sparse Vector Space Model
- Logistic Regression: Supervised linear classification
- Naive Bayes: Probabilistic classification using conditional independence
- Text preprocessing: Standard NLP pipeline

## License

MIT License. See LICENSE file for details.

## Contact

For questions or contributions, please open an issue on the repository.

---

**Repository**: https://github.com/Codift05/spam-detector-using-ml
