# Chrome Extension - Fake News Detector

This extension is part of a complete fake news detection system and allows users to analyze online news articles directly within their browser. It uses an advanced AI model to classify articles as "REAL" or "FAKE", and integrates with the Google Fact Check API to increase credibility.

## Features

- AI-based automatic classification using a fine-tuned BERT + LSTM model (batch size: 32)
- Automatic generation of a factual claim extracted from the article
- Fact verification through Google Fact Check API
- Textual feature extraction: readability, sentiment, syntactic structure, etc.
- All classification happens locally on the user's device

## Selected AI Model

The best-performing model identified during the evaluation phase is a hybrid architecture based on:
- BERT (bert-base-uncased) for contextual embeddings
- LSTM layer for sequential learning
- Concatenation with 17 handcrafted linguistic and stylistic features
- A linear layer for final binary classification ("REAL" vs "FAKE")

Model and scaler files:
- `bert_lstm_features_32_model_extensie.pt`
- `scaler_bert.pkl`

## Folder Structure

```
Extension/
│
├── app.py                # Flask server exposing prediction and hint generation endpoints
├── background.js         # Handles messaging and async requests to the server
├── content.js            # Extracts page title and paragraphs from the news article
├── popup.html            # Chrome extension interface
├── popup.js              # Client logic: fetches results and manages interaction
├── popup.css             # Styling for the popup
├── manifest.json         # Chrome extension manifest (v3)
├── bert_lstm_features_32_model_extensie.pt  # PyTorch model
└── scaler_bert.pkl       # Scaler for feature normalization
```

## How to Run

### 1. Start the Flask server

Make sure you have Python 3.8+ and the required packages installed. You can install them with:

```bash
pip install flask flask-cors torch transformers nltk textblob textstat joblib
```

Then run:

```bash
cd Extension
python app.py
```

This will launch the local server on `http://localhost:5000`.

### 2. Load the extension in Chrome

1. Open `chrome://extensions` in your browser
2. Enable **Developer mode** (top right)
3. Click **Load unpacked**
4. Select the `Extension/` directory

## Requirements

- Python 3.8+
- Chrome browser
- PyTorch, Transformers, Flask, NLTK, TextBlob, textstat

## Note

The Google Fact Check API key is included for demonstration purposes. In production environments, this key should be secured.


