# Fake News Detector – Complete Project

This repository implements a full pipeline for fake news detection, combining deep learning, NLP feature engineering, and a browser extension for real-time news validation.

## Project Overview

- **Goal**: Detect whether a news article is real or fake, using both linguistic analysis and neural models.
- **Approach**:
  - Data cleaning and preprocessing
  - Linguistic feature extraction (readability, polarity, POS ratios, etc.)
  - Training multiple classifiers (Logistic Regression, SVC, Decision Tree, BiLSTM)
  - Testing various embeddings (TF-IDF, Word2Vec, GloVe, BERT)
  - Exporting best model (BERT + BiLSTM + features)
  - Deploying as a real-time Chrome extension

## Best Model

The final deployed model is:
- **BERT + BiLSTM + Linguistic Features**
- Trained with `batch size = 32`, `dropout = 0.3`, early stopping on F1
- Exported to: `Extension/bert_lstm_features_32_model_extensie.pt`

## Extension Capabilities

- Classifies the current article using a local AI model
- Generates a simple factual claim from the text
- Optionally queries the **Google Fact Check API** for similar claims
- Interface built with popup + server communication via content/background scripts
