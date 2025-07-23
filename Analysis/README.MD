# Full Model Analysis – Fake News Detector

This directory contains the entire research pipeline: from raw dataset processing and linguistic feature extraction, to training and comparing classical and neural models using various embeddings, with and without additional features.

## 1. Dataset

**WELFake** source: https://www.kaggle.com/datasets/MIKEDUD/WELFake

I've followed the WELFake dataset authors’ paper as a starting point for extracting linguistic and stylistic features, and extended their methodology.

## 2. Preprocessing (`preprocess.ipynb`)

Steps:
1. Drop nulls and duplicates
2. Clean text: remove special characters, extra spaces
3. Tokenization and lowercasing
4. Remove stopwords and punctuation
5. Lemmatization + POS tagging (NLTK)
6. Outlier removal using Z-score

## 3. Feature Extraction (`features_extraction.ipynb`)

Extracted over 30 features, including:
- **Readability**: Gunning Fog, SMOG, ARI
- **Psycholinguistics**: polarity and subjectivity (TextBlob)
- **Syntactic ratios**: proportion of verbs, adjectives, adverbs
- **Title heuristics**: uppercase ratio, title-text similarity

Filtering done via:
- T-test (statistical significance)
- Pearson correlation
- Mutual Information

## 4. Text Representation (Embeddings)

| Embedding | Implementation |
|-----------|----------------|
| **TF-IDF** | scikit-learn |
| **Word2Vec** (CBOW, 300d) | gensim |
| **GloVe** (6B, 100d) | pretrained |
| **BERT** (`bert-base-uncased`) | HuggingFace Transformers |

## 5. Models and Notebooks

Each model was tested **with and without** features. In classical models, both variants are included in the same notebook.

| Model | Embedding | Without Features | With Features | 
|-------|-----------|------------------|---------------|
| **Logistic Regression** | TF-IDF, W2V-avg, GloVe-avg, BERT-avg | ✔ | ✔ |
| **Decision Tree** | same | ✔ | ✔ |
| **SVC** | same | ✔ | ✔ | 
| **BiLSTM (RNN)** | **Word2Vec** | `w2v_bilstm_16.ipynb`, `w2v_bilstm_32.ipynb` | `w2v_bilstm_16_features.ipynb`, `w2v_bilstm_32_features.ipynb` | 
|  | **GloVe** | `glove_bilstm_16.ipynb`, `glove_bilstm_32.ipynb` | `glove_bilstm_16_features.ipynb`, `glove_bilstm_32_features.ipynb` | 
|  | **BERT** | `bert_bilstm_16.ipynb`, `bert_bilstm_32.ipynb` | `bert_bilstm_16_features.ipynb`, `bert_bilstm_32_features.ipynb` | 

> All BiLSTM models were GPU-accelerated (CUDA + cuDNN), with dropout=0.3 and early stopping based on validation F1-score.

---

## 6. Results and Final Model

- The best model was **BERT + BiLSTM + features**, batch size 32, reaching ~96% F1-score with consistent performance across 5 runs.
- The exported model (`bert_lstm_features_32_model_extensie.pt`) and its scaler (`scaler_bert.pkl`) are included in the `Extension/` folder.


