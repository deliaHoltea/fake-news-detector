from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel, pipeline
import numpy as np
import re
import joblib
from nltk import wordpunct_tokenize, pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords, wordnet
from textblob import TextBlob
import textstat
import nltk
from torch.nn import LSTM


app = Flask(__name__)
CORS(app)  

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download("stopwords")
nltk.download("wordnet")

# Preprocessing 
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

# Load model and scaler
torch.serialization.add_safe_globals({'LSTM': LSTM})
lstm_model, linear_model = torch.load("bert_lstm_features_32_model_extensie.pt", map_location="cpu", weights_only=False)
lstm_model.eval()
linear_model.eval()
scaler = joblib.load("scaler_bert.pkl")

# Helper functions
def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'): return wordnet.ADJ
    elif treebank_tag.startswith('V'): return wordnet.VERB
    elif treebank_tag.startswith('N'): return wordnet.NOUN
    elif treebank_tag.startswith('R'): return wordnet.ADV
    return wordnet.NOUN

def preprocess(text):
    tokens = wordpunct_tokenize(text)
    tokens = [re.sub(r"[^a-zA-Z0-9]", "", t.lower()) for t in tokens]
    tokens = [t for t in tokens if len(t) > 2 and t not in stop_words]
    pos_tags = pos_tag(tokens)
    return [lemmatizer.lemmatize(w, get_wordnet_pos(tag)) for w, tag in pos_tags]

def extract_features(title, text):
    title_clean = preprocess(title)
    text_clean = preprocess(text)
    text_blob = TextBlob(" ".join(text_clean))

    def pos_ratio(tokens, pos_prefix):
        tags = pos_tag(tokens)
        return sum(1 for _, t in tags if t.startswith(pos_prefix)) / len(tokens) if tokens else 0

    try:
        features = [
            textstat.gunning_fog(text),
            textstat.smog_index(text),
            text_blob.sentiment.subjectivity,
            len(set(title_clean).intersection(set(text_clean))) / len(set(title_clean)) if title_clean else 0,
            textstat.avg_sentence_length(text),
            pos_ratio(title_clean, 'RB'),
            pos_ratio(title_clean, 'NN'),
            len(title_clean),
            textstat.syllable_count(text) / len(text_clean) if text_clean else 0,
            sum(1 for c in title if c.isupper()),
            (sum(1 for c in title + text if c.isupper()) / (len(title + text) + 1)),
            pos_ratio(text_clean, 'JJ'),
            pos_ratio(text_clean, 'RB'),
            pos_ratio(text_clean, 'VB'),
            pos_ratio(text_clean, 'NN'),
            len([s for s in text.split('.') if len(s.split()) <= 5]) / max(1, len(text.split('.'))),
            len([s for s in text.split('.') if len(s.split()) >= 15]) / max(1, len(text.split('.')))
        ]
    except:
        features = [0.0] * 17
    return np.array(features).reshape(1, -1), title_clean + text_clean

def get_bert_embedding(tokens):
    text = " ".join(tokens)
    encoded = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    with torch.no_grad():
        output = bert_model(**encoded)
    return output.last_hidden_state, encoded["attention_mask"]

def predict(title, text):
    features, tokens = extract_features(title, text)
    scaled = scaler.transform(features)
    feats_tensor = torch.tensor(scaled, dtype=torch.float32)

    embedding, mask = get_bert_embedding(tokens)

    with torch.no_grad():
        out, _ = lstm_model(embedding)
        lens = mask.sum(1) - 1
        last = out[range(out.size(0)), lens.long()]
        concat = torch.cat((last, feats_tensor), dim=1)
        logits = linear_model(concat)
        probs = F.softmax(logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        return {
            "label": "FAKE" if pred == 1 else "REAL",
            "score_fake": round(probs[0][1].item(), 4),
            "score_real": round(probs[0][0].item(), 4)
        }

@app.route("/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    title = data.get("title", "")
    text = data.get("text", "")
    try:
        result = predict(title, text)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

fact_rewriter = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=0  # -1 pentru CPU, 0 pentru GPU
)

def extract_first_sentence(text: str) -> str:
    sentences = re.split(r'[.!?]', text)
    for s in sentences:
        s = s.strip()
        if len(s.split()) > 4 and s[0].isupper():
            return s
    return text.strip()

@app.route("/generate-hint", methods=["POST"])
def generate_hint():
    data = request.json
    full_text = f"{data.get('title', '')}. {data.get('text', '')}"
    sentence = extract_first_sentence(full_text)
    prompt = f"Rewrite this as a factual claim: {sentence}"

    output = fact_rewriter(prompt, max_length=40, do_sample=False)[0]['generated_text']
    return jsonify({ "hint": output.strip() })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
