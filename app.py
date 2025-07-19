from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import spacy
import os

app = Flask(__name__)

# 🚀 Load model and tokenizer
print("🚀 Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("mrm8488/bert-tiny-finetuned-sms-spam-detection")
model = AutoModelForSequenceClassification.from_pretrained("mrm8488/bert-tiny-finetuned-sms-spam-detection")
nlp = spacy.load("en_core_web_sm")
print("✅ Model and tokenizer loaded.")

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "SMS Spam Classifier backend is running."})

@app.route("/predict", methods=["POST"])
def predict():
    print("📥 Received a request at /predict")
    
    try:
        data = request.get_json()
        print(f"📝 Message received: {data}")

        message = data.get("message", "")
        if not message:
            return jsonify({"error": "No message provided."}), 400

        # Clean and tokenize
        doc = nlp(message)
        cleaned = " ".join([token.text.lower() for token in doc if not token.is_punct])
        print(f"🧹 Cleaned message: {cleaned}")

        # Tokenize for BERT
        inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs).item()
        confidence = float(probs[0][predicted_class]) * 100
        label = "Spam" if predicted_class == 1 else "Not Spam"

        print(f"📊 Prediction: {label}, Confidence: {confidence:.2f}%")

        return jsonify({
            "label": label,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        print(f"❌ Error inside /predict: {e}")
        return jsonify({"error": str(e)}), 500

# 🌐 Required for Render deployment
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
