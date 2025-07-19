from flask import Flask, request, jsonify
import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

app = Flask(__name__)

print("🚀 Loading model and tokenizer...")
nlp = spacy.load("en_core_web_sm")
model_name = "mrm8488/bert-tiny-finetuned-sms-spam-detection"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
print("✅ Model and tokenizer loaded.")

@app.route('/predict', methods=['POST'])
def predict():
    print("📥 Received a request at /predict")
    
    try:
        data = request.get_json()
        print(f"📝 Message received: {data}")

        message = data.get("message", "")
        doc = nlp(message)
        cleaned = " ".join([token.text.lower() for token in doc if not token.is_punct])
        print(f"🧹 Cleaned message: {cleaned}")

        inputs = tokenizer(cleaned, return_tensors="pt", truncation=True, padding=True)
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        predicted_class = torch.argmax(probs).item()
        confidence = float(probs[0][predicted_class]) * 100
        label = "Spam" if predicted_class == 1 else "Not Spam"

        print(f"📊 Prediction: {label}, Confidence: {confidence:.2f}%")

        return jsonify({"label": label, "confidence": round(confidence, 2)})

    except Exception as e:
        print(f"❌ Error inside /predict: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("✅ Flask server starting on http://127.0.0.1:5000/")
    app.run(debug=True)
