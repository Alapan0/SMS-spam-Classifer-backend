import requests
import json

print("➡️ Starting test.py...")
message = "Congratulations! You've won a free iPhone. Click here to claim."

print("📤 Sending message to Flask server...")
try:
    response = requests.post(
        "http://127.0.0.1:5000/predict",
        json={"message": message},
        timeout=20
    )
    print("📥 Server responded with:", response.text)
    print("✅ Final JSON Output:", response.json())
except requests.exceptions.Timeout:
    print("⏱️ Timeout error: Flask server took too long to respond.")
except Exception as e:
    print(f"❌ Error: {e}")

input("🔚 Press Enter to exit...")
