import joblib

model = joblib.load("prompt_detector_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

print("\n🔐 Prompt Injection Detector (type 'exit' to quit)\n")

while True:
    prompt = input("Enter prompt: ")
    if prompt.lower() in ["exit", "quit"]:
        break

    prompt_vec = vectorizer.transform([prompt])
    prediction = model.predict(prompt_vec)[0]

    print("Result:", "🚨 Injection" if prediction == 1 else "✅ Safe")
    print("-" * 50)
