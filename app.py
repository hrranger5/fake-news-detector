from flask import Flask, render_template, request
import joblib
import os

# Load model and vectorizer (use absolute paths for safety)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "Python files", "fake_news_model.pkl"))
vectorizer = joblib.load(os.path.join(BASE_DIR, "Python files", "tfidf_vectorizer.pkl"))


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        input_text = request.form.get("news", "").strip()

        if not input_text:
            return render_template("index.html", prediction="⚠️ No input provided", confidence=0, input_text="")

        transformed_input = vectorizer.transform([input_text])

        # Predict class and confidence
        prediction = model.predict(transformed_input)[0]
        try:
            decision = model.decision_function(transformed_input)
            confidence_score = abs(decision[0])
            confidence = min(round((confidence_score / 5) * 100, 2), 100.0)
        except Exception:
            confidence = "N/A"

        # Decide label based on confidence
        if confidence != "N/A" and confidence < 60:
            result = "Uncertain ⚠️ (Low confidence)"
        elif prediction == 1:
            result = "REAL ✅"
        else:
            result = "FAKE 🔴"

        # Highlight top 5 keywords
        feature_names = vectorizer.get_feature_names_out()
        input_vector = transformed_input.toarray()[0]
        top_indices = input_vector.argsort()[-5:][::-1]

        highlighted_text = input_text
        for idx in top_indices:
            word = feature_names[idx]
            if word.lower() in highlighted_text.lower():
                highlighted_text = highlighted_text.replace(
                    word, f"<strong style='color:#ffd700'>{word}</strong>"
                )

        return render_template(
            "index.html",
            prediction=result,
            confidence=confidence,
            input_text=highlighted_text
        )

if __name__ == "__main__":
    app.run(debug=True)
