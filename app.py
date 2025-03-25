from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("model.pkl")

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    if request.method == "POST":
        sms = request.form["sms"]
        prediction = model.predict([sms])[0]
        result = "Ham (Not Spam)" if prediction == 1 else "Spam"
    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
