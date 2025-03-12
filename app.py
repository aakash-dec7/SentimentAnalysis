from src.sentana.logger import logger
from flask import Flask, render_template, request
from src.sentana.prediction.prediction import Prediction
from src.sentana.config.configuration import ConfigurationManager


app = Flask(__name__)

# Load Prediction Model ONCE
prediction_model = Prediction(config=ConfigurationManager().get_prediction_config())


@app.route("/", methods=["GET"])
def home_page():
    return render_template("index.html", sentiment=None)


@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        text_input = request.form.get("text_input", "").strip()
        if not text_input:
            return render_template("index.html", sentiment=None)

        sentiment = prediction_model.predict(text_input).lower().strip()

        return render_template("index.html", sentiment=sentiment)

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        return render_template(
            "index.html", sentiment="An error occurred. Please try again later."
        )


if __name__ == "__main__":
    # app.run(host="0.0.0.0", port=3000, debug=True)
    app.run(host="0.0.0.0", port=8080)
