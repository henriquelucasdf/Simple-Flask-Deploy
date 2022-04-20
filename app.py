import joblib
import numpy as np

from flask import Flask, request
from flask import render_template

from preprocess_src import CustomTextPreprocessor, format_prediction

# creates the app
app = Flask(__name__)

# Home page


@app.route("/")
def index():
    """
    It returns the rendered template of the index.html file

    Returns:
      The index.html file is being returned.
    """

    return render_template("index.html", result=None, prob=None)

# prediction page


@app.route("/estimate", methods=["POST"])
def estimate():
    """
    It loads the pipeline, predicts the class of the input text, and returns the class and the
    probability of the prediction

    Returns:
      The result of the prediction and the probability of the prediction.
    """

    # gets the input text as a list (the 'form' returns a string)
    input_text = [request.form['input_text']]

    # load the pipeline
    try:
        pipeline = joblib.load('models/best_pipeline_16-04-2022-14h16.pkl')
    except Exception as e:
        raise e

    # predicting probabilities:
    yhat_prob = pipeline.predict_proba(
        X=input_text)[0]  # the probabilities array
    probability = f'{100*max(yhat_prob):.2f}%'

    # classes
    yhat = np.argmax(yhat_prob)  # the index of the maximum probability
    predicted_class = format_prediction(yhat)  # index to class name

    return render_template("index.html", result=predicted_class, prob_=probability)


if __name__ == "__main__":
    app.run(host='localhost', port=3000, debug=True)
