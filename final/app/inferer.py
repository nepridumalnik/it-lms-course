import flask
import joblib
import pandas as pd

from .features import TRAIN_COLUMNS


class Inferer:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def inference(self) -> flask.json:
        data = flask.request.get_json()

        try:
            features = {column: float(data[column]) for column in TRAIN_COLUMNS}
        except (KeyError, TypeError, ValueError):
            return flask.jsonify({"error": "Missing or invalid parameters"}), 400

        if any(value < 0 for value in features.values()):
            return flask.jsonify({"error": "Parameters cannot be negative"}), 400

        prediction = self.model.predict(pd.DataFrame([features], columns=TRAIN_COLUMNS))

        return flask.jsonify({"prediction": prediction[0].tolist()})
