import flask
import joblib


class Inferer:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def inference(self) -> flask.json:
        data = flask.request.get_json()

        try:
            features = [
                float(data["frequency"]),
                float(data["attack-angle"]),
                float(data["chord-length"]),
                float(data["free-stream-velocity"]),
                float(data["suction-side-displacement-thickness"]),
            ]
        except (KeyError, TypeError, ValueError):
            return flask.jsonify({"error": "Missing or invalid parameters"}), 400

        if any(value < 0 for value in features):
            return flask.jsonify({"error": "Parameters cannot be negative"}), 400

        prediction = self.model.predict([features])

        return flask.jsonify({"prediction": prediction[0].tolist()})
