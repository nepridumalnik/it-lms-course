import flask
from pathlib import Path

from .inferer import Inferer


class AppController:
    def __init__(self):
        self.inferer = Inferer("data/random_forest_model.pkl")
        static_path = Path(__file__).resolve().parent.parent / "static"
        self.app = flask.Flask("final task", static_folder=static_path)

        self.app.add_url_rule(
            "/api/inference",
            endpoint="inference",
            view_func=self.inferer.inference,
            methods=["POST"],
        )

        self.app.add_url_rule(
            "/", endpoint="test", view_func=self.test, methods=["GET"]
        )

    def test(self):
        return self.app.send_static_file("index.html")

    def run(self, host="127.0.0.1", port=8080) -> None:
        self.app.run(host=host, port=port)
