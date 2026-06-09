import numpy as np


TRAIN_COLUMNS = [
    "frequency",
    "attack-angle",
    "chord-length",
    "free-stream-velocity",
    "suction-side-displacement-thickness",
]


def add_engineered_features(data):
    result = data.copy()
    result["frequency_velocity_ratio"] = (
        result["frequency"] / (result["free-stream-velocity"] + 1e-9)
    )
    result["angle_chord_product"] = result["attack-angle"] * result["chord-length"]
    result["velocity_chord_product"] = (
        result["free-stream-velocity"] * result["chord-length"]
    )
    result["log_frequency"] = np.log1p(result["frequency"])
    return result
