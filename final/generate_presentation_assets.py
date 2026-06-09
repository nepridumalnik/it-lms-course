from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split

from app.features import TRAIN_COLUMNS, add_engineered_features


TARGET_COLUMN = "scaled-sound-pressure"
RANDOM_SEED = 42
TEST_SIZE = 0.2
VALID_SIZE = 0.25
ASSETS_DIR = Path("presentation_assets")
BG_COLOR = "#0f172a"
TEXT_COLOR = "#e5e7eb"
MUTED_COLOR = "#94a3b8"
ACCENT_COLOR = "#38bdf8"
SECONDARY_COLOR = "#f59e0b"
FEATURE_LABELS = {
    "frequency": "частота",
    "attack-angle": "угол атаки",
    "chord-length": "длина хорды",
    "free-stream-velocity": "скорость потока",
    "suction-side-displacement-thickness": "толщина вытеснения",
}


def save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight", transparent=True)
    plt.close()


def style_axis(ax, title: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_facecolor("none")
    ax.set_title(title, fontsize=14, pad=12, color=TEXT_COLOR)
    if xlabel:
        ax.set_xlabel(xlabel, color=TEXT_COLOR)
    if ylabel:
        ax.set_ylabel(ylabel, color=TEXT_COLOR)
    ax.tick_params(colors=TEXT_COLOR)
    for spine in ax.spines.values():
        spine.set_color(MUTED_COLOR)
        spine.set_alpha(0.5)
    ax.grid(axis="y", color=MUTED_COLOR, alpha=0.2)


def make_target_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_alpha(0)
    ax.hist(df[TARGET_COLUMN], bins=30, color=ACCENT_COLOR, edgecolor=BG_COLOR)
    style_axis(
        ax,
        "Распределение целевой переменной",
        "scaled-sound-pressure",
        "Количество наблюдений",
    )
    save_figure(ASSETS_DIR / "target_distribution.png")


def make_feature_engineering_chart() -> None:
    results = pd.DataFrame(
        [
            {"features": "Базовые", "RMSE": 1.807526, "R2": 0.934786},
            {"features": "С новыми признаками", "RMSE": 1.686461, "R2": 0.943229},
        ]
    )

    fig, ax = plt.subplots(figsize=(7, 4.3))
    fig.patch.set_alpha(0)
    bars = ax.bar(results["features"], results["RMSE"], color=[MUTED_COLOR, ACCENT_COLOR])
    style_axis(ax, "Влияние создания признаков", ylabel="RMSE")
    ax.set_ylim(0, 2.1)
    for bar, value in zip(bars, results["RMSE"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.04,
            f"{value:.2f}",
            ha="center",
            fontsize=12,
            color=TEXT_COLOR,
        )
    save_figure(ASSETS_DIR / "feature_engineering_rmse.png")


def make_model_comparison_chart() -> None:
    results = pd.DataFrame(
        [
            {"model": "RandomForest\nPipeline", "valid_RMSE": 1.973917},
            {"model": "DecisionTree", "valid_RMSE": 3.040305},
            {"model": "KNN", "valid_RMSE": 6.188416},
        ]
    )

    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    fig.patch.set_alpha(0)
    bars = ax.bar(results["model"], results["valid_RMSE"], color=ACCENT_COLOR)
    style_axis(ax, "Сравнение моделей на validation", ylabel="RMSE")
    ax.set_ylim(0, 7)
    for bar, value in zip(bars, results["valid_RMSE"]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.12,
            f"{value:.2f}",
            ha="center",
            fontsize=12,
            color=TEXT_COLOR,
        )
    save_figure(ASSETS_DIR / "model_comparison.png")


def make_prediction_quality_and_importance(df: pd.DataFrame) -> None:
    model = joblib.load("data/random_forest_model.pkl")
    features = df[TRAIN_COLUMNS]
    target = df[TARGET_COLUMN]

    train_full, test, y_train_full, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    train, valid, y_train, y_valid = train_test_split(
        train_full,
        y_train_full,
        test_size=VALID_SIZE,
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    _ = train, valid, y_train, y_valid

    prediction = model.predict(test)
    rmse = root_mean_squared_error(y_test, prediction)
    r2 = r2_score(y_test, prediction)

    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_alpha(0)
    ax.scatter(y_test, prediction, s=26, color=ACCENT_COLOR, alpha=0.85, edgecolor="none")
    min_value = min(y_test.min(), prediction.min())
    max_value = max(y_test.max(), prediction.max())
    ax.plot([min_value, max_value], [min_value, max_value], color=SECONDARY_COLOR, linewidth=2)
    style_axis(
        ax,
        f"Факт vs прогноз на test\nRMSE={rmse:.2f}, R2={r2:.3f}",
        "Фактическое значение",
        "Прогноз",
    )
    save_figure(ASSETS_DIR / "prediction_quality.png")

    importance = permutation_importance(
        model,
        test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_SEED,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )
    importance_table = (
        pd.DataFrame(
            {
                "feature": [FEATURE_LABELS[column] for column in TRAIN_COLUMNS],
                "importance": importance.importances_mean,
            }
        )
        .sort_values("importance", ascending=True)
        .tail(5)
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    fig.patch.set_alpha(0)
    ax.barh(importance_table["feature"], importance_table["importance"], color=ACCENT_COLOR)
    style_axis(ax, "Перестановочная важность признаков", "Рост RMSE при перемешивании")
    save_figure(ASSETS_DIR / "permutation_importance.png")


def main() -> None:
    df = pd.read_csv("data.csv")
    make_target_distribution(df)
    make_feature_engineering_chart()
    make_model_comparison_chart()
    make_prediction_quality_and_importance(df)


if __name__ == "__main__":
    main()
