"""
Serving-cost comparison of the three candidate models.

All three models scored about the same on FD001 (PR-AUC around 0.94), so the
choice of model for deployment depends on what each costs the API on every
request: file size, time to predict one engine, and time to explain one
prediction with SHAP (the dashboard explains every prediction).

Run from the project root:
    python -m experiments.model_cost_benchmark
"""

import json
import pickle
import time

import pandas as pd
import shap

from src.config import PROJECT_ROOT, load_params
from src.features.feature_logic import feature_names
from src.models.train import MODEL_TYPES, build_model

REPORT_PATH = PROJECT_ROOT / "reports" / "model_cost_benchmark.json"


def average_ms(func, repeats: int) -> float:
    """Run func several times and return the average time in milliseconds."""
    func()  # warm-up run, not timed
    start = time.perf_counter()
    for _ in range(repeats):
        func()
    return (time.perf_counter() - start) / repeats * 1000


def main():
    params = load_params()
    tcfg = params["train"]
    features = feature_names(params["features"]["sensors"], int(params["features"]["window"]))
    data_dir = PROJECT_ROOT / tcfg["input_dir"]

    train = pd.read_parquet(data_dir / "train_FD001.parquet")
    X, y = train[features].astype("float64"), train["label"]
    test = pd.read_parquet(data_dir / "test_FD001.parquet")
    one_engine = test[test["unit"] == 1][features].astype("float64")  # one operator upload
    last_cycle = one_engine.tail(1)  # the row we explain
    pos_weight = (y == 0).sum() / (y == 1).sum()

    results = []
    for model_type in MODEL_TYPES:
        model = build_model(model_type, tcfg[model_type], pos_weight, int(tcfg["random_state"]))
        model.fit(X, y)

        if model_type == "logistic_regression":
            # Linear model: explain on the scaled features it actually sees
            background = shap.maskers.Independent(model[0].transform(X.sample(100, random_state=0)))
            explainer = shap.LinearExplainer(model[-1], background)
            explain_input = model[0].transform(last_cycle)
        else:
            explainer = shap.TreeExplainer(model)
            explain_input = last_cycle

        results.append(
            {
                "model": model_type,
                "size_kb": round(len(pickle.dumps(model)) / 1024),
                "predict_one_engine_ms": round(
                    average_ms(lambda: model.predict_proba(one_engine), 20), 1
                ),
                "shap_one_row_ms": round(
                    average_ms(lambda: explainer.shap_values(explain_input), 10), 1
                ),
            }
        )
        print(results[-1])

    REPORT_PATH.write_text(json.dumps(results, indent=2))
    print("\n" + pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
