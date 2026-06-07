"""Export trained model + scaler to model.json for browser inference."""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def export_tree(tree):
    t = tree.tree_
    return {
        "feature": t.feature.astype(int).tolist(),
        "threshold": [float(x) for x in t.threshold],
        "children_left": t.children_left.astype(int).tolist(),
        "children_right": t.children_right.astype(int).tolist(),
        "value": t.value.reshape(-1, t.value.shape[1], t.value.shape[2]).tolist(),
    }


def main():
    data = pd.read_csv("water_potability.csv")

    for col in ["ph", "Sulfate", "Trihalomethanes"]:
        random_sample = data[col].dropna().sample(data[col].isnull().sum(), random_state=0)
        random_sample.index = data[data[col].isnull()].index
        data.loc[data[col].isnull(), col] = random_sample
        data.rename(columns={col: col + "_random"}, inplace=True)

    features = [
        "ph_random", "Hardness", "Solids", "Chloramines", "Sulfate_random",
        "Conductivity", "Organic_carbon", "Trihalomethanes_random", "Turbidity",
    ]

    X = data[features].values
    y = data["Potability"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, train_size=0.85, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=100, max_depth=14, random_state=100, n_jobs=-1
    )
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Test accuracy: {acc:.2%}")

    payload = {
        "model": "random_forest",
        "accuracy": round(acc * 100, 2),
        "n_estimators": len(model.estimators_),
        "scaler": {
            "mean": scaler.mean_.tolist(),
            "scale": scaler.scale_.tolist(),
        },
        "trees": [export_tree(est) for est in model.estimators_],
    }

    with open("model.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, separators=(",", ":"))

    size_kb = len(json.dumps(payload)) / 1024
    print(f"Saved model.json ({size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
