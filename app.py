import gradio as gr
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")

# ── Train model at startup ─────────────────────────────────────────────────────
def train_model():
    data = pd.read_csv("water_potability.csv")

    # Random imputation (same as notebook)
    for col in ["ph", "Sulfate", "Trihalomethanes"]:
        random_sample = data[col].dropna().sample(data[col].isnull().sum(), random_state=0)
        random_sample.index = data[data[col].isnull()].index
        data.loc[data[col].isnull(), col] = random_sample
        data.rename(columns={col: col + "_random"}, inplace=True)

    # Drop turbid_class equivalent — not needed here
    # Final feature set
    features = [
        "Hardness", "Solids", "Chloramines", "Conductivity",
        "Organic_carbon", "Turbidity",
        "ph_random", "Sulfate_random", "Trihalomethanes_random"
    ]

    X = data[features].values
    y = data["Potability"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, train_size=0.85, random_state=42
    )

    model = RandomForestClassifier(n_estimators=500, oob_score=True, random_state=100)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    print(f"Model ready — test accuracy: {acc:.2%}")
    return model, scaler, acc


model, scaler, accuracy = train_model()

FEATURE_RANGES = {
    "pH":               (0.0,  14.0,  7.08),
    "Hardness (mg/L)":  (47.0, 323.0, 196.0),
    "Solids (ppm)":     (320.0, 61227.0, 22014.0),
    "Chloramines (ppm)":(0.35, 13.13, 7.12),
    "Sulfate (mg/L)":   (129.0, 481.0, 333.0),
    "Conductivity (μS/cm)": (181.0, 753.0, 426.0),
    "Organic Carbon (ppm)": (2.2,  28.3,  14.28),
    "Trihalomethanes (μg/L)": (0.74, 124.0, 66.4),
    "Turbidity (NTU)":  (1.45, 6.99,  3.97),
}

LABELS = list(FEATURE_RANGES.keys())


def predict(*inputs):
    arr = np.array(inputs, dtype=float).reshape(1, -1)
    arr_scaled = scaler.transform(arr)
    pred = model.predict(arr_scaled)[0]
    proba = model.predict_proba(arr_scaled)[0]

    safe_pct   = proba[1] * 100
    unsafe_pct = proba[0] * 100

    if pred == 1:
        verdict = "✅ POTABLE — Safe to drink"
        color   = "green"
    else:
        verdict = "❌ NOT POTABLE — Unsafe to drink"
        color   = "red"

    result = (
        f"## {verdict}\n\n"
        f"| | |\n|---|---|\n"
        f"| 💧 Safe probability   | **{safe_pct:.1f}%** |\n"
        f"| ☠️ Unsafe probability | **{unsafe_pct:.1f}%** |\n\n"
        f"*Model: Random Forest (500 trees) — Test accuracy: {accuracy:.2%}*"
    )
    return result


# ── Build UI ──────────────────────────────────────────────────────────────────
sliders = []
for label, (lo, hi, default) in FEATURE_RANGES.items():
    sliders.append(
        gr.Slider(minimum=lo, maximum=hi, value=default, step=round((hi - lo) / 200, 3), label=label)
    )

demo = gr.Interface(
    fn=predict,
    inputs=sliders,
    outputs=gr.Markdown(),
    title="💧 Water Quality Classifier",
    description=(
        "Enter the physicochemical properties of a water sample to predict "
        "whether it is **safe for human consumption**.\n\n"
        "Adjust the sliders to match your water sample's measurements."
    ),
    examples=[
        [7.08, 196.0, 22014.0, 7.12, 333.0, 426.0, 14.28, 66.4, 3.97],   # average sample
        [6.50, 150.0, 15000.0, 5.00, 250.0, 380.0, 10.0,  45.0, 2.5],    # low-concern sample
        [3.50, 310.0, 55000.0, 12.0, 470.0, 720.0, 26.0, 118.0, 6.8],    # high-concern sample
    ],
    theme=gr.themes.Soft(primary_hue="blue"),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
