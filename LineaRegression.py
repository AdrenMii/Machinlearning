import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.linear_model import LinearRegression

data = {
    "Study Hours": [10, 15, 12, 8, 14, 5, 16, 7, 11, 13, 9, 4, 18, 3, 17, 6, 14, 2, 20, 1],
    "Final Grade": [3.8, 4.2, 3.6, 3, 4.5, 2.5, 4.8, 2.8, 3.7, 4, 3.2, 2.2, 5, 1.8, 4.9, 2.7, 4.4, 1.5, 5, 1]
}

df = pd.DataFrame(data)

X = df[["Study Hours"]]
y = df["Final Grade"]

model = LinearRegression()
model.fit(X, y)

def calculateGrade(hours):
    result = model.predict([[hours]])[0]
    return round(float(result), 2)

def getGraphBase64():
    fig, ax = plt.subplots(figsize=(8, 5))

    # Puntos de datos
    ax.scatter(df["Study Hours"], df["Final Grade"],
               color="#6610f2", alpha=0.8, s=60, zorder=3, label="Students")

    # Línea de regresión
    import numpy as np
    x_line = np.linspace(0, 21, 100)
    y_line = model.predict([[x] for x in x_line])
    ax.plot(x_line, y_line, color="#0d6efd", linewidth=2.5, label="Regression line")

    # Estilo
    ax.set_xlabel("Study Hours (X)", fontsize=12, color="#374151")
    ax.set_ylabel("Final Grade (Y)", fontsize=12, color="#374151")
    ax.set_title("Linear Regression — Study Hours vs Final Grade", fontsize=13, color="#1a1a1a", pad=15)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.set_facecolor("#fafafa")
    fig.patch.set_facecolor("#ffffff")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Convertir a base64
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_base64