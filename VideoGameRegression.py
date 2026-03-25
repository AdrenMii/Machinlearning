import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from sklearn.linear_model import LinearRegression

# Cargar dataset
data = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')

# Convertir User_Score a numero y limpiar datos
data["User_Score"] = pd.to_numeric(data["User_Score"], errors="coerce")
data = data.dropna()

# --- CAMBIO 1: Entrenar con las 4 variables que pide tu HTML ---
features = ["Critic_Score", "User_Score", "Critic_Count", "User_Count"]
X = data[features]
y = data["Global_Sales"]

# Modelo entrenado con 4 columnas
model = LinearRegression()
model.fit(X, y)

# --- CAMBIO 2: La función ahora recibe 4 argumentos ---
def predictSales(critic, user, critic_count, user_count):
    # Usamos un DataFrame con nombres de columnas para evitar el UserWarning
    input_data = pd.DataFrame(
        [[critic, user, critic_count, user_count]], 
        columns=features
    )
    result = model.predict(input_data)[0]
    return round(float(result), 2)

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import numpy as np
from sklearn.linear_model import LinearRegression

# Cargar dataset
data = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')

# Convertir User_Score a numero y limpiar datos
data["User_Score"] = pd.to_numeric(data["User_Score"], errors="coerce")
data = data.dropna()

# --- CAMBIO 1: Entrenar con las 4 variables que pide tu HTML ---
features = ["Critic_Score", "User_Score", "Critic_Count", "User_Count"]
X = data[features]
y = data["Global_Sales"]

# Modelo entrenado con 4 columnas
model = LinearRegression()
model.fit(X, y)

# --- CAMBIO 2: La función ahora recibe 4 argumentos ---
def predictSales(critic, user, critic_count, user_count):
    # Usamos un DataFrame con nombres de columnas para evitar el UserWarning
    input_data = pd.DataFrame(
        [[critic, user, critic_count, user_count]], 
        columns=features
    )
    result = model.predict(input_data)[0]
    return round(float(result), 2)

def getGraphBase64():
    # Tamaño optimizado para la columna del HTML
    fig, ax = plt.subplots(figsize=(10, 6))

    # Puntos de datos: Muy sutiles para que la tendencia mande
    ax.scatter(data["Critic_Score"], data["Global_Sales"], 
               color="#6610f2", alpha=0.15, s=8, zorder=2, label="Datos Históricos")

    # Línea de tendencia: Gruesa y al frente
    x_line = np.linspace(data["Critic_Score"].min(), data["Critic_Score"].max(), 100)
    test_data = pd.DataFrame({
        "Critic_Score": x_line,
        "User_Score": data["User_Score"].mean(),
        "Critic_Count": data["Critic_Count"].mean(),
        "User_Count": data["User_Count"].mean()
    })
    y_line = model.predict(test_data)

    ax.plot(x_line, y_line, color="#0d6efd", linewidth=3, zorder=5, label="Tendencia Predictiva")

    # Ajustes estéticos (Sin bordes innecesarios)
    ax.set_ylim(-0.5, 12) # Recorte para evitar que los outliers aplasten la línea
    ax.set_xlabel("Critic Score (Puntuación de la Crítica)")
    ax.set_ylabel("Ventas Globales (Millones)")
    ax.legend(loc="upper left", frameon=False)
    ax.grid(True, alpha=0.1, linestyle='--')
    
    # Quitar marcos superior y derecho para estética moderna
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)

    buf = io.BytesIO()
    plt.tight_layout()
    # dpi=120 para que se vea nítido pero no pesado
    plt.savefig(buf, format="png", dpi=120, bbox_inches='tight', transparent=True)
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img