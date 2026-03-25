import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score

# Carga y entrenamiento (Tal cual lo tenías)
data = pd.read_csv('dataset_regresion_logistica.csv')
X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)

# --- FUNCIÓN DE LA GRÁFICA (LA QUE DA LOS DATOS REALES) ---
def get_logistic_graph():
    # Usamos la edad para el eje X
    x_range = np.linspace(data['edad'].min(), data['edad'].max(), 300).reshape(-1, 1)
    
    # Creamos datos ficticios manteniendo las otras variables en su promedio
    dummy_data = pd.DataFrame({
        'edad': x_range.flatten(),
        'ingreso_mensual': [data['ingreso_mensual'].mean()] * 300,
        'visitas_web_mes': [data['visitas_web_mes'].mean()] * 300,
        'tiempo_sitio_min': [data['tiempo_sitio_min'].mean()] * 300,
        'compras_previas': [data['compras_previas'].mean()] * 300,
        'descuento_usado': [data['descuento_usado'].mean()] * 300
    })

    # Escalar y predecir probabilidad
    dummy_scaled = scaler.transform(dummy_data)
    probabilidades = logistic_model.predict_proba(dummy_scaled)[:, 1]

    # Dibujar
    plt.figure(figsize=(10, 6))
    plt.scatter(data['edad'], data['target'], color='#6610f2', alpha=0.3, label='Datos Reales')
    plt.plot(x_range, probabilidades, color='#6610f2', linewidth=3, label='Curva Sigmoide')
    plt.axhline(0.5, color='red', linestyle='--', alpha=0.6)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Convertir a Base64
    img = io.BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight', transparent=True)
    img.seek(0)
    plt.close()
    return base64.b64encode(img.getvalue()).decode()

# Tus otras funciones de utilidad
def predict_purchase(edad, ingreso_mensual, visitas_web_mes, tiempo_sitio_min, compras_previas, descuento_usado):
    input_data = pd.DataFrame([[edad, ingreso_mensual, visitas_web_mes, tiempo_sitio_min, compras_previas, descuento_usado]],
                               columns=['edad', 'ingreso_mensual', 'visitas_web_mes', 'tiempo_sitio_min', 'compras_previas', 'descuento_usado'])
    input_scaled = scaler.transform(input_data)
    prediction = logistic_model.predict(input_scaled)[0]
    probability = logistic_model.predict_proba(input_scaled)[0][1]
    return int(prediction), round(float(probability) * 100, 2)

def get_model_accuracy():
    y_pred = logistic_model.predict(X_test_scaled)
    return round(accuracy_score(y_test, y_pred) * 100, 2)

def get_confusion_matrix():
    y_pred = logistic_model.predict(X_test_scaled)
    return confusion_matrix(y_test, y_pred).tolist()

