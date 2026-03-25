import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# Dataset interno (para que funcione sin CSV)
data = pd.DataFrame({
    "edad": [25, 30, 45, 35, 22, 50, 48, 33, 27, 40],
    "ingreso_mensual": [2000, 3000, 5000, 4000, 1500, 6000, 5800, 3200, 2700, 4500],
    "visitas_web_mes": [5, 8, 2, 4, 10, 1, 2, 6, 7, 3],
    "tiempo_sitio_min": [10, 15, 5, 7, 20, 3, 4, 12, 14, 6],
    "compras_previas": [1, 2, 5, 3, 0, 6, 5, 2, 1, 4],
    "descuento_usado": [0, 1, 1, 0, 0, 1, 1, 0, 0, 1],
    "target": [0, 1, 1, 0, 0, 1, 1, 0, 0, 1]
})

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

def predict_purchase(edad, ingreso_mensual, visitas_web_mes, tiempo_sitio_min, compras_previas, descuento_usado):
    valores = [[edad, ingreso_mensual, visitas_web_mes, tiempo_sitio_min, compras_previas, descuento_usado]]
    pred = model.predict(valores)[0]
    prob = model.predict_proba(valores)[0][1]
    return int(pred), round(prob * 100, 2)

def get_accuracy():
    return round(accuracy * 100, 2)

def get_confusion():
    return confusion_matrix(y_test, y_pred).tolist()