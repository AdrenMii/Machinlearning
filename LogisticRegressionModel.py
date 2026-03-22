import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

data = pd.read_csv('dataset_regresion_logistica.csv')

X = data.drop('target', axis=1)
y = data['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

logistic_model = LogisticRegression()
logistic_model.fit(X_train_scaled, y_train)

y_pred = logistic_model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

def predict_purchase(edad, ingreso_mensual, visitas_web_mes, tiempo_sitio_min, compras_previas, descuento_usado):
    input_data = pd.DataFrame([[edad, ingreso_mensual, visitas_web_mes, tiempo_sitio_min, compras_previas, descuento_usado]],
                               columns=['edad', 'ingreso_mensual', 'visitas_web_mes', 'tiempo_sitio_min', 'compras_previas', 'descuento_usado'])
    input_scaled = scaler.transform(input_data)
    prediction = logistic_model.predict(input_scaled)[0]
    probability = logistic_model.predict_proba(input_scaled)[0][1]
    return int(prediction), round(float(probability) * 100, 2)

def get_model_accuracy():
    return round(accuracy * 100, 2)

def get_confusion_matrix():
    return confusion_matrix(y_test, y_pred).tolist()

def get_classification_report():
    return classification_report(y_test, y_pred)