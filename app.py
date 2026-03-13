from flask import Flask, render_template, request
from LineaRegression import calculateGrade
from LogisticRegressionModel import predict_purchase, get_accuracy, get_confusion

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/npc')
def npc():
    return render_template('npc.html')

@app.route('/procedural')
def procedural():
    return render_template('procedural.html')

@app.route('/LineaRegression', methods=["GET", "POST"])
def calculateGradeRoute():
    calculateResult = None
    if request.method == "POST":
        hours = float(request.form["hours"])
        calculateResult = calculateGrade(hours)
    return render_template("linearRegressionGrades.html", result=calculateResult)

@app.route('/logistic', methods=["GET", "POST"])
def logistic():
    result = None
    probability = None
    form_data = None

    if request.method == "POST":
        edad = float(request.form["edad"])
        ingreso_mensual = float(request.form["ingreso_mensual"])
        visitas_web_mes = float(request.form["visitas_web_mes"])
        tiempo_sitio_min = float(request.form["tiempo_sitio_min"])
        compras_previas = float(request.form["compras_previas"])
        descuento_usado = float(request.form["descuento_usado"])

        result, probability = predict_purchase(
            edad,
            ingreso_mensual,
            visitas_web_mes,
            tiempo_sitio_min,
            compras_previas,
            descuento_usado
        )

        form_data = request.form

    return render_template(
        "logisticRegression.html",
        result=result,
        probability=probability,
        accuracy=get_accuracy(),
        conf_matrix=get_confusion(),
        form_data=form_data
    )

if __name__ == '__main__':
    app.run(debug=True)