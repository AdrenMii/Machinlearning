from flask import Flask, render_template, request
from LineaRegression import calculateGrade, getGraphBase64 as getStudentGraph
from LogisticRegressionModel import predict_purchase, get_logistic_graph, get_model_accuracy, get_confusion_matrix
from VideoGameRegression import predictSales, getGraphBase64 as getVideoGraph

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

@app.route('/adaptive')
def adaptive():
    return render_template('adaptive.html')

@app.route('/anticheat')
def anticheat():
    return render_template('anticheat.html')

@app.route('/linear-concepts', methods=["GET", "POST"])
def linearConcepts():
    graph = getStudentGraph()
    result = None
    if request.method == "POST":
        hours = float(request.form["hours"])
        result = calculateGrade(hours)
    return render_template('linearRegressionConcepts.html', graph=graph, result=result)

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
        # ... (tu código de captura de datos que ya funciona)
        edad = float(request.form["edad"])
        ingreso_mensual = float(request.form["ingreso_mensual"])
        visitas_web_mes = float(request.form["visitas_web_mes"])
        tiempo_sitio_min = float(request.form["tiempo_sitio_min"])
        compras_previas = float(request.form["compras_previas"])
        descuento_usado = float(request.form["descuento_usado"])
        
        result, probability = predict_purchase(edad, ingreso_mensual, visitas_web_mes, 
                                               tiempo_sitio_min, compras_previas, descuento_usado)
        form_data = request.form

    return render_template(
        "logisticRegression.html",
        result=result,
        probability=probability,
        accuracy=get_model_accuracy(),
        conf_matrix=get_confusion_matrix(),
        form_data=form_data
    )

@app.route('/video-games', methods=["GET", "POST"])
def video_games():
    graph = getVideoGraph()
    result = None

    if request.method == "POST":
        critic = float(request.form["critic"])
        user = float(request.form["user"])
        critic_count = float(request.form["critic_count"])
        user_count = float(request.form["user_count"])
        result = predictSales(critic, user, critic_count, user_count)

    return render_template("videoGameRegression.html", result=result, graph=graph)

@app.route('/logistic-concepts')
def logistic_concepts():
    # Esta función llama a la que importamos arriba
    graph_base64 = get_logistic_graph() 
    return render_template("logisticRegressionConcepts.html", graph=graph_base64)

if __name__ == '__main__':
    app.run(debug=True)