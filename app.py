from flask import Flask, render_template

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

if __name__ == '__main__':
    app.run(debug=True)