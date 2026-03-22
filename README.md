# 🤖 Machine Learning in Video Games — Flask Web Application

A Flask web application that presents four Machine Learning use cases applied to video games, along with supervised learning modules focused on Linear and Logistic Regression.

---

## 📋 Description

This project was developed as part of the Machine Learning course (Semester 6). It demonstrates the practical application of ML concepts through an interactive web interface built with Flask, Python and Bootstrap 5.

**Topic:** Artificial Intelligence in Video Games

---

## 🚀 Features

- **4 ML Use Cases:** Intelligent NPCs, Procedural Generation, Adaptive Difficulty, Anti-Cheat Detection
- **Linear Regression:** Grade prediction based on study hours, with regression graph visualization
- **Logistic Regression:** Purchase prediction based on user profile, with confusion matrix
- **Global navbar** using Jinja2 template inclusion
- **Responsive design** with Bootstrap 5

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python 3 | Backend language |
| Flask | Web framework |
| scikit-learn | ML models |
| pandas | Data handling |
| matplotlib | Regression graph visualization |
| Bootstrap 5 | Frontend styling |
| GitHub | Version control |

---

## 📁 Project Structure

```
Machinlearning/
├── app.py                          # Flask routes
├── LineaRegression.py              # Linear regression model + graph
├── LogisticRegressionModel.py      # Logistic regression model
├── dataset_regresion_logistica.csv # Dataset for logistic regression
├── static/
│   └── home.png                    # Home button icon
│   └── maxresdefault.jpg           # NPC page background image
└── templates/
    ├── navbar_light.html           # Global light navbar (Jinja2 include)
    ├── navbar_dark.html            # Global dark navbar for NPC page
    ├── home.html                   # Main page
    ├── npc.html                    # Use Case 1 - Intelligent NPCs
    ├── procedural.html             # Use Case 2 - Procedural Generation
    ├── adaptive.html               # Use Case 3 - Adaptive Difficulty
    ├── anticheat.html              # Use Case 4 - Anti-Cheat Detection
    ├── linearRegressionConcepts.html  # Linear Regression concepts + graph
    ├── linearRegressionGrades.html    # Linear Regression predictor
    └── logisticRegression.html        # Logistic Regression predictor
```

---

## ⚙️ How to Run Locally

**1. Clone the repository:**
```bash
git clone https://github.com/AdrenMii/Machinlearning.git
cd Machinlearning
```

**2. Create and activate a virtual environment:**
```bash
python -m venv .venv

# Windows
.venv\Scripts\Activate.ps1

# Mac/Linux
source .venv/bin/activate
```

**3. Install dependencies:**
```bash
pip install flask scikit-learn pandas matplotlib
```

**4. Run the application:**
```bash
py -m flask run
```

**5. Open in browser:**
```
http://127.0.0.1:5000
```

---

## 🌿 Branches

| Branch | Description |
|---|---|
| `main` | Stable version |
| `use-cases` | ML use case pages (NPC, Procedural, Adaptive, Anti-Cheat) |
| `linear-regression` | Linear regression module (concepts, graph, predictor) |

---

## 📌 Application Routes

| Route | Page |
|---|---|
| `/` | Home |
| `/npc` | Intelligent NPCs |
| `/procedural` | Procedural Generation |
| `/adaptive` | Adaptive Difficulty |
| `/anticheat` | Anti-Cheat Detection |
| `/linear-concepts` | Linear Regression — Basic Concepts + Graph |
| `/LineaRegression` | Linear Regression — Grade Predictor |
| `/logistic` | Logistic Regression — Purchase Predictor |

---

## 👤 Author

**Jose Hernandez** — Systems Engineering Student  
GitHub: [@AdrenMii](https://github.com/AdrenMii)
