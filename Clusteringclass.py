import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import base64
import io

def get_class_dataset():
    return [
        {"nombre": "Ana", "edad": 22, "ingresos": 1200, "gasto": 300},
        {"nombre": "Luis", "edad": 25, "ingresos": 1500, "gasto": 350},
        {"nombre": "Carlos", "edad": 23, "ingresos": 1300, "gasto": 280},
        {"nombre": "Marta", "edad": 45, "ingresos": 4000, "gasto": 1200},
        {"nombre": "Sofía", "edad": 50, "ingresos": 4200, "gasto": 1400},
        {"nombre": "Jorge", "edad": 47, "ingresos": 3900, "gasto": 1100},
        {"nombre": "Elena", "edad": 31, "ingresos": 2500, "gasto": 700},
        {"nombre": "Pedro", "edad": 33, "ingresos": 2700, "gasto": 750},
        {"nombre": "Laura", "edad": 29, "ingresos": 2400, "gasto": 680},
        {"nombre": "Andrés", "edad": 52, "ingresos": 5000, "gasto": 1600},
        {"nombre": "Camila", "edad": 21, "ingresos": 1100, "gasto": 250},
        {"nombre": "Diego", "edad": 38, "ingresos": 3200, "gasto": 900}
    ]

# ESTE NOMBRE DEBE COINCIDIR CON LO QUE PUSISTE EN APP.PY
def apply_class_clustering(): 
    data = get_class_dataset()
    x_features = [(p["edad"], p["ingresos"], p["gasto"]) for p in data]
    
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_features)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(x_scaled)

    results_list = []
    for i, person in enumerate(data):
        row = person.copy()
        row["cluster"] = int(labels[i])
        results_list.append(row)
    
    summary = {}
    for label in labels:
        l = int(label)
        summary[l] = summary.get(l, 0) + 1

    # Generación del gráfico
    points = np.array(x_features)
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ['#6610f2', '#0d6efd', '#20c997']
    
    for k in range(3):
        mask = labels == k
        ax.scatter(points[mask, 0], points[mask, 1], c=colors[k], label=f"Cluster {k}", s=100, edgecolors='white')

    ax.set_title("Class Exercise: Age vs Income", fontsize=12, fontweight='bold')
    ax.set_xlabel("Age")
    ax.set_ylabel("Monthly Income")
    ax.legend()
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120)
    plt.close(fig)
    graph_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

    return {
        "clusters": results_list,
        "summary": summary,
        "graph": graph_base64
    }