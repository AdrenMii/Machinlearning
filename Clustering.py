import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import base64
import io

# ─────────────────────────────────────────────────────────────────────────────
#  PART 1 ── Manual K-Means Simulation
#  Dataset: 100 records  |  2 features: Age & Monthly Income (USD)
# ─────────────────────────────────────────────────────────────────────────────

MANUAL_DATASET = [
    {"id":1,"age":22,"monthly_income":1100},{"id":2,"age":24,"monthly_income":1250},
    {"id":3,"age":21,"monthly_income":980}, {"id":4,"age":26,"monthly_income":1400},
    {"id":5,"age":23,"monthly_income":1050},{"id":6,"age":25,"monthly_income":1180},
    {"id":7,"age":20,"monthly_income":900}, {"id":8,"age":27,"monthly_income":1320},
    {"id":9,"age":22,"monthly_income":1060},{"id":10,"age":28,"monthly_income":1500},
    {"id":11,"age":19,"monthly_income":850},{"id":12,"age":24,"monthly_income":1200},
    {"id":13,"age":26,"monthly_income":1350},{"id":14,"age":21,"monthly_income":1000},
    {"id":15,"age":29,"monthly_income":1600},{"id":16,"age":23,"monthly_income":1130},
    {"id":17,"age":27,"monthly_income":1450},{"id":18,"age":20,"monthly_income":920},
    {"id":19,"age":25,"monthly_income":1280},{"id":20,"age":28,"monthly_income":1520},
    {"id":21,"age":34,"monthly_income":2400},{"id":22,"age":36,"monthly_income":2650},
    {"id":23,"age":32,"monthly_income":2200},{"id":24,"age":38,"monthly_income":2900},
    {"id":25,"age":35,"monthly_income":2500},{"id":26,"age":37,"monthly_income":2750},
    {"id":27,"age":33,"monthly_income":2300},{"id":28,"age":39,"monthly_income":3000},
    {"id":29,"age":31,"monthly_income":2100},{"id":30,"age":40,"monthly_income":3100},
    {"id":31,"age":36,"monthly_income":2700},{"id":32,"age":34,"monthly_income":2450},
    {"id":33,"age":38,"monthly_income":2850},{"id":34,"age":32,"monthly_income":2150},
    {"id":35,"age":41,"monthly_income":3200},{"id":36,"age":35,"monthly_income":2550},
    {"id":37,"age":37,"monthly_income":2800},{"id":38,"age":33,"monthly_income":2250},
    {"id":39,"age":39,"monthly_income":2950},{"id":40,"age":42,"monthly_income":3300},
    {"id":41,"age":47,"monthly_income":4800},{"id":42,"age":49,"monthly_income":5100},
    {"id":43,"age":45,"monthly_income":4500},{"id":44,"age":51,"monthly_income":5400},
    {"id":45,"age":48,"monthly_income":4950},{"id":46,"age":50,"monthly_income":5200},
    {"id":47,"age":46,"monthly_income":4650},{"id":48,"age":52,"monthly_income":5500},
    {"id":49,"age":44,"monthly_income":4400},{"id":50,"age":53,"monthly_income":5700},
    {"id":51,"age":48,"monthly_income":4850},{"id":52,"age":47,"monthly_income":4750},
    {"id":53,"age":51,"monthly_income":5300},{"id":54,"age":45,"monthly_income":4600},
    {"id":55,"age":54,"monthly_income":5800},{"id":56,"age":49,"monthly_income":5050},
    {"id":57,"age":50,"monthly_income":5150},{"id":58,"age":46,"monthly_income":4700},
    {"id":59,"age":52,"monthly_income":5450},{"id":60,"age":55,"monthly_income":5950},
    {"id":61,"age":23,"monthly_income":1150},{"id":62,"age":21,"monthly_income":970},
    {"id":63,"age":26,"monthly_income":1380},{"id":64,"age":24,"monthly_income":1220},
    {"id":65,"age":28,"monthly_income":1480},{"id":66,"age":22,"monthly_income":1080},
    {"id":67,"age":27,"monthly_income":1410},{"id":68,"age":20,"monthly_income":910},
    {"id":69,"age":25,"monthly_income":1260},{"id":70,"age":29,"monthly_income":1570},
    {"id":71,"age":35,"monthly_income":2520},{"id":72,"age":33,"monthly_income":2280},
    {"id":73,"age":37,"monthly_income":2720},{"id":74,"age":31,"monthly_income":2080},
    {"id":75,"age":39,"monthly_income":2970},{"id":76,"age":36,"monthly_income":2620},
    {"id":77,"age":34,"monthly_income":2420},{"id":78,"age":38,"monthly_income":2820},
    {"id":79,"age":32,"monthly_income":2170},{"id":80,"age":40,"monthly_income":3070},
    {"id":81,"age":49,"monthly_income":5020},{"id":82,"age":47,"monthly_income":4720},
    {"id":83,"age":51,"monthly_income":5320},{"id":84,"age":45,"monthly_income":4520},
    {"id":85,"age":53,"monthly_income":5620},{"id":86,"age":48,"monthly_income":4870},
    {"id":87,"age":50,"monthly_income":5120},{"id":88,"age":46,"monthly_income":4620},
    {"id":89,"age":52,"monthly_income":5420},{"id":90,"age":54,"monthly_income":5720},
    {"id":91,"age":22,"monthly_income":1020},{"id":92,"age":24,"monthly_income":1170},
    {"id":93,"age":26,"monthly_income":1310},{"id":94,"age":28,"monthly_income":1460},
    {"id":95,"age":30,"monthly_income":1650},{"id":96,"age":36,"monthly_income":2580},
    {"id":97,"age":38,"monthly_income":2880},{"id":98,"age":48,"monthly_income":4920},
    {"id":99,"age":50,"monthly_income":5170},{"id":100,"age":52,"monthly_income":5480},
]


def getManualDataset():
    """Returns the 100-record manual exercise dataset (Part 1)."""
    return MANUAL_DATASET


def runManualKMeans():
    """
    Runs 3 manual K-Means iterations on the 100-record dataset.
    Returns iteration tables, centroids, variances, and a Base64 variance chart.
    """
    records = MANUAL_DATASET
    points = np.array([[r["age"], r["monthly_income"]] for r in records], dtype=float)

    # Manually chosen initial centroids (as required by the rubric)
    centroids = np.array([
        [24.0, 1250.0],   # C1 – Young / Low income
        [36.0, 2600.0],   # C2 – Middle-aged / Mid income
        [50.0, 5100.0],   # C3 – Senior / High income
    ])

    iterations_data = []
    variances = []

    for iteration in range(3):
        # Euclidean distances from each point to each centroid
        diffs = points[:, np.newaxis, :] - centroids[np.newaxis, :, :]
        distances = np.sqrt((diffs ** 2).sum(axis=2))

        # Assign each point to nearest centroid
        assignments = np.argmin(distances, axis=1)

        # Within-cluster sum of squared distances (variance)
        variance = float(sum(distances[i, assignments[i]] ** 2 for i in range(len(points))))
        variances.append(round(variance, 2))

        # Build table rows (all 100 records)
        table_rows = []
        for i in range(len(points)):
            table_rows.append({
                "id": records[i]["id"],
                "age": int(points[i][0]),
                "income": int(points[i][1]),
                "d1": round(float(distances[i][0]), 2),
                "d2": round(float(distances[i][1]), 2),
                "d3": round(float(distances[i][2]), 2),
                "cluster": int(assignments[i]) + 1,
            })

        # Recalculate centroids
        new_centroids = []
        for k in range(3):
            mask = assignments == k
            if mask.any():
                new_centroids.append(points[mask].mean(axis=0))
            else:
                new_centroids.append(centroids[k])
        new_centroids = np.array(new_centroids)

        iterations_data.append({
            "iteration": iteration + 1,
            "centroids_before": [
                {"id": j + 1, "age": round(float(centroids[j][0]), 2),
                 "income": round(float(centroids[j][1]), 2)}
                for j in range(3)
            ],
            "centroids_after": [
                {"id": j + 1, "age": round(float(new_centroids[j][0]), 2),
                 "income": round(float(new_centroids[j][1]), 2)}
                for j in range(3)
            ],
            "table": table_rows,
            "variance": round(variance, 2),
        })

        centroids = new_centroids

    # ── Variance line chart ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4), facecolor="#f8f5ff")
    ax.set_facecolor("#f8f5ff")
    ax.plot([1, 2, 3], variances, marker="o", linewidth=2.5,
            color="#6610f2", markersize=9, markerfacecolor="#ff4081",
            markeredgecolor="#6610f2", markeredgewidth=1.5)
    for i, v in enumerate(variances):
        ax.annotate(f"{v:,.0f}", (i + 1, v),
                    textcoords="offset points", xytext=(0, 12),
                    ha="center", color="#333", fontsize=9, fontweight="bold")
    ax.set_title("Variance Reduction Across Iterations", color="#333",
                 fontsize=13, fontweight="bold", pad=14)
    ax.set_xlabel("Iteration", color="#555", fontsize=11)
    ax.set_ylabel("Total Variance (SSE)", color="#555", fontsize=11)
    ax.set_xticks([1, 2, 3])
    ax.tick_params(colors="#555")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    variance_chart = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "dataset": records,
        "iterations": iterations_data,
        "variances": variances,
        "variance_chart": variance_chart,
        "initial_centroids": [
            {"id": 1, "age": 24.0, "income": 1250.0},
            {"id": 2, "age": 36.0, "income": 2600.0},
            {"id": 3, "age": 50.0, "income": 5100.0},
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
#  PART 2 ── Sklearn K-Means Application
#  Dataset: 1 200 synthetic records  |  Context: E-commerce Customer Segmentation
#  Features: Age, Annual Income (USD), Spending Score (1-100)
# ─────────────────────────────────────────────────────────────────────────────

def getLargeDataset():
    """
    Generates 1 200 synthetic e-commerce customer records.
    Features: age (18-70), annual_income (15 000–120 000), spending_score (1-100).
    """
    rng = np.random.default_rng(seed=42)
    n = 1200

    # Three natural customer segments seeded into the data
    seg_sizes = [400, 400, 400]
    records = []

    params = [
        # (age_mu, age_sig, income_mu, income_sig, score_mu, score_sig)
        (25, 4,  28000, 6000,  72, 12),   # Young / Budget shoppers / High engagement
        (40, 6,  65000, 12000, 45, 15),   # Mid-career / Medium spenders
        (55, 7, 105000, 10000, 25, 10),   # Senior / High income / Low spending score
    ]

    pid = 1
    for seg_idx, (n_seg, p) in enumerate(zip(seg_sizes, params)):
        ages    = np.clip(rng.normal(p[0], p[1], n_seg), 18, 70).astype(int)
        incomes = np.clip(rng.normal(p[2], p[3], n_seg), 15000, 120000).astype(int)
        scores  = np.clip(rng.normal(p[4], p[5], n_seg), 1, 100).astype(int)
        for i in range(n_seg):
            records.append({
                "id": pid,
                "age": int(ages[i]),
                "annual_income": int(incomes[i]),
                "spending_score": int(scores[i]),
            })
            pid += 1

    # Shuffle so segments aren't in order
    rng.shuffle(records)
    for new_id, r in enumerate(records, start=1):
        r["id"] = new_id

    return records


def applyClusteringKMeans():
    """
    Applies scaled K-Means (k=3) to the large 1 200-record dataset.
    Returns cluster assignments, summary, centroids (original scale), and a Base64 scatter plot.
    """
    data = getLargeDataset()
    X_raw = np.array([[r["age"], r["annual_income"], r["spending_score"]] for r in data], dtype=float)

    # ── Preprocessing ──────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    # ── Model training ─────────────────────────────────────────────────────
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10, max_iter=300)
    labels = kmeans.fit_predict(X_scaled)

    # ── Cluster assignments ────────────────────────────────────────────────
    result = []
    for i, record in enumerate(data):
        row = record.copy()
        row["cluster"] = int(labels[i]) + 1
        result.append(row)

    # ── Cluster summary ────────────────────────────────────────────────────
    summary = {}
    for i in range(3):
        mask = labels == i
        cluster_points = X_raw[mask]
        summary[i + 1] = {
            "count": int(mask.sum()),
            "avg_age": round(float(cluster_points[:, 0].mean()), 1),
            "avg_income": round(float(cluster_points[:, 1].mean()), 0),
            "avg_score": round(float(cluster_points[:, 2].mean()), 1),
        }

    # ── Centroids in original scale ────────────────────────────────────────
    centroids_scaled = kmeans.cluster_centers_
    centroids_original = scaler.inverse_transform(centroids_scaled)
    centroids = [
        {
            "cluster": i + 1,
            "age": round(float(centroids_original[i][0]), 1),
            "annual_income": round(float(centroids_original[i][1]), 0),
            "spending_score": round(float(centroids_original[i][2]), 1),
        }
        for i in range(3)
    ]

    # ── Scatter plot (Age vs Annual Income, coloured by cluster) ──────────
    colors = ["#6610f2", "#0d6efd", "#20c997"]
    labels_display = ["Cluster 1", "Cluster 2", "Cluster 3"]

    fig, ax = plt.subplots(figsize=(9, 6), facecolor="#f8f5ff")
    ax.set_facecolor("#f8f5ff")

    for k in range(3):
        mask = labels == k
        ax.scatter(X_raw[mask, 0], X_raw[mask, 1],
                   c=colors[k], label=labels_display[k],
                   alpha=0.55, edgecolors="none", s=30)

    # Plot centroids
    for k in range(3):
        ax.scatter(centroids_original[k, 0], centroids_original[k, 1],
                   c=colors[k], marker="*", s=280,
                   edgecolors="black", linewidths=0.8,
                   zorder=5, label=f"Centroid {k+1}")

    ax.set_title("K-Means Clustering — Age vs Annual Income",
                 fontsize=13, fontweight="bold", color="#333", pad=12)
    ax.set_xlabel("Age", fontsize=11, color="#555")
    ax.set_ylabel("Annual Income (USD)", fontsize=11, color="#555")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.7)
    ax.tick_params(colors="#555")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle="--", alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    scatter_plot = base64.b64encode(buf.read()).decode("utf-8")

    return {
        "clusters": result[:50],          # first 50 rows for the HTML table
        "all_clusters": result,
        "summary": summary,
        "centroids": centroids,
        "scatter_plot": scatter_plot,
        "total_records": len(data),
    }