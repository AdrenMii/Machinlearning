import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)


# Load and prepare dataset
data = pd.read_csv('Video_Games_Sales_as_at_22_Dec_2016.csv')
data['User_Score'] = pd.to_numeric(data['User_Score'], errors='coerce')
data = data.dropna(subset=['Critic_Score', 'User_Score', 'Critic_Count', 'User_Count', 'Global_Sales'])

# Binary target: 1 = commercial success (Global_Sales > median), 0 = not
median_sales = data['Global_Sales'].median()
data['Success'] = (data['Global_Sales'] > median_sales).astype(int)

features = ['Critic_Score', 'User_Score', 'Critic_Count', 'User_Count']
X = data[features]
y = data['Success']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = ExtraTreesClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

def get_metrics():
    return {
        'accuracy': round(accuracy_score(y_test, y_pred) * 100, 2),
        'precision': round(precision_score(y_test, y_pred) * 100, 2),
        'recall': round(recall_score(y_test, y_pred) * 100, 2),
        'f1': round(f1_score(y_test, y_pred) * 100, 2),
        'roc_auc': round(roc_auc_score(y_test, y_prob) * 100, 2),
        'median_sales': round(median_sales, 2)
    }

def predict_success(critic, user, critic_count, user_count):
    input_data = pd.DataFrame([[critic, user, critic_count, user_count]], columns=features)
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]
    return int(prediction), round(float(probability) * 100, 2)

def get_confusion_matrix_graph():
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation='nearest', cmap='Purples')
    ax.set_title('Confusion Matrix', fontsize=12, pad=12)
    tick_marks = [0, 1]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(['Not Successful', 'Successful'], fontsize=9)
    ax.set_yticklabels(['Not Successful', 'Successful'], fontsize=9)
    ax.set_ylabel('True Label', fontsize=10)
    ax.set_xlabel('Predicted Label', fontsize=10)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > thresh else '#6610f2',
                    fontsize=14, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.patch.set_facecolor('#ffffff')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img

def get_feature_importance_graph():
    importances = model.feature_importances_
    fig, ax = plt.subplots(figsize=(6, 3.5))
    colors = ['#6610f2' if i == np.argmax(importances) else '#ede9ff' for i in range(len(features))]
    bars = ax.barh(features, importances, color=colors, edgecolor='none')
    ax.set_xlabel('Importance', fontsize=10)
    ax.set_title('Feature Importance', fontsize=12, pad=10)
    for bar, val in zip(bars, importances):
        ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, color='#374151')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_alpha(0.3)
    ax.spines['bottom'].set_alpha(0.3)
    fig.patch.set_facecolor('#ffffff')
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img