# 🎓 JEE Journey Predictor & Insights Hub

[![Python Version](https://img.shields.ly/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![Flask Version](https://img.shields.ly/badge/flask-3.1.3-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.ly/badge/scikit--learn-1.7.2-orange.svg)](https://scikit-learn.org/)
[![LIME Version](https://img.shields.ly/badge/XAI-LIME-blueviolet.svg)](#)
[![License](https://img.shields.ly/badge/license-MIT-brightgreen.svg)](#)

A full-stack, state-of-the-art machine learning web application designed to forecast and mitigate dropout risks for students preparing for the highly competitive Joint Entrance Examination (JEE) in India. 

Leveraging historical student dataset indicators and modern predictive modeling, the platform serves as a powerful diagnostic tool for educators, counselors, and students to analyze academic standing, preparation habits, and socio-emotional factors—delivering immediate actionable guidance and wellness support recommendations.

---

## ✨ Features & Capabilities

### 1. 📋 Multi-Step Profile Wizard
* **Effortless Ingestion**: Captures **14 critical student variables** categorized into three elegant logical phases:
  * **Step 1: Academic Standing** (JEE Main %ile, JEE Advanced %, Mock Averages, Attempts, Class 12th %)
  * **Step 2: Preparation Setting** (Coaching, Daily Study Hours, Alternative backups, Location)
  * **Step 3: Socio-Emotional Profile** (School Board, Family Income Bracket, Parent Education, Peer rivalry, Mental strain)
* **Real-time Boundary Checking**: Front-end validation toasts immediately alert users if values exceed mathematical bounds.

### 2. ⚡ Asynchronous REST API Inference
* **No Flashing Screen Reloads**: Inferences are managed asynchronously via a high-performance custom Flask REST endpoint using the **Fetch API**.
* **Immersive Loader**: Displays simulated computing operations (e.g. *Ingesting features... Scaling vectors... Running classifier...*) to convey diagnostic rigor.

### 3. 🎯 Real-Time "What-If" Simulator
* **Interactive Tweak Sliders**: Allows counselors to adjust study hours or mock scores directly on the results screen.
* **Instant Recalculation**: Background fetch events instantly pulse and update the risk percentage circle and counsel boxes in real-time.

### 4. 🧭 Actionable Recommendations Matrix
* **Targeted Intervention**: Displays immediate color-coded directives mapped directly to risk parameters:
  * ⚠️ **Burnout warnings** if study hours exceed 9.5 hours daily.
  * 📈 **Preparation optimization guidelines** if mock scores drop below 55%.
  * ❤️ **Wellness and counseling encouragement** if high mental strain or peer pressure is checked.

### 5. 📊 Model Diagnostics Console
* **Tabbed Diagnostics**: Includes a mockup container that toggles between:
  1. **Supervised Algorithms Comparison**: Displaying metrics for Logistic Regression, Decision Tree, KNN, Random Forest, and SVM.
  2. **Model Confusion Matrix**: Detailing precision and classification accuracy.
  3. **Key Predictors Guide**: Reviewing which variables influence outcomes most heavily.

### 6. 🧠 Explainable AI (LIME) Local Attributions
* **Official LIME Integration**: Integrates the official `lime.lime_tabular.LimeTabularExplainer` fitted on the training dataset to explain individual model predictions in real-time.
* **Feature Contribution Visualizer**: Renders dynamic, responsive horizontal contribution bars:
  * 🔴 **Glow Coral Red**: Features that increase dropout risk (positive contributions).
  * 🟢 **Glow Emerald Green**: Factors that act as protective drivers (negative contributions).
* **Reverse Preprocessing on-the-fly**: Preprocessed values are decoded back to human-readable scales (e.g., `2.0 hrs` daily study, `Low` family income) for high interpretability.

---

## 🛠️ Technology Stack

| Layer | Technologies |
| :--- | :--- |
| **Frontend** | HTML5, Vanilla HSL CSS3, JavaScript (ES6+ REST Fetch), Lucide Icons, Outfit Typography |
| **Backend** | Python 3.10+, Flask REST API |
| **Machine Learning & XAI** | Scikit-Learn, Pandas, NumPy, LIME (Local Interpretable Model-agnostic Explanations), Joblib |
| **Graphics** | Matplotlib, Seaborn |

---

## 📂 Project Directory Structure

Consolidated project directories now map elegantly under a single, clean workspace:

```text
JEE-DROPOUT-SYSTEM/
│
├── dataset/
│   └── JEE_Dropout_After_Class_12.csv   # Historical training database
│
├── static/
│   ├── style.css                        # Immersive glassmorphic design system
│   ├── model_comparison.png             # Graph: F1-score comparisons
│   └── confusion_matrix.png             # Graph: Confusion matrix of the champion model
│
├── templates/
│   └── index.html                       # Multi-step UI wizard, simulator and scripts
│
├── app.py                               # Flask server and inference REST endpoint
├── model_training.py                    # Automated ML training pipeline
├── PROJECT_REPORT.md                    # Core internship/project summary report
├── .gitignore                           # Consolidated Git ignore list
└── README.md                            # Comprehensive hub documentation
```

---

## 🚀 Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/Advaith4/JEE-DROPOUT-SYSTEM.git
cd JEE-DROPOUT-SYSTEM
```

### 2. Install dependencies
Ensure you have **Python 3.10 or later** installed. Install the necessary mathematical, statistical, and server packages:
```bash
pip install flask joblib pandas numpy scikit-learn matplotlib seaborn lime
```

### 3. (Optional) Run the ML Pipeline
To retrain the models, select the champion F1-score model, and regenerate diagnostic graphs inside `static/`:
```bash
python model_training.py
```

### 4. Boot the server
Launch the local development Flask server:
```bash
python app.py
```
Open your favorite web browser and navigate to **[http://127.0.0.1:5000/](http://127.0.0.1:5000/)** to access the system!

---

## 🔬 Machine Learning Performance Summary

The automated pipeline evaluates 5 algorithms (Logistic Regression, Decision Trees, Random Forests, SVM, and KNN) on stratified test splits. The system selects the champion based on **F1 Score** to balance precision and recall. 

You can view the comparisons and matrices directly in the application's **Diagnostics Console**!
