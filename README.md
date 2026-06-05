# 📊 LinearLearn — Python/Flask ML Edition

> **A complete conversion of a JavaScript Linear Regression app into a production-grade Python Machine Learning web application using Flask, scikit-learn, Pandas, NumPy, and Matplotlib.**

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0+-black?logo=flask)](https://flask.palletsprojects.com)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-orange?logo=scikit-learn)](https://scikit-learn.org)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-3.8+-green)](https://matplotlib.org)

---

## 🔍 Project Overview

**LinearLearn** is an interactive, step-by-step educational web application that teaches Linear Regression through hands-on experimentation. This version is a **complete rewrite** of the original JavaScript/Chart.js frontend application into a full-stack Python web application.

The original project performed all regression computations in the browser using JavaScript. This version moves all ML logic to a **Flask backend** powered by industry-standard Python ML libraries — making it suitable for AI/ML internship portfolios and production deployment.

---

## ✨ Features

| Feature | JavaScript (Original) | Python/Flask (This Version) |
|---|---|---|
| Data loading | FileReader + PapaParse | `pandas.read_csv()` |
| Missing value imputation | Manual JS loops | `df.fillna(median)` |
| Categorical encoding | Frequency maps in JS | `sklearn.LabelEncoder` |
| Linear Regression | Manual β = (XᵀX)⁻¹Xᵀy | `sklearn.LinearRegression` |
| Feature scaling | Manual z-score in JS | `sklearn.StandardScaler` |
| Train/test split | Manual array slicing | `train_test_split()` |
| Cross-validation | Not implemented | `cross_val_score()` |
| Plots | Chart.js (browser) | Matplotlib PNG (server) |
| Metrics | Manual JS summations | `sklearn.metrics` |

### Complete Feature List

- 📁 **CSV Upload** or built-in sample Housing dataset (50 rows)
- 👁️ **Data Preview** — shape, dtypes, missing values, descriptive stats
- 🔧 **Preprocessing** — median imputation, label encoding, automated pipeline
- 📊 **EDA** — server-generated correlation heatmap + feature vs target scatter plots
- 📖 **Theory Section** — hypothesis, cost function, gradient descent, normal equation, scikit-learn implementation
- ⚙️ **Training Config** — multi-feature selection, test split slider, CV folds, optional StandardScaler
- 🚀 **Model Training** — regression line, predicted vs actual, residual plot, coefficient bar chart
- 🎯 **Prediction** — step-by-step dot product breakdown for any new input
- 📈 **Evaluation** — MSE, RMSE, MAE, R², Adjusted R², comparison table, summary chart

---

## 📁 Project Structure

```
linear_regression_flask/
│
├── app.py                        # Flask backend — all ML logic
├── requirements.txt              # Python dependencies
│
├── templates/
│   └── index.html                # Jinja2 template — full UI
│
├── static/
│   ├── css/
│   │   └── style.css             # Dark terminal aesthetic stylesheet
│   ├── js/
│   │   └── app.js                # Frontend JS — API calls & UI state
│   └── generated_plots/          # Matplotlib PNGs saved here at runtime
│
└── README.md
```

---

## 🛠️ Technology Stack

- **Python 3.10+** — backend language
- **Flask 3.0** — lightweight WSGI web framework
- **pandas 2.2** — DataFrame operations, CSV parsing, imputation
- **NumPy 1.26** — numerical computations, array operations
- **scikit-learn 1.4** — `LinearRegression`, `StandardScaler`, `LabelEncoder`, `train_test_split`, `cross_val_score`, `mean_squared_error`, `r2_score`
- **Matplotlib 3.8** — server-side plot generation (non-interactive `Agg` backend)
- **Gunicorn** — production WSGI server

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or newer
- pip

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/linear-regression-flask.git
cd linear-regression-flask

# 2. (Recommended) Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Running Locally

```bash
python app.py
```

Open your browser at **http://localhost:5000**

The app runs in Flask debug mode by default. All generated plots are saved to `static/generated_plots/` and served via Flask's static file handler.

---

## 🔌 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the main HTML page |
| `POST` | `/api/upload` | Upload CSV or load sample dataset |
| `POST` | `/api/preprocess` | Run pandas/sklearn preprocessing pipeline |
| `POST` | `/api/eda` | Generate Matplotlib EDA plots |
| `POST` | `/api/train` | Fit LinearRegression + generate training plots |
| `POST` | `/api/predict` | Run inference with step-by-step breakdown |
| `POST` | `/api/evaluate` | Return metrics + evaluation summary plot |

---

## 🧠 JavaScript → Python Conversion Summary

### Data Loading
```
JS:  new FileReader() + Papa.parse(file)
PY:  pandas.read_csv(file)              # one line, handles encoding, dtypes automatically
```

### Missing Value Imputation
```
JS:  for(let i=0;i<data.length;i++) if(isNaN(data[i][col])) data[i][col] = median;
PY:  df[col] = df[col].fillna(df[col].median())
```

### Categorical Encoding
```
JS:  manual frequency map + integer assignment
PY:  from sklearn.preprocessing import LabelEncoder
     le = LabelEncoder()
     df[col] = le.fit_transform(df[col].astype(str))
```

### Linear Regression (Core)
```
JS:  beta1 = sumXY / sumXX;  beta0 = meanY - beta1 * meanX;  // only worked for 1 feature
PY:  from sklearn.linear_model import LinearRegression
     model = LinearRegression()
     model.fit(X_train, y_train)        # supports any number of features via LAPACK QR
```

### Train/Test Split
```
JS:  const split = Math.floor(n * 0.8);  X_train = X.slice(0, split); ...
PY:  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

### Evaluation Metrics
```
JS:  let mse=0; for(i) mse += (pred[i]-actual[i])**2; mse/=n;
PY:  from sklearn.metrics import mean_squared_error, r2_score
     mse = mean_squared_error(y_test, y_pred)
     r2  = r2_score(y_test, y_pred)
```

### Plots
```
JS:  Chart.js (canvas, browser-side)
PY:  import matplotlib.pyplot as plt
     fig, ax = plt.subplots(...)
     ax.scatter(...); ax.plot(...)
     fig.savefig("static/generated_plots/plot.png")   # served as static file
```

---

## 🚢 Deployment Instructions

### Option 1: Render (Free tier)

1. Push repo to GitHub
2. Create account at [render.com](https://render.com)
3. New → Web Service → Connect GitHub repo
4. Settings:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `gunicorn app:app`
5. Deploy

### Option 2: Railway

```bash
railway login
railway init
railway up
```

### Option 3: Heroku

```bash
# Add Procfile
echo "web: gunicorn app:app" > Procfile
heroku create your-app-name
git push heroku main
```

### Option 4: Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 5000
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5000"]
```

```bash
docker build -t linearlearn .
docker run -p 5000:5000 linearlearn
```

---

## 📸 Application Flow

```
Upload CSV → pandas.read_csv()
     ↓
Preview   → df.describe(), df.dtypes
     ↓
Preprocess → fillna(median) + LabelEncoder
     ↓
EDA       → Matplotlib heatmap + scatter PNGs → static/generated_plots/
     ↓
Theory    → Interactive tabs explaining math
     ↓
Config    → Select features/target, split%, CV folds, scaling
     ↓
Train     → StandardScaler + LinearRegression.fit() + 4 diagnostic plots
     ↓
Predict   → model.predict() + step-by-step breakdown
     ↓
Evaluate  → sklearn.metrics + summary chart
```

---

## 📄 License

Educational project. Original concept by **Mouly Sikdar**. Converted to Python/Flask ML stack.

---

## 🤝 Portfolio Notes

This project demonstrates:
- **Full-stack Python development** (Flask REST API + Jinja2 templates)
- **ML pipeline implementation** (data → preprocess → train → evaluate)
- **scikit-learn proficiency** (LinearRegression, preprocessing, model selection)
- **Data engineering** (pandas DataFrames, missing value handling, encoding)
- **Data visualization** (Matplotlib server-side rendering)
- **Software architecture** (session management, API design, error handling)
