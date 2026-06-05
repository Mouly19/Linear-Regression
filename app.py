"""
Linear Regression ML Web Application
Flask backend with scikit-learn, pandas, numpy, and matplotlib
Author: Converted from JavaScript to Python/Flask
"""

import os
import uuid
import json
import traceback
import io
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server-side rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

from flask import Flask, request, jsonify, render_template, send_from_directory
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

# ── App Setup ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB upload limit

PLOTS_DIR = os.path.join('static', 'generated_plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# In-memory session store (keyed by session_id sent from the client)
# Stores: raw df, processed df, model, scaler, encoders, column info
SESSION_STORE: dict = {}

# ── Matplotlib Style ───────────────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#0f1117',
    'axes.facecolor':   '#1a1d2e',
    'axes.edgecolor':   '#3a3d5c',
    'axes.labelcolor':  '#c8cce8',
    'text.color':       '#c8cce8',
    'xtick.color':      '#8a8db0',
    'ytick.color':      '#8a8db0',
    'grid.color':       '#2a2d45',
    'grid.alpha':       0.6,
    'font.family':      'monospace',
    'axes.titlesize':   13,
    'axes.labelsize':   11,
})

ACCENT   = '#6c63ff'
ACCENT2  = '#ff6584'
ACCENT3  = '#43e8d8'
BG_DARK  = '#0f1117'
BG_MID   = '#1a1d2e'

# ── Helper: save figure and return URL path ────────────────────────────────────
def _save_fig(fig, prefix='plot') -> str:
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}.png"
    filepath = os.path.join(PLOTS_DIR, filename)
    fig.savefig(filepath, bbox_inches='tight', dpi=130, facecolor=BG_DARK)
    plt.close(fig)
    return f"/static/generated_plots/{filename}"


# ── Helper: build sample housing dataset ──────────────────────────────────────
def _sample_dataset() -> pd.DataFrame:
    np.random.seed(42)
    n = 50
    area      = np.random.randint(600, 3500, n)
    bedrooms  = np.random.choice([1, 2, 3, 4, 5], n)
    age       = np.random.randint(1, 40, n)
    location  = np.random.choice(['Urban', 'Suburban', 'Rural'], n)
    noise     = np.random.normal(0, 15000, n)
    price     = (area * 150 + bedrooms * 20000 - age * 500
                 + (location == 'Urban').astype(int) * 50000
                 + (location == 'Suburban').astype(int) * 20000
                 + noise).astype(int)
    return pd.DataFrame({
        'area': area, 'bedrooms': bedrooms, 'age': age,
        'location': location, 'price': price
    })


# ── Route: index ───────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')


# ── Route: serve plots ─────────────────────────────────────────────────────────
@app.route('/static/generated_plots/<filename>')
def serve_plot(filename):
    return send_from_directory(PLOTS_DIR, filename)


# ══════════════════════════════════════════════════════════════════════════════
# API: Upload / Sample Dataset
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/upload', methods=['POST'])
def upload_dataset():
    """
    Accepts a CSV file or returns the built-in sample dataset.
    JS equivalent: FileReader → PapaParse → store in memory
    Python:        pandas.read_csv → DataFrame stored in SESSION_STORE
    """
    try:
        session_id = request.form.get('session_id') or uuid.uuid4().hex

        if 'file' in request.files and request.files['file'].filename:
            file = request.files['file']
            if not file.filename.lower().endswith('.csv'):
                return jsonify({'error': 'Only CSV files are supported.'}), 400
            df = pd.read_csv(file)
        else:
            df = _sample_dataset()

        if df.empty:
            return jsonify({'error': 'The uploaded CSV is empty.'}), 400
        if len(df.columns) < 2:
            return jsonify({'error': 'Dataset must have at least 2 columns.'}), 400

        SESSION_STORE[session_id] = {
            'raw_df':       df,
            'processed_df': None,
            'model':        None,
            'scaler':       None,
            'encoders':     {},
            'feature_cols': [],
            'target_col':   None,
            'X_test':       None,
            'y_test':       None,
            'y_pred':       None,
        }

        # Determine column types
        numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        missing          = df.isnull().sum().to_dict()

        # Basic stats for numeric cols
        stats = {}
        for col in numeric_cols:
            s = df[col].describe()
            stats[col] = {k: round(float(v), 4) for k, v in s.items()}

        return jsonify({
            'session_id':      session_id,
            'shape':           list(df.shape),
            'columns':         df.columns.tolist(),
            'numeric_cols':    numeric_cols,
            'categorical_cols': categorical_cols,
            'missing':         {k: int(v) for k, v in missing.items()},
            'dtypes':          df.dtypes.astype(str).to_dict(),
            'stats':           stats,
            'preview':         df.head(10).fillna('').to_dict(orient='records'),
        })

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ══════════════════════════════════════════════════════════════════════════════
# API: Preprocessing
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/preprocess', methods=['POST'])
def preprocess():
    """
    JS equivalent: manual loops for imputation + Label-encoding
    Python:        pandas fillna + sklearn LabelEncoder
    Steps:
      1. Drop columns with > 50 % missing
      2. Numeric NaN → column median
      3. Categorical NaN → mode
      4. Label-encode categoricals
    """
    try:
        data       = request.get_json()
        session_id = data.get('session_id')
        if session_id not in SESSION_STORE:
            return jsonify({'error': 'Session not found. Please upload data first.'}), 400

        df = SESSION_STORE[session_id]['raw_df'].copy()

        # Drop high-missing columns
        threshold   = len(df) * 0.5
        df          = df.dropna(thresh=threshold, axis=1)

        encoders    = {}
        num_cols    = df.select_dtypes(include=[np.number]).columns
        cat_cols    = df.select_dtypes(exclude=[np.number]).columns

        # Impute numerics with median
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())

        # Impute and encode categoricals
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = {
                'classes': le.classes_.tolist(),
                'mapping': {str(c): int(i) for i, c in enumerate(le.classes_)}
            }

        SESSION_STORE[session_id]['processed_df'] = df
        SESSION_STORE[session_id]['encoders']     = encoders

        return jsonify({
            'status':        'success',
            'shape':         list(df.shape),
            'columns':       df.columns.tolist(),
            'encoders':      encoders,
            'preview':       df.head(10).to_dict(orient='records'),
            'missing_after': int(df.isnull().sum().sum()),
        })

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ══════════════════════════════════════════════════════════════════════════════
# API: EDA — correlation heatmap + scatter
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/eda', methods=['POST'])
def eda():
    """
    JS equivalent: Chart.js scatter plots (browser-side)
    Python:        Matplotlib heatmap + scatter saved as PNG
    """
    try:
        data       = request.get_json()
        session_id = data.get('session_id')
        target_col = data.get('target_col')

        if session_id not in SESSION_STORE:
            return jsonify({'error': 'Session not found.'}), 400

        df = SESSION_STORE[session_id].get('processed_df')
        if df is None:
            return jsonify({'error': 'Please preprocess data first.'}), 400

        numeric_df = df.select_dtypes(include=[np.number])
        corr       = numeric_df.corr()

        # ── Correlation heatmap ────────────────────────────────────────────────
        n   = len(corr)
        fig = plt.figure(figsize=(max(8, n * 0.9), max(6, n * 0.8)),
                         facecolor=BG_DARK)
        ax  = fig.add_subplot(111)
        im  = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1, aspect='auto')
        ax.set_xticks(range(n));  ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(n));  ax.set_yticklabels(corr.columns, fontsize=9)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr.values[i, j]:.2f}",
                        ha='center', va='center', fontsize=8,
                        color='white' if abs(corr.values[i, j]) > 0.5 else '#888')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors='#8a8db0')
        ax.set_title('Correlation Heatmap', pad=15, color='white', fontsize=14, fontweight='bold')
        fig.tight_layout()
        heatmap_url = _save_fig(fig, 'heatmap')

        plots = [{'title': 'Correlation Heatmap', 'url': heatmap_url}]

        # ── Scatter: each feature vs target ───────────────────────────────────
        if target_col and target_col in df.columns:
            feature_cols = [c for c in numeric_df.columns if c != target_col]
            nf = len(feature_cols)
            if nf > 0:
                ncols = min(3, nf)
                nrows = (nf + ncols - 1) // ncols
                fig2, axes = plt.subplots(nrows, ncols,
                                          figsize=(5 * ncols, 4 * nrows),
                                          facecolor=BG_DARK)
                axes = np.array(axes).flatten()
                colors_cycle = [ACCENT, ACCENT2, ACCENT3, '#f9c74f', '#90be6d']
                for idx, feat in enumerate(feature_cols):
                    ax2 = axes[idx]
                    c   = colors_cycle[idx % len(colors_cycle)]
                    ax2.scatter(df[feat], df[target_col],
                                alpha=0.65, s=30, color=c, edgecolors='none')
                    # Regression line overlay
                    m, b = np.polyfit(df[feat].values, df[target_col].values, 1)
                    xs   = np.linspace(df[feat].min(), df[feat].max(), 200)
                    ax2.plot(xs, m * xs + b, color='white', linewidth=1.5, alpha=0.8)
                    ax2.set_xlabel(feat);  ax2.set_ylabel(target_col)
                    ax2.set_title(f'{feat} vs {target_col}', color='white', fontsize=11)
                    ax2.grid(True, alpha=0.3)
                for extra in range(idx + 1, len(axes)):
                    axes[extra].set_visible(False)
                fig2.tight_layout(pad=2)
                scatter_url = _save_fig(fig2, 'scatter')
                plots.append({'title': 'Feature vs Target Scatter Plots', 'url': scatter_url})

        # Correlation values with target
        corr_with_target = {}
        if target_col and target_col in corr:
            corr_with_target = {
                k: round(float(v), 4)
                for k, v in corr[target_col].drop(target_col, errors='ignore').items()
            }

        return jsonify({'status': 'success', 'plots': plots, 'corr_with_target': corr_with_target})

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ══════════════════════════════════════════════════════════════════════════════
# API: Train
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/train', methods=['POST'])
def train():
    """
    JS equivalent: manually computing β = (XᵀX)⁻¹Xᵀy or gradient descent loop
    Python:        sklearn.LinearRegression.fit() — uses LAPACK DGELS (optimized QR)
                   with optional StandardScaler for feature scaling
    """
    try:
        data         = request.get_json()
        session_id   = data.get('session_id')
        feature_cols = data.get('feature_cols', [])
        target_col   = data.get('target_col')
        test_size    = float(data.get('test_size', 0.2))
        scale        = bool(data.get('scale_features', True))
        cv_folds     = int(data.get('cv_folds', 5))

        if session_id not in SESSION_STORE:
            return jsonify({'error': 'Session not found.'}), 400
        if not feature_cols or not target_col:
            return jsonify({'error': 'Please select feature(s) and target column.'}), 400

        df = SESSION_STORE[session_id].get('processed_df')
        if df is None:
            return jsonify({'error': 'Please preprocess data first.'}), 400

        missing_cols = [c for c in feature_cols + [target_col] if c not in df.columns]
        if missing_cols:
            return jsonify({'error': f'Columns not found: {missing_cols}'}), 400

        X = df[feature_cols].values
        y = df[target_col].values

        if len(X) < 10:
            return jsonify({'error': 'Need at least 10 rows to train.'}), 400

        # ── Train/test split ───────────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # ── Optional feature scaling ───────────────────────────────────────────
        scaler = None
        if scale:
            scaler  = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test  = scaler.transform(X_test)

        # ── Fit model ─────────────────────────────────────────────────────────
        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        # ── Metrics ───────────────────────────────────────────────────────────
        mse    = float(mean_squared_error(y_test, y_pred))
        rmse   = float(np.sqrt(mse))
        mae    = float(mean_absolute_error(y_test, y_pred))
        r2     = float(r2_score(y_test, y_pred))
        adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - len(feature_cols) - 1)

        # ── Cross-validation ──────────────────────────────────────────────────
        cv_scores = []
        if scale:
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import StandardScaler as SS
            pipe = Pipeline([('sc', SS()), ('lr', LinearRegression())])
            cv_scores = cross_val_score(pipe, X, y, cv=cv_folds, scoring='r2').tolist()
        else:
            cv_scores = cross_val_score(model, X, y, cv=cv_folds, scoring='r2').tolist()

        # ── Store session data ─────────────────────────────────────────────────
        SESSION_STORE[session_id].update({
            'model':        model,
            'scaler':       scaler,
            'feature_cols': feature_cols,
            'target_col':   target_col,
            'X_test':       X_test,
            'y_test':       y_test.tolist(),
            'y_pred':       y_pred.tolist(),
        })

        # ── Coefficients ──────────────────────────────────────────────────────
        coefficients = {
            feat: round(float(coef), 6)
            for feat, coef in zip(feature_cols, model.coef_)
        }

        # ── Plots ─────────────────────────────────────────────────────────────
        plots = []

        # 1. Regression line (only for single-feature)
        if len(feature_cols) == 1:
            feat  = feature_cols[0]
            x_all = df[feat].values
            y_all = df[target_col].values

            fig, ax = plt.subplots(figsize=(8, 5), facecolor=BG_DARK)
            ax.scatter(x_all, y_all, alpha=0.6, s=35, color=ACCENT, edgecolors='none', label='Data')

            # Regression line (unscaled for display)
            xs = np.linspace(x_all.min(), x_all.max(), 300)
            if scale:
                xs_scaled = scaler.transform(xs.reshape(-1, 1))
                ys = model.predict(xs_scaled)
            else:
                ys = model.predict(xs.reshape(-1, 1))
            ax.plot(xs, ys, color=ACCENT2, linewidth=2.5, label='Regression Line')
            ax.set_xlabel(feat);  ax.set_ylabel(target_col)
            ax.set_title(f'Linear Regression: {feat} → {target_col}',
                         color='white', fontsize=13, fontweight='bold')
            ax.legend(facecolor='#1a1d2e', edgecolor='#3a3d5c', labelcolor='white')
            ax.grid(True, alpha=0.3)
            plots.append({'title': 'Regression Line', 'url': _save_fig(fig, 'regression')})

        # 2. Predicted vs Actual
        fig2, ax2 = plt.subplots(figsize=(7, 5), facecolor=BG_DARK)
        ax2.scatter(y_test, y_pred, alpha=0.65, s=35, color=ACCENT3, edgecolors='none')
        mn, mx = min(min(y_test), min(y_pred)), max(max(y_test), max(y_pred))
        ax2.plot([mn, mx], [mn, mx], color=ACCENT2, linewidth=2, linestyle='--', label='Perfect prediction')
        ax2.set_xlabel('Actual');  ax2.set_ylabel('Predicted')
        ax2.set_title('Predicted vs Actual', color='white', fontsize=13, fontweight='bold')
        ax2.legend(facecolor='#1a1d2e', edgecolor='#3a3d5c', labelcolor='white')
        ax2.grid(True, alpha=0.3)
        plots.append({'title': 'Predicted vs Actual', 'url': _save_fig(fig2, 'pred_actual')})

        # 3. Residuals plot
        residuals = np.array(y_pred) - np.array(y_test)
        fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5), facecolor=BG_DARK)
        axes3[0].scatter(y_pred, residuals, alpha=0.65, s=30, color=ACCENT, edgecolors='none')
        axes3[0].axhline(0, color=ACCENT2, linewidth=1.5, linestyle='--')
        axes3[0].set_xlabel('Predicted');  axes3[0].set_ylabel('Residuals')
        axes3[0].set_title('Residual Plot', color='white', fontweight='bold')
        axes3[0].grid(True, alpha=0.3)
        axes3[1].hist(residuals, bins=25, color=ACCENT, edgecolor='none', alpha=0.8)
        axes3[1].axvline(0, color=ACCENT2, linewidth=1.5, linestyle='--')
        axes3[1].set_xlabel('Residual');  axes3[1].set_ylabel('Count')
        axes3[1].set_title('Residual Distribution', color='white', fontweight='bold')
        axes3[1].grid(True, alpha=0.3)
        fig3.tight_layout()
        plots.append({'title': 'Residual Analysis', 'url': _save_fig(fig3, 'residuals')})

        # 4. Feature coefficients bar chart
        if len(feature_cols) > 1:
            fig4, ax4 = plt.subplots(figsize=(max(7, len(feature_cols) * 0.9), 5), facecolor=BG_DARK)
            colors = [ACCENT if c >= 0 else ACCENT2 for c in model.coef_]
            bars   = ax4.bar(feature_cols, model.coef_, color=colors, edgecolor='none')
            ax4.axhline(0, color='white', linewidth=0.8, alpha=0.5)
            ax4.set_xlabel('Feature');  ax4.set_ylabel('Coefficient')
            ax4.set_title('Feature Coefficients', color='white', fontsize=13, fontweight='bold')
            ax4.grid(True, axis='y', alpha=0.3)
            for bar, val in zip(bars, model.coef_):
                ax4.text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() + (abs(model.coef_).max() * 0.02),
                         f'{val:.3f}', ha='center', va='bottom', fontsize=9, color='white')
            plt.xticks(rotation=30, ha='right')
            fig4.tight_layout()
            plots.append({'title': 'Feature Coefficients', 'url': _save_fig(fig4, 'coeff')})

        return jsonify({
            'status':        'success',
            'metrics': {
                'mse':    round(mse, 4),
                'rmse':   round(rmse, 4),
                'mae':    round(mae, 4),
                'r2':     round(r2, 4),
                'adj_r2': round(adj_r2, 4),
            },
            'cv_scores':     [round(s, 4) for s in cv_scores],
            'cv_mean':       round(float(np.mean(cv_scores)), 4),
            'cv_std':        round(float(np.std(cv_scores)), 4),
            'intercept':     round(float(model.intercept_), 6),
            'coefficients':  coefficients,
            'n_train':       int(len(y_train)),
            'n_test':        int(len(y_test)),
            'plots':         plots,
        })

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ══════════════════════════════════════════════════════════════════════════════
# API: Predict
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/predict', methods=['POST'])
def predict():
    """
    JS equivalent: manually computing ŷ = β₀ + β₁x₁ + … in browser
    Python:        model.predict(scaler.transform(X_new))
    Returns step-by-step breakdown for educational display.
    """
    try:
        data       = request.get_json()
        session_id = data.get('session_id')
        input_vals = data.get('input_values', {})

        if session_id not in SESSION_STORE:
            return jsonify({'error': 'Session not found.'}), 400

        sess = SESSION_STORE[session_id]
        if sess['model'] is None:
            return jsonify({'error': 'Train the model first.'}), 400

        model        = sess['model']
        scaler       = sess['scaler']
        feature_cols = sess['feature_cols']

        # Parse and validate inputs
        try:
            X_new = np.array([[float(input_vals[f]) for f in feature_cols]])
        except (KeyError, ValueError) as e:
            return jsonify({'error': f'Invalid input value: {e}'}), 400

        X_input = X_new.copy()
        if scaler is not None:
            X_input = scaler.transform(X_new)

        prediction = float(model.predict(X_input)[0])

        # Step-by-step breakdown (using unscaled coefficients logic for display)
        steps = []
        steps.append({'step': 'Start with intercept', 'value': round(float(model.intercept_), 4)})
        running = float(model.intercept_)
        for feat, raw_val, coef in zip(feature_cols, X_new[0], model.coef_):
            scaled_val = float(X_input[0][feature_cols.index(feat)])
            contrib    = float(coef * scaled_val)
            running   += contrib
            steps.append({
                'step':        f'+ {coef:.4f} × {feat}({raw_val:.2f})',
                'value':       round(running, 4),
                'contribution': round(contrib, 4),
            })

        return jsonify({
            'prediction': round(prediction, 4),
            'steps':      steps,
        })

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ══════════════════════════════════════════════════════════════════════════════
# API: Evaluate (returns stored metrics + evaluation plot)
# ══════════════════════════════════════════════════════════════════════════════
@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    """Returns stored evaluation data and a formatted metrics summary plot."""
    try:
        data       = request.get_json()
        session_id = data.get('session_id')

        if session_id not in SESSION_STORE:
            return jsonify({'error': 'Session not found.'}), 400

        sess = SESSION_STORE[session_id]
        if sess['model'] is None:
            return jsonify({'error': 'Train the model first.'}), 400

        y_test = np.array(sess['y_test'])
        y_pred = np.array(sess['y_pred'])

        mse    = float(mean_squared_error(y_test, y_pred))
        rmse   = float(np.sqrt(mse))
        mae    = float(mean_absolute_error(y_test, y_pred))
        r2     = float(r2_score(y_test, y_pred))
        n      = len(y_test)
        k      = len(sess['feature_cols'])
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

        # Comparison table (first 20 rows)
        comparison = [
            {'actual': round(float(a), 4), 'predicted': round(float(p), 4),
             'residual': round(float(p - a), 4)}
            for a, p in zip(y_test[:20], y_pred[:20])
        ]

        # Metrics bar chart
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=BG_DARK)
        metric_names = ['MSE', 'RMSE', 'MAE']
        metric_vals  = [mse, rmse, mae]
        try:
            bars = axes[0].bar(metric_names, metric_vals,
                           color=[ACCENT, ACCENT2, ACCENT3], edgecolor='none')
            axes[0].set_title('Error Metrics', color='white', fontweight='bold')
            axes[0].set_yscale('log')
            axes[0].set_ylabel('Log Scale', color='white')
            axes[0].grid(True, axis='y', alpha=0.3)
            for bar, val in zip(bars, metric_vals):
                axes[0].text(bar.get_x() + bar.get_width() / 2,
                         bar.get_height() * 1.02, f'{val:.2f}',
                         ha='center', va='bottom', fontsize=9, color='white')
        except Exception as e:
            print(e)
        # R² gauge (simple donut)
        sizes  = [max(0, r2), max(0, 1 - r2)]
        colors = [ACCENT, '#2a2d45']
        axes[1].pie(sizes, colors=colors, startangle=90,
                    wedgeprops={'width': 0.45, 'edgecolor': BG_DARK, 'linewidth': 2})
        axes[1].text(0, 0, f'R²\n{r2:.3f}', ha='center', va='center',
                     fontsize=16, fontweight='bold', color='white')
        axes[1].set_title('R² Score', color='white', fontweight='bold')
        fig.tight_layout()
        eval_plot = _save_fig(fig, 'eval_metrics')

        return jsonify({
            'status': 'success',
            'metrics': {
                'mse': round(mse, 4), 'rmse': round(rmse, 4),
                'mae': round(mae, 4), 'r2':   round(r2, 4),
                'adj_r2': round(adj_r2, 4),
            },
            'comparison': comparison,
            'eval_plot':  eval_plot,
        })

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
