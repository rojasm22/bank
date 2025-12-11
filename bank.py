# This file must be run in the command line with: streamlit run myst_v4.py

import streamlit as st
from joblib import load
import numpy as np
import pandas as pd
import os
from PIL import Image


st.set_page_config(page_title="Bank Prediction", layout="centered")

# ------------------------------------------------------
#  GLOBAL CONFIG & DICTIONARIES
# ------------------------------------------------------
LABELS_MAP = {
    0: "No",
    1: "Yes",
    "0": "No",
    "1": "Yes",
    "no": "No",
    "yes": "Yes"
}

TOOLTIPS = {
    "age": "Age in years",
    "job": "Type of job",
    "marital": "Marital status",
    "education": "Education level",
    "default": "Has credit in default?",
    "balance": "Average yearly balance",
    "housing": "Has a housing loan?",
    "loan": "Has a personal loan?",
    "contact": "Contact communication type",
    "day": "Last contact day of the month",
    "day_of_week": "Last contact day of the week",
    "month": "Last contact month",
    "duration": "Last contact duration, in seconds (The 'Leakage' Variable)",
    "campaign": "Number of contacts performed during this campaign",
    "pdays": "Days passed after last contact (-1 if no contact)",
    "previous": "Number of contacts performed before this campaign",
    "poutcome": "Outcome of the previous marketing campaign"
}

# ------------------------------------------------------
#  HELPER FUNCTIONS
# ------------------------------------------------------
def feature_engineering_honest(df):
    X = df.copy()
    if 'duration' in X.columns:
        X = X.drop(columns=['duration'])
    return X

@st.cache_resource
def load_packs():
    p_std = load("pack_streamlit.joblib")
    p_hst = load("pack_streamlit_honest.joblib")
    return p_std, p_hst

try:
    pack, pack_honest = load_packs()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# --- Load Models ---
model = pack["model"]
num_cols = pack["num_cols"]
cat_cols = pack["cat_cols"]
num_summary = pack["num_summary"]
cat_options = pack["cat_options"]
classes_ = pack["classes_"]
acc = pack.get("accuracy", None)

model_honest = pack_honest["model"]
acc_h = pack_honest.get("accuracy", None)


# ======================================================
#  APP NAVIGATION
# ======================================================
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to:",
    ["Predictions", "Glossary", "Models & Math"]
)

# ======================================================
#  PAGE 1: PREDICTIONS
# ======================================================
if page == "Predictions":
    st.title("Prediction App: Standard vs Honest")
    
    # --- Show accuracies ---
    st.markdown("### Model Performance (Test Set)")
    col_acc1, col_acc2 = st.columns(2)
    if acc: 
        col_acc1.metric("Standard Accuracy", f"{acc:.3f}")
    if acc_h: 
        col_acc2.metric("Honest Accuracy", f"{acc_h:.3f}")

    st.markdown("---")
    st.markdown("Input values and press **Predict**:")

    # --- Form ---
    with st.form("form"):
        st.subheader("Numerical features")
        num_inputs = {}
        
        for c in num_cols:
            p1 = num_summary[c]["p1"]
            p99 = num_summary[c]["p99"]
            med = num_summary[c]["median"]
            
            label = c
            if c == 'duration':
                label = f"{c} (Used only by Standard)"
            
            help_text = TOOLTIPS.get(c, "")
            raw = st.text_input(label, value=str(med), help=help_text)

            if raw.strip() == "":
                num_inputs[c] = np.nan
            else:
                try:
                    num_inputs[c] = float(raw)
                except:
                    num_inputs[c] = np.nan

        st.subheader("Categorical features")
        cat_inputs = {}
        for c in cat_cols:
            opts = cat_options[c]
            help_text = TOOLTIPS.get(c, "")
            cat_inputs[c] = st.selectbox(c, opts, index=0, help=help_text)

        submitted = st.form_submit_button("Predict with Both Models")

    # --- Prediction Logic ---
    if submitted:
        data = {**num_inputs, **cat_inputs}
        X_one = pd.DataFrame([data], columns=num_cols + cat_cols)

        st.divider()
        st.subheader("Prediction Results: Did the client subscribe?")

        col_std, col_hst = st.columns(2)

        # Standard
        with col_std:
            st.markdown("#### Standard Model")
            st.caption("Uses 'duration'.")
            try:
                proba = model.predict_proba(X_one)[0]
                pred_raw = classes_[int(np.argmax(proba))]
                pred_text = LABELS_MAP.get(pred_raw, str(pred_raw))
                
                st.success(f"Subscribed? **{pred_text}**")
                
                st.write("**Probabilities:**")
                for cls, p in zip(classes_, proba):
                    cls_text = LABELS_MAP.get(cls, str(cls))
                    st.write(f"- {cls_text}: {p*100:.1f}%")
            except Exception as e:
                st.error(f"Error: {e}")

        # Honest
        with col_hst:
            st.markdown("#### Honest Model")
            st.caption("Ignores 'duration'.")
            try:
                proba_h = model_honest.predict_proba(X_one)[0]
                pred_h_raw = classes_[int(np.argmax(proba_h))]
                pred_h_text = LABELS_MAP.get(pred_h_raw, str(pred_h_raw))
                
                st.info(f"Subscribed? **{pred_h_text}**")
                
                st.write("**Probabilities:**")
                for cls, p in zip(classes_, proba_h):
                    cls_text = LABELS_MAP.get(cls, str(cls))
                    st.write(f"- {cls_text}: {p*100:.1f}%")
            except Exception as e:
                st.error(f"Error: {e}")

# ======================================================
#  PAGE 2: GLOSSARY
# ======================================================
elif page == "Glossary":
    st.title("Feature Glossary")
    st.markdown("Explanation of the variables used in the models.")

    # Create a nice dataframe for the glossary
    glossary_data = [{"Variable": k, "Description": v} for k, v in TOOLTIPS.items()]
    df_glossary = pd.DataFrame(glossary_data)
    
    # Display as a clean table
    st.table(df_glossary)

# ======================================================
#  PAGE 3: MODELS & MATH
# ======================================================
elif page == "Models & Math":
    st.title("Technical Methodology & Analysis")

    # --- 1. MODEL SELECTION (New Narrative) ---
    st.header("1. Model Selection Protocol")
    st.markdown("""
    Prior to finalizing the predictive engine, a rigorous **benchmarking study** was conducted involving 20 distinct model configurations. 
    The experimental design covered a spectrum of algorithms including:
    * **Linear Models:** Logistic Regression.
    * **Distance-based Algorithms:** K-Nearest Neighbors (KNN).
    * **Ensemble Methods:** Random Forest, SVM, and XGBoost.
    """)
    
    st.markdown("""
    Each algorithm was evaluated in conjunction with various encoding strategies (Ordinal, Fixed Effects, Cyclical) 
    and hyperparameter optimization techniques (GridSearch, Randomized Search, Optuna).
    """)

    st.info("""
    **Conclusion:** While linear models plateaued at ~84% accuracy, tree-based ensemble methods demonstrated superior capacity to capture non-linear relationships. 
    **XGBoost optimized with Optuna** emerged as the statistical victor (F1-Score: 0.8792), validating the trade-off between higher computational cost and predictive precision.
    """)

    st.divider()

    # --- 2. XGBoost Introduction ---
    st.header("2. Algorithmic Foundation: XGBoost")
    st.markdown("""
    XGBoost (Extreme Gradient Boosting) is an ensemble method that constructs a series of Decision Trees sequentially. 
    Unlike Random Forest, where trees are independent, XGBoost builds each new tree $f_t$ to correct the residual errors 
    of the previous ensemble.
    """)
    st.markdown("[Reference: Chen & Guestrin, 2016](https://arxiv.org/abs/1603.02754)")

    st.subheader("A. The Objective Function")
    st.markdown("At iteration $t$, the algorithm minimizes the following regularized objective function:")
    st.latex(r'''
    \mathcal{L}^{(t)} = \sum_{i=1}^{n} l\left(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)
    ''')
    st.markdown("""
    Where $l$ is the loss function and $\Omega(f_t)$ is the regularization term:
    """)
    st.latex(r'''
    \Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2
    ''')

    st.subheader("B. Second-Order Approximation")
    st.markdown("""
    Optimization is achieved via a second-order Taylor expansion using the gradient ($g_i$) and the hessian ($h_i$):
    """)
    st.latex(r'''
    \mathcal{L}^{(t)} \approx \sum_{i=1}^{n} \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)
    ''')

    st.subheader("C. Optimal Weights & Gain")
    st.markdown("The optimal leaf weight $w_j^*$ and the split gain (reduction in loss) are given by:")
    st.latex(r'''
    w_j^* = -\frac{G_j}{H_j + \lambda}, \quad \text{Gain} = \frac{1}{2} \left[ \frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{(G_L + G_R)^2}{H_L + H_R + \lambda} \right] - \gamma
    ''')

    st.divider()

    # --- 3. Preprocessing Strategy ---
    st.header("3. Preprocessing & Feature Engineering")
    st.markdown("The data pipeline was constructed to handle heterogeneous data types while minimizing statistical bias.")

    st.subheader("Imputation Strategy: The 'Marital' Case")
    st.markdown("""
    Categorical variables were imputed using the mode (most frequent value). Specifically, the `marital` variable presented 
    532 missing observations ($N_{missing} = 532$). 
    Given that the total dataset size is significant (approx. 11,000 observations), these missing values represent only 
    **4.8%** of the population. Consequently, mode imputation is statistically safe as it does not introduce 
    significant bias into the distribution.
    """)

    st.subheader("Feature Extraction: 'Pdays'")
    st.markdown("""
    The `pdays` variable (days since last contact) exhibits a mixed distribution: it contains positive integers 
    for contacted clients and the value `-1` for those never contacted. 
    To allow the model to interpret this correctly, a **Feature Extraction** step was applied:
    * **Indicator Feature:** A binary column created via `MissingIndicator` to explicitly flag `-1` values (Never Contacted).
    * **Value Feature:** The numerical value itself, processed by `StandardScaler`.
    """)

    st.subheader("Temporal Encoding: The 'Month' Variable")
    st.markdown("""
    The `month` feature required careful consideration due to its cyclicity. Three methods were rigorously tested:
    1.  **Cyclical Encoding:** Transformation using Sine/Cosine functions.
    2.  **Fixed Effects:** One-Hot Encoding.
    3.  **Ordinal Encoding:** Mapping months to integers (0-11).
    
    **Result:** Ordinal Encoding proved to be the superior method. While it theoretically ignores the cyclical 
    nature (Dec to Jan), it preserves the temporal hierarchy without the sparsity of One-Hot Encoding, 
    allowing XGBoost to find efficient splits.
    """)

    st.divider()

    # --- 4. Feature Selection & Optimization ---
    st.header("4. Optimization & Feature Analysis")

    st.subheader("Bayesian Hyperparameter Tuning & Feature Importance")
    st.markdown("""
    We utilized **Optuna** (Tree-structured Parzen Estimator) to navigate the high-dimensional hyperparameter space. 
    The resulting configurations reveal distinct structural differences between the models.
    """)

    # Columns for metrics and feature importance images
    col_std, col_hon = st.columns(2)
    
    # --- STANDARD MODEL (LEAKAGE) ---
    with col_std:
        st.markdown("**A. Standard Model (Benchmark)**")
        st.caption("Deep tree structure optimized for complex decision boundaries.")
        st.json({
            'n_estimators': 1000, 
            'learning_rate': 0.0129, 
            'max_depth': 6,
            'min_child_weight': 1,
            'gamma': 0.2012,
            'subsample': 0.8021
        })
        st.metric("Accuracy", "0.8795")
        st.metric("F1 Score", "0.8792")
        st.metric("ROC AUC", "0.9812")
        
        st.markdown("**Feature Analysis:**")
        st.markdown("""
        The influence of `cat__poutcome_success` is dominant (0.182), but `num__duration` ranks significantly high 
        at 4th place (0.084). The model relies on the interaction between past success and the current call length 
        to maximize accuracy, confirming the leakage hypothesis.
        """)
        
        if os.path.exists("feature_leakage.png"):
            st.image("feature_leakage.png", use_container_width=True)
        else:
            st.warning("File 'feature_leakage.png' not found.")

    # --- HONEST MODEL ---
    with col_hon:
        st.markdown("**B. Honest Model (Production)**")
        st.caption("Conservative structure with higher regularization to prevent overfitting.")
        st.json({
            'n_estimators': 500, 
            'learning_rate': 0.0277, 
            'max_depth': 4,
            'min_child_weight': 1,
            'gamma': 0.1514,
            'subsample': 0.7689
        })
        st.metric("Accuracy", "0.7705")
        st.metric("F1 Score", "0.7377")
        st.metric("ROC AUC", "0.8645")

        st.markdown("**Feature Analysis:**")
        st.markdown("""
        With `duration` removed, the model concentrates heavily on historical data. 
        `cat__poutcome_success` becomes the overwhelming driver (0.377), followed by `cat__marital_None` (0.127). 
        This shift indicates the model is predicting based on client relationship history rather than immediate call characteristics.
        """)
        
        if os.path.exists("feature_honest.png"):
            st.image("feature_honest.png", use_container_width=True)
        else:
            st.warning("File 'feature_honest.png' not found.")

    # --- ROC CURVE ---
    st.markdown("---")
    st.subheader("Comparative Evaluation: ROC Curve")
    st.markdown("""
    The Receiver Operating Characteristic (ROC) curve illustrates the diagnostic ability of the classifier. 
    The AUC (Area Under Curve) serves as the primary metric for separability.
    """)
    
    if os.path.exists("ROC_curve.png"):
        st.image("ROC_curve.png", caption="Figure 3: ROC Curve Comparison. The Standard Model (Left) shows inflated performance (AUC=0.98) due to leakage, while the Honest Model (Right) reflects realistic predictive power (AUC=0.86).", use_container_width=True)
    else:
        st.warning("File 'ROC_curve.png' not found.")

    st.divider()

    # --- 5. The Leakage Analysis ---
    st.header("5. Target Leakage & Operational Utility")
    
    st.markdown("During feature engineering, a critical anomaly was identified in the `duration` variable.")

    st.subheader("The Theoretical Ceiling (Benchmark)")
    st.markdown("""
    The variable `duration` is not known *ex-ante* (before the call occurs). Including it creates **Target Leakage**, 
    as a `duration = 0` implies a `target = 0` deterministically.
    
    Therefore, the Standard Model functions as a **Theoretical Benchmark**: it represents the maximum 
    information capacity of the dataset (~88% Accuracy) if we had perfect hindsight.
    """)

    st.subheader("Operational Value: Opportunity Cost Analysis")
    st.markdown("""
    While the Standard Model is invalid for lead scoring, it provides critical insights for **Resource Allocation**. 
    It allows us to model the *Conditional Probability of Subscription given Duration*.
    
    **Example Scenario:**
    If the model indicates that the probability of success plateaus at $t = 300s$ (5 minutes), 
    extending a call to $t = 1200s$ (20 minutes) is irrational.
    
    * **Outlier Risk:** Extremely long calls are often outliers that do not guarantee conversion.
    * **Opportunity Cost:** In the 15 minutes wasted on a single non-converting long call, 
        an agent could have contacted 3 distinct clients (5 minutes each), statistically maximizing the expected conversion rate.
    """)