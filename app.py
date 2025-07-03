
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve, auc)
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules
import matplotlib.pyplot as plt

st.set_page_config(page_title="VaporIQ Analytics Dashboard", layout="wide")

# ---------- Data Load ----------
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv('Data/vaporiq_synthetic_dataset_10k.csv')

data = load_data()

numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

# ---------- Tabs ----------
pages = st.tabs(["Data Visualization", "Classification",
                 "Clustering", "Association Rules", "Regression"])

# 1. Data Visualization
with pages[0]:
    st.header("📊 Data Visualization")
    st.write("A couple of starter insights:")

    fig1, ax1 = plt.subplots()
    data['age'].hist(bins=20, color='steelblue', edgecolor='white', ax=ax1)
    ax1.set_title("Age Distribution")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.boxplot(data['income'])
    ax2.set_title("Income (with outliers)")
    st.pyplot(fig2)

# 2. Classification
with pages[1]:
    st.header("🤖 Classification")
    feat_default = ['age', 'income', 'monthly_vape_spend']
    features = st.multiselect("Select predictors:", numeric_cols,
                              default=[c for c in feat_default if c in numeric_cols])

    if not features:
        st.warning("Select at least one feature column.")
    else:
        target = 'willingness_to_subscribe'
        X_train, X_test, y_train, y_test = train_test_split(
            data[features], data[target], test_size=0.3, random_state=42, stratify=data[target])

        models = {
            "KNN": KNeighborsClassifier(),
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(random_state=42),
            "Gradient Boosting": GradientBoostingClassifier(random_state=42)
        }

        results = []
        for name, mdl in models.items():
            mdl.fit(X_train, y_train)
            pred = mdl.predict(X_test)
            results.append({
                "Model": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred),
                "Recall": recall_score(y_test, pred),
                "F1": f1_score(y_test, pred)
            })

        st.subheader("Metrics")
        st.table(pd.DataFrame(results).set_index("Model").style.format("{:.2f}"))

        selected = st.selectbox("Confusion matrix of:", list(models.keys()))
        cm = confusion_matrix(y_test, models[selected].predict(X_test))
        st.write(pd.DataFrame(cm, index=["Actual 0","Actual 1"], columns=["Pred 0","Pred 1"]))

        # ROC Curves
        roc_fig, roc_ax = plt.subplots()
        for name, mdl in models.items():
            if hasattr(mdl, "predict_proba"):
                prob = mdl.predict_proba(X_test)[:,1]
                fpr, tpr, _ = roc_curve(y_test, prob)
                roc_ax.plot(fpr, tpr, label=name)
        roc_ax.plot([0,1],[0,1],'k--')
        roc_ax.set_xlabel("False Positive Rate")
        roc_ax.set_ylabel("True Positive Rate")
        roc_ax.legend()
        st.pyplot(roc_fig)

        # Predict new data
        st.markdown("#### Predict New Data")
        up = st.file_uploader("Upload CSV (without target)", type="csv")
        if up is not None:
            new_df = pd.read_csv(up)
            missing = [c for c in features if c not in new_df.columns]
            if missing:
                st.error(f"Missing columns in upload: {missing}")
            else:
                new_df['prediction'] = models['Random Forest'].predict(new_df[features])
                st.write(new_df.head())
                st.download_button("Download predictions",
                                   new_df.to_csv(index=False).encode(),
                                   file_name="predictions.csv",
                                   mime='text/csv')

# 3. Clustering
with pages[2]:
    st.header("📍 Clustering (K‑Means)")
    k = st.slider("Number of clusters (k)", 2, 10, 4, 1)

    km_data = data[numeric_cols]
    kmeans = KMeans(n_clusters=k, random_state=42)
    data['cluster'] = kmeans.fit_predict(km_data)

    # Elbow
    inertias = []
    Ks = range(2, 11)
    for ki in Ks:
        inertias.append(KMeans(n_clusters=ki, random_state=42).fit(km_data).inertia_)
    elbow_fig, elbow_ax = plt.subplots()
    elbow_ax.plot(Ks, inertias, marker='o')
    elbow_ax.set_title("Elbow Method")
    elbow_ax.set_xlabel("k")
    elbow_ax.set_ylabel("Inertia")
    st.pyplot(elbow_fig)

    # Personas using numeric subset
    persona_cols = [c for c in ['age','income','monthly_vape_spend'] if c in numeric_cols]
    persona = data.groupby('cluster')[persona_cols].mean().round(2)
    st.subheader("Cluster Personas")
    st.table(persona)

    st.download_button("Download labelled data",
                       data.to_csv(index=False).encode(),
                       file_name="vaporiq_clustered.csv",
                       mime='text/csv')

# 4. Association Rules
with pages[3]:
    st.header("🔗 Association Rules")
    trans_cols = st.multiselect("Transaction columns", ['liked_flavors','disliked_flavors'],
                                default=['liked_flavors'])
    support = st.slider("Min support", 0.01, 0.5, 0.05, step=0.01)
    confidence = st.slider("Min confidence", 0.1, 0.9, 0.3, step=0.05)

    if trans_cols:
        # One‑hot encode items across selected columns
        onehot = None
        for col in trans_cols:
            ohe = data[col].str.get_dummies(sep=',')
            onehot = ohe if onehot is None else (onehot | ohe)
        freq = apriori(onehot, min_support=support, use_colnames=True)
        rules = (association_rules(freq, metric="confidence", min_threshold=confidence)
                 .sort_values('confidence', ascending=False)
                 .head(10))
        if rules.empty:
            st.info("No rules found with current thresholds.")
        else:
            st.write(rules[['antecedents','consequents','support','confidence','lift']])

# 5. Regression
with pages[4]:
    st.header("📈 Regression")
    reg_feats = st.multiselect("Predictor variables", numeric_cols,
                               default=['income','age','usage_freq_per_week'])
    target_y = st.selectbox("Target", ['monthly_vape_spend','satisfaction_rating'])

    if reg_feats:
        X_train, X_test, y_train, y_test = train_test_split(
            data[reg_feats], data[target_y], test_size=0.3, random_state=42)

        regressors = {
            "Ridge": Ridge(random_state=42),
            "Lasso": Lasso(random_state=42),
            "Decision Tree": DecisionTreeRegressor(random_state=42)
        }
        rows = []
        for name, reg in regressors.items():
            reg.fit(X_train, y_train)
            rows.append({"Model": name, "R²": reg.score(X_test, y_test)})
        st.table(pd.DataFrame(rows).set_index("Model").style.format("{:.2f}"))
