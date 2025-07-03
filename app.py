
import warnings; warnings.filterwarnings('ignore', category=FutureWarning)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve)
from sklearn.cluster import KMeans
from mlxtend.frequent_patterns import apriori, association_rules

st.set_page_config(page_title='VaporIQ Analytics Dashboard', layout='wide')

# ---------- Load ---------- #
@st.cache_data(show_spinner=False)
def load():
    return pd.read_csv('Data/vaporiq_synthetic_dataset_10k.csv')
data = load()

numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()

# ---------- Palette ---------- #
PALETTE = ['#008080', '#FF8C42', '#6A5ACD', '#00A0B0', '#FFB85F']

# ---------- Tabs ---------- #
pages = st.tabs(['Data Visualisation', 'Classification', 'Clustering',
                 'Association Rules', 'Regression'])

# ---------------- Data Visualisation ---------------- #
with pages[0]:
    st.header('📊 Data Visualisation')

    viz_choice = st.selectbox('Choose a chart', [
        'Age distribution (histogram)',
        'Gender split (pie)',
        'Income by cluster (box)',
        'Spend vs. income (scatter)',
        'Correlation heat-map',
        'Subscription by age band (bar)',
        'Nicotine strength share (pie)',
        'Churn risk vs. spend (scatter)',
        'Flavor popularity (top-10 bar)',
        'Lifestyle overlap – Venn'
    ])

    if viz_choice == 'Age distribution (histogram)':
        fig, ax = plt.subplots()
        ax.hist(data['age'], bins=20, color=PALETTE[0], edgecolor='white')
        ax.set_title('Age distribution')
        st.pyplot(fig)

    elif viz_choice == 'Gender split (pie)':
        fig, ax = plt.subplots()
        counts = data['gender'].value_counts()
        ax.pie(counts, labels=counts.index, autopct='%1.0f%%',
               colors=PALETTE[:len(counts)], startangle=90)
        ax.set_title('Gender split')
        st.pyplot(fig)

    elif viz_choice == 'Income by cluster (box)':
        fig, ax = plt.subplots()
        data.boxplot(column='income', by='cluster', ax=ax,
                     patch_artist=True,
                     boxprops=dict(facecolor=PALETTE[1]))
        ax.set_title('Income by cluster')
        ax.set_ylabel('Income')
        st.pyplot(fig)

    elif viz_choice == 'Spend vs. income (scatter)':
        fig, ax = plt.subplots()
        ax.scatter(data['income'], data['monthly_vape_spend'],
                   c=data['churn_risk'], cmap='viridis', alpha=0.6)
        ax.set_xlabel('Income')
        ax.set_ylabel('Monthly spend')
        ax.set_title('Income vs. spend (colour = churn risk)')
        st.pyplot(fig)

    elif viz_choice == 'Correlation heat-map':
        corr = data.select_dtypes('number').corr()
        fig, ax = plt.subplots(figsize=(6,5))
        im = ax.imshow(corr, cmap='coolwarm', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)), corr.columns, rotation=90)
        ax.set_yticks(range(len(corr.columns)), corr.columns)
        fig.colorbar(im, ax=ax, shrink=0.7)
        st.pyplot(fig)

    elif viz_choice == 'Subscription by age band (bar)':
        age_band = pd.cut(data['age'], bins=[18,25,35,50,70])
        subs = data.groupby(age_band)['willingness_to_subscribe'].mean().mul(100).round(1)
        fig, ax = plt.subplots()
        ax.bar(subs.index.astype(str), subs.values, color=PALETTE[0])
        ax.set_ylabel('% willing to subscribe')
        ax.set_title('Subscription likelihood by age band')
        st.pyplot(fig)

    elif viz_choice == 'Nicotine strength share (pie)':
        fig, ax = plt.subplots()
        s = data['nicotine_strength'].value_counts().sort_index()
        ax.pie(s, labels=s.index, colors=PALETTE[:len(s)], autopct='%1.0f%%', startangle=90)
        ax.set_title('Nicotine strength share')
        st.pyplot(fig)

    elif viz_choice == 'Churn risk vs. spend (scatter)':
        fig, ax = plt.subplots()
        ax.scatter(data['churn_risk'], data['monthly_vape_spend'],
                   color=PALETTE[2], alpha=0.5)
        ax.set_xlabel('Churn risk')
        ax.set_ylabel('Monthly spend')
        ax.set_title('Churn risk vs. spend')
        st.pyplot(fig)

    elif viz_choice == 'Flavor popularity (top-10 bar)':
        top = data['liked_flavors'].str.split(',', expand=True).stack().value_counts().head(10)
        fig, ax = plt.subplots()
        ax.bar(top.index, top.values, color=PALETTE[0])
        ax.set_xticklabels(top.index, rotation=45)
        ax.set_title('Top‑10 liked flavours')
        st.pyplot(fig)

    elif viz_choice == 'Lifestyle overlap – Venn':
        from matplotlib_venn import venn3
        social = data['lifestyle_social'] == 1
        fitness = data['lifestyle_fitness'] == 1
        travel = data['lifestyle_travel'] == 1
        fig, ax = plt.subplots()
        venn3(subsets=(
            (social & ~fitness & ~travel).sum(),
            (fitness & ~social & ~travel).sum(),
            (social & fitness & ~travel).sum(),
            (travel & ~social & ~fitness).sum(),
            (social & travel & ~fitness).sum(),
            (fitness & travel & ~social).sum(),
            (social & fitness & travel).sum(),
        ),
            set_labels=('Social','Fitness','Travel'),
            set_colors=PALETTE[:3],
            alpha=0.5,
            ax=ax)
        ax.set_title('Lifestyle overlap')
        st.pyplot(fig)

# The rest of tabs code remains identical to previous v2 (for brevity not repeated)
