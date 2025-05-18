import pandas as pd
import numpy as np
import streamlit as st

# Data preparation
data = {
    'Provider': ['Azure', 'Amazon', 'Google', 'Private'],
    'Cost': [5, 3, 6, 4],
    'ExecutionTime': [2, 1, 8, 5],
    'Security': [1, 2, 3, 4],         # Security levels
    'IT': [1, 2, 3, 4],               # IT support levels
    'Services': [1, 2, 1, 1],         # Services offered
    'DataCompliance': [1, 2, 1, 2],   # Compliance Level
    'MigrationEase': [3, 2, 1, 4],    # Migration ease
    'Location': [1, 2, 3, 4],         # Location codes
    'InfraType': [1, 1, 1, 2]         # Infrastructure type
}
train_df = pd.DataFrame(data)

# --- Manual Cosine Similarity for Educational Purposes ---
def manual_cosine_similarity(v1, v2):
    dot_product = sum(a * b for a, b in zip(v1, v2))
    norm_a = np.sqrt(sum(a * a for a in v1))
    norm_b = np.sqrt(sum(b * b for b in v2))
    if norm_a == 0 or norm_b == 0:
        return 0
    return dot_product / (norm_a * norm_b)

def recommend_cf(client_features):
    best_score = -1
    best_provider = None
    for idx, row in train_df.iterrows():
        provider_features = row.drop(labels=['Provider']).values.tolist()
        score = manual_cosine_similarity(client_features, provider_features)
        if score > best_score:
            best_score = score
            best_provider = row['Provider']
    return best_provider

# --- Streamlit UI ---
st.set_page_config(page_title="Cloud Provider Recommender", layout="centered")
st.title("📊 Cloud Provider Recommender")

st.write("Choisissez les critères du client pour recommander un fournisseur cloud:")

# Inputs
cost = st.slider('Coût (1=Low importance, 10=High importance)', 1, 10, 5)
execution_time = st.slider("Temps d'exécution (1=Rapide, 10=Lent)", 1, 10, 5)
security = st.selectbox('Niveau de sécurité requis (ISO/GDPR)', [1, 2, 3, 4], format_func=lambda x: f"Niveau {x}")
services = st.multiselect('Services requis', ['IA', 'Big Data', 'Containers', 'Databases'], default=['IA'])
services_code = len(services)
location = st.selectbox('Localisation des data centers', ['Americas', 'Europe', 'Asia', 'Multi'])
loc_map = {'Americas': 1, 'Europe': 2, 'Asia': 3, 'Multi': 4}
migration = st.selectbox('Facilité de migration (1=Facile, 5=Difficile)', [1, 2, 3, 4, 5], index=2)
infra = st.selectbox("Type d'infrastructure", ['Public Cloud', 'Private Cloud'])
infra_map = {'Public Cloud': 1, 'Private Cloud': 2}
compliance = st.selectbox('Conformité aux normes (1=Strict, 2=Modérée)', [1, 2])

if st.button('Obtenir la recommandation'):
    client_features = [cost, execution_time, security, 1, services_code, compliance, migration, loc_map[location], infra_map[infra]]
    recommendation = recommend_cf(client_features)
    st.success(f"✅ Recommandation: **{recommendation}**")

st.caption("Ce système utilise une similarité cosinus calculée manuellement pour une meilleure compréhension pédagogique.")
