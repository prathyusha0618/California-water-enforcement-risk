import streamlit as st
from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np
import plotly.express as px
import shap
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="CA Water Risk")
st.title("California Water System Enforcement Risk Dashboard")

@st.cache_data
def load_data():
    cols = ["REGISTRY_ID", "FAC_NAME", "FAC_CITY", "FAC_STATE", "FAC_LAT", "FAC_LONG",
              "SDWIS_FLAG", "FAC_POP_DEN", "FAC_INSPECTION_COUNT", "FAC_DAYS_LAST_INSPECTION",
              "SDWA_FORMAL_ACTION_COUNT", "FAC_MAJOR_FLAG", "SDWA_SYSTEM_TYPES",
              "AIR_FLAG", "NPDES_FLAG", "RCRA_FLAG", "TRI_FLAG", "GHG_FLAG"]
    df = pd.read_csv("ECHO_EXPORTER.csv", usecols=cols)
    df = df[df['FAC_STATE'] == 'CA']
    df = df[df['SDWIS_FLAG'] == 'Y']
    df['had_enforcement'] = (df['SDWA_FORMAL_ACTION_COUNT'].fillna(0) > 0).astype(int)
    df = df.reset_index(drop=True)

    df['FAC_MAJOR_FLAG'] = (df['FAC_MAJOR_FLAG'] == 'Y').astype(int)

    other_program_flags = ['AIR_FLAG', 'NPDES_FLAG', 'RCRA_FLAG', 'TRI_FLAG', 'GHG_FLAG']
    for col in other_program_flags:
          df[col] = (df[col] == 'Y').astype(int)
    df['num_other_epa_programs'] = df[other_program_flags].sum(axis=1)

    system_type_dummies = pd.get_dummies(df['SDWA_SYSTEM_TYPES'], prefix='sys_type').astype(int)
    df = pd.concat([df, system_type_dummies], axis=1)

    feature_cols = ['FAC_POP_DEN', 'FAC_INSPECTION_COUNT', 'FAC_DAYS_LAST_INSPECTION',
                       'FAC_MAJOR_FLAG', 'num_other_epa_programs'] + list(system_type_dummies.columns)

    return df, feature_cols

@st.cache_resource
def train_model(df, feature_cols):
    X = df[feature_cols]  # no .fillna(0) -- the pipeline's median imputer handles this properly
    y = df['had_enforcement']

    # Only used to sanity-check performance on unseen data -- NOT used for the scores shown below.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    eval_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
    ])
    eval_pipeline.fit(X_train, y_train)
    test_auc = roc_auc_score(y_test, eval_pipeline.predict_proba(X_test)[:, 1])

    # Final model, trained on ALL the data -- this is what actually scores every facility.
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model', RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42))
    ])
    pipeline.fit(X, y)

    X_all_imputed = pipeline.named_steps['imputer'].transform(X)
    y_prob_all = pipeline.predict_proba(X)[:, 1]
    explainer = shap.TreeExplainer(pipeline.named_steps['model'])

    return pipeline, y_prob_all, explainer, X_all_imputed, test_auc

ca_df, FEATURE_COLS = load_data()
pipeline, y_prob_all, explainer, X_all_imputed, test_auc = train_model(ca_df, FEATURE_COLS)

ca_df['enforcement_probability'] = y_prob_all
ca_df['risk_tier'] = pd.cut(
    ca_df['enforcement_probability'],
    bins=[0, 0.33, 0.66, 1.0],
    labels=['Low', 'Medium', 'High'],
    include_lowest=True
)

st.subheader("Model Overview")
st.write(f"Predicting real historical enforcement actions, not an invented score. Held-out test ROC AUC: {test_auc:.3f}.")

risk_tier_filter = st.sidebar.radio("Risk tier:", ["All", "High", "Medium", "Low"])

if risk_tier_filter != "All":
    ca_filtered = ca_df[ca_df["risk_tier"] == risk_tier_filter]
else:
    ca_filtered = ca_df.copy()

st.subheader("Distribution of Predicted Enforcement Probability")
fig = px.histogram(
    ca_filtered,
    x="enforcement_probability",
    color="risk_tier",
    color_discrete_map={"High": "red", "Medium": "orange", "Low": "green"},
    nbins=50,
    title="Distribution of Predicted Enforcement Probability"
)
st.plotly_chart(fig, use_container_width=True)
import folium
from streamlit_folium import st_folium

st.subheader("Map of Facilities by Risk Tier")
map_data = ca_filtered.dropna(subset=['FAC_LAT', 'FAC_LONG', 'risk_tier'])

color_map = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}

m = folium.Map(location=[37.0, -119.5], zoom_start=6)

for _, row in map_data.iterrows():
    folium.CircleMarker(
        location=[row['FAC_LAT'], row['FAC_LONG']],
        radius=4,
        color=color_map.get(row['risk_tier'], 'gray'),
        fill=True,
        fill_opacity=0.7,
        popup=f"{row['FAC_NAME']} ({row['enforcement_probability']:.2f})"
    ).add_to(m)

st_folium(m, width=900, height=500)

st.subheader("Highest Risk Water Systems")
top20 = ca_filtered.sort_values("enforcement_probability", ascending=False).head(20)
st.dataframe(
    top20[["REGISTRY_ID", "FAC_NAME", "FAC_CITY", "enforcement_probability", "risk_tier"]],
    use_container_width=True
)

st.subheader("Why is this facility high risk? (SHAP explanation)")
high_risk_facilities = ca_df[ca_df["risk_tier"] == "High"].sort_values(
    "enforcement_probability", ascending=False
)
if len(high_risk_facilities) > 0:
    top_facilities = high_risk_facilities.head(50)
    selected_id = st.selectbox(
        "Select a high risk facility:",
        top_facilities["REGISTRY_ID"].tolist(),
        format_func=lambda rid: top_facilities.loc[top_facilities["REGISTRY_ID"] == rid, "FAC_NAME"].values[0]
    )

    if selected_id:
        row_pos = ca_df.index[ca_df["REGISTRY_ID"] == selected_id][0]
        prob = ca_df.loc[row_pos, 'enforcement_probability']
        st.write(f"Predicted enforcement probability: {prob:.3f}")

        row_shap = explainer.shap_values(X_all_imputed[row_pos:row_pos + 1])
        shap.waterfall_plot(
            shap.Explanation(
                values=row_shap[0, :, 1],
                base_values=explainer.expected_value[1],
                data=X_all_imputed[row_pos],
                feature_names=FEATURE_COLS
            ),
            show=False
        )
        st.pyplot(plt.gcf())
        plt.close()
else:
    st.write("No high risk facilities found with current filter.")