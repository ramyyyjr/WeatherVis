import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import gdown
import os

# ======================
# 1️⃣ Page config
# ======================
st.set_page_config(
    page_title="Weather Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title(" Weather Dashboard — Analyse Température et Météo")

# ======================
# 2️⃣ Load full dataset (all columns)
# ======================
@st.cache_data(show_spinner=True)
def load_data_from_folder(folder_id):
    folder_path = "data"

    if not os.path.exists(folder_path):
        gdown.download_folder(
            id=folder_id,
            output=folder_path,
            quiet=False,
            use_cookies=False
        )

    csv_files = [
        os.path.join(folder_path, f)
        for f in os.listdir(folder_path)
        if f.endswith(".csv")
    ]

    if not csv_files:
        return pd.DataFrame()

    dfs = []
    for file in csv_files:
        df_part = pd.read_csv(file, low_memory=False, on_bad_lines="skip")
        dfs.append(df_part)  # keep all columns

    df = pd.concat(dfs, ignore_index=True)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter

    return df

# ======================
# 3️⃣ Dataset config
# ======================
FOLDER_ID = "1diUPY7F_xY-Cez6fzS67Km_SD1xZsONy"

df = load_data_from_folder(FOLDER_ID)

if df.empty:
    st.error("❌ Dataset is empty. Check Google Drive sharing.")
    st.stop()

TEMP_COL = "temp_mean_c_approx"

# ======================
# 4️⃣ Sidebar: Filter + Visualisation checkboxes
# ======================
st.sidebar.header("🎛️ Filtres")

countries = st.sidebar.multiselect(
    "🌍 Pays",
    options=sorted(df["country"].dropna().unique())
)

if countries:
    df = df[df["country"].isin(countries)]

st.sidebar.markdown("---")
st.sidebar.header("📊 Visualisations")

# List of all visualizations
viz_list = [
    "Température moyenne par année",
    "Distribution des températures",
    "Top 10 villes les plus chaudes",
    "Top 10 villes les plus froides",
    "Évolution d'une ville",
    "Température moyenne par mois",
    "Top 30 villes × mois (heatmap)",
    "Distribution par mois (boxplot)",
    "Précipitations moyennes par mois",
    "Densité température par année (hexbin)",
    "Température (mois × année) - contour",
    "Jours ≤ 0°C par année",
    "Durée du soleil par mois",
    "Vitesse du vent — heatmap",
     "Top 20 pays - vitesse du vent"
]

# Select all checkbox
select_all = st.sidebar.checkbox("SELECTIONNER TOUS", value=True)

# Dict to store which visualisations are checked
viz_checked = {}
for v in viz_list:
    if select_all:
        viz_checked[v] = True
    else:
        viz_checked[v] = st.sidebar.checkbox(v, value=False)

# ======================
# 5️⃣ Dataset preview
# ======================
st.subheader(" Aperçu du Dataset")
st.dataframe(df.head(50), use_container_width=True, height=450)
st.caption(f"🔢 {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

# ======================
# 6️⃣ Helper for optional columns
# ======================
def find_col(keys):
    for c in df.columns:
        for k in keys:
            if k.lower() in c.lower():
                return c
    return None

PRECIP_COL = find_col(["precip", "rain", "snow"])
DAYLIGHT_COL = find_col(["daylight", "sunshine", "sun"])
WIND_COL = find_col(["windspeed", "wind"])

# ======================
# 7️⃣ Filter function for a city
# ======================
def filter_city(city_name):
    if city_name in df["capital"].values:
        return df[df["capital"] == city_name]
    else:
        return pd.DataFrame()

# ======================
# 8️⃣ Plots
# ======================

# 📈 Température moyenne par année
if viz_checked[" Température moyenne par année"]:
    st.markdown("###  Température moyenne par année")
    yearly = df.groupby("year")[TEMP_COL].mean().reset_index()
    fig = px.line(yearly, x="year", y=TEMP_COL, markers=True, labels={TEMP_COL:"Température (°C)", "year":"Année"})
    st.plotly_chart(fig, use_container_width=True)

# 📊 Distribution des températures
if viz_checked[" Distribution des températures"]:
    st.markdown("###  Distribution des températures")
    fig = px.histogram(df, x=TEMP_COL, nbins=40, labels={TEMP_COL:"Température (°C)"})
    st.plotly_chart(fig, use_container_width=True)

# 🌡️ Top 10 villes les plus chaudes
if viz_checked[" Top 10 villes les plus chaudes"]:
    st.markdown("###  Top 10 villes les plus chaudes")
    top_hot = df.groupby("capital")[TEMP_COL].mean().sort_values(ascending=False).head(10)
    st.bar_chart(top_hot)

# ❄️ Top 10 villes les plus froides
if viz_checked[" Top 10 villes les plus froides"]:
    st.markdown("###  Top 10 villes les plus froides")
    top_cold = df.groupby("capital")[TEMP_COL].mean().sort_values().head(10)
    st.bar_chart(top_cold)

# 🌡️ Évolution d'une ville
if viz_checked[" Évolution d'une ville"]:
    st.markdown("###  Évolution d'une ville")
    city_sel = st.sidebar.selectbox("Choisir une ville pour évolution", df["capital"].dropna().unique())
    dcity = filter_city(city_sel)
    if not dcity.empty:
        fig = px.line(dcity, x="date", y=TEMP_COL, labels={TEMP_COL:"Température (°C)", "date":"Date"})
        st.plotly_chart(fig, use_container_width=True)

# 📅 Température moyenne par mois
if viz_checked[" Température moyenne par mois"]:
    st.markdown("###  Température moyenne par mois")
    monthly = df.groupby("month")[TEMP_COL].mean()
    st.line_chart(monthly)

# 🔥 Top 30 villes × mois (heatmap)
if viz_checked[" Top 30 villes × mois (heatmap)"]:
    st.markdown("###  Top 30 villes × mois (heatmap)")
    pivot = df.pivot_table(values=TEMP_COL, index="capital", columns="month", aggfunc="mean")
    top30 = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).head(30).index]
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(top30, cmap="coolwarm", linewidths=0.3, cbar_kws={'label':'Température (°C)'}, ax=ax)
    st.pyplot(fig)

# 📦 Distribution par mois (boxplot)
if viz_checked[" Distribution par mois (boxplot)"]:
    st.markdown("###  Distribution par mois (boxplot)")
    fig, ax = plt.subplots(figsize=(12,5))
    sns.boxplot(x=df["month"], y=df[TEMP_COL], palette="Set2", ax=ax)
    st.pyplot(fig)

# 🌧️ Précipitations moyennes par mois
if viz_checked[" Précipitations moyennes par mois"] and PRECIP_COL:
    st.markdown("###  Précipitations moyennes par mois")
    monthly_precip = df.groupby("month")[PRECIP_COL].mean()
    st.bar_chart(monthly_precip)

# 🔶 Densité température par année (hexbin)
if viz_checked[" Densité température par année (hexbin)"]:
    st.markdown("###  Densité température par année (hexbin)")
    fig, ax = plt.subplots(figsize=(12,5))
    hb = ax.hexbin(df["year"], df[TEMP_COL], gridsize=25, cmap='YlOrRd', mincnt=1)
    fig.colorbar(hb, ax=ax, label='Nombre d\'observations')
    ax.set_xlabel('Année'); ax.set_ylabel('Température (°C)')
    st.pyplot(fig)

# 🗻 Température (mois × année) - contour
if viz_checked[" Température (mois × année) - contour"]:
    st.markdown("###  Température (mois × année) - contour")
    pivot = df.pivot_table(values=TEMP_COL, index="year", columns="month", aggfunc="mean")
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(12,7))
        cs = ax.contourf(pivot.columns, pivot.index, pivot.values, levels=20, cmap="RdYlBu_r")
        fig.colorbar(cs, ax=ax, label='Température (°C)')
        ax.set_xlabel('Mois'); ax.set_ylabel('Année')
        st.pyplot(fig)

# ❄️ Jours ≤ 0°C par année
if viz_checked[" Jours ≤ 0°C par année"]:
    st.markdown("###  Jours ≤ 0°C par année")
    df['is_freezing'] = df[TEMP_COL] <= 0
    freezing = df.groupby('year')['is_freezing'].sum()
    st.bar_chart(freezing)

# ☀️ Durée du soleil par mois
if viz_checked[" Durée du soleil par mois"] and DAYLIGHT_COL:
    st.markdown("### Durée du soleil par mois")
    monthly_sun = df.groupby('month')[DAYLIGHT_COL].mean()
    st.line_chart(monthly_sun)

# 💨 Vitesse du vent — heatmap
if viz_checked["💨 Vitesse du vent — heatmap"] and WIND_COL:
    st.markdown("### 💨 Vitesse du vent — heatmap")
    pivot = df.pivot_table(values=WIND_COL, index='year', columns='month', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(pivot, cmap='Blues', ax=ax)
    st.pyplot(fig)

# 🏆 Top 20 pays - vitesse du vent
if viz_checked["Top 20 pays - vitesse du vent"] and WIND_COL:
    st.markdown("###  Top 20 pays - vitesse du vent")
    country_col = 'country' if 'country' in df.columns else 'capital'
    top20 = df.groupby(country_col)[WIND_COL].mean().sort_values(ascending=False).head(20)
    st.bar_chart(top20)

# ======================
# Footer
# ======================
st.markdown("---")
st.caption("🌍 Weather Dashboard | Data Visualisation Project | Réalise par : Bouzidi Ramy & Benarbia Imed Eddine [4eme Ingénieur Informatique - Data Science] ")

