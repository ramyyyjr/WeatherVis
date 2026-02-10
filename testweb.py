# merged_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
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

st.title("🌡️ Dashboard Climat — Analyse Température et Météo")

# ======================
# 2️⃣ Load Dataset (Google Drive FOLDER)
# ======================
@st.cache_data(show_spinner=True)
def load_data_from_folder(folder_id, needed_columns=None):
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
        if needed_columns:
            df_part = df_part[[c for c in needed_columns if c in df_part.columns]]
        dfs.append(df_part)

    df = pd.concat(dfs, ignore_index=True)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter

    return df

# ======================
# 3️⃣ Dataset FOLDER ID
# ======================
FOLDER_ID = "1diUPY7F_xY-Cez6fzS67Km_SD1xZsONy"

NEEDED_COLS = [
    "date", "capital", "country",
    "temp_mean_c_approx", "temp_min_c_approx", "temp_max_c_approx",
    "precip_mm", "windspeed_10m", "windgusts_10m", "daylight_hours"
]

df = load_data_from_folder(FOLDER_ID, NEEDED_COLS)

if df.empty:
    st.error("❌ Dataset is empty. No CSV files found in the folder.")
    st.stop()

TEMP_COL = "temp_mean_c_approx"

# ======================
# 4️⃣ Sidebar filters
# ======================
st.sidebar.header("🔎 Filtres")

cities = np.insert(df["capital"].dropna().unique(), 0, "Toutes")
city = st.sidebar.selectbox("Ville :", cities)

year_min, year_max = int(df["year"].min()), int(df["year"].max())
year_range = st.sidebar.slider("Années :", year_min, year_max, (year_min, year_max))

def filter_df(data):
    d = data[
        (data["year"] >= year_range[0]) &
        (data["year"] <= year_range[1])
    ]
    if city != "Toutes":
        d = d[d["capital"] == city]
    return d

dff = filter_df(df)

st.subheader("Aperçu des données")
st.dataframe(dff.head())

# ======================
# 5️⃣ Column detection
# ======================
def find_col(keys):
    for c in df.columns:
        for k in keys:
            if k in c.lower():
                return c
    return None

PRECIP_COL = find_col(["precip"])
DAYLIGHT_COL = find_col(["daylight", "sun"])
WIND_COL = find_col(["windspeed"])

# ======================
# 6️⃣ Sidebar checkboxes
# ======================
st.sidebar.header("📊 Visualisations")

show_dist = st.sidebar.checkbox("Distribution des températures")
show_yearly = st.sidebar.checkbox("Température moyenne par année")
show_hot = st.sidebar.checkbox("Top 10 villes les plus chaudes")
show_cold = st.sidebar.checkbox("Top 10 villes les plus froides")
show_city = st.sidebar.checkbox("Évolution d'une ville")
show_monthly = st.sidebar.checkbox("Température moyenne par mois")
show_heatmap = st.sidebar.checkbox("Top 30 villes × mois (heatmap)")
show_box = st.sidebar.checkbox("Distribution par mois (boxplot)")
show_precip = st.sidebar.checkbox("Précipitations moyennes par mois")
show_hex = st.sidebar.checkbox("Densité température par année")
show_contour = st.sidebar.checkbox("Température (mois × année)")
show_freeze = st.sidebar.checkbox("Jours ≤ 0°C par année")
show_daylight = st.sidebar.checkbox("Durée du soleil par mois")
show_wind_heat = st.sidebar.checkbox("Vitesse du vent — heatmap")
show_wind_top = st.sidebar.checkbox("Top 20 pays - vitesse du vent")

# ======================
# 7️⃣ Visualisations
# ======================
if show_dist:
    st.subheader("Distribution des températures")
    fig = px.histogram(dff, x=TEMP_COL, nbins=50)
    st.plotly_chart(fig, use_container_width=True)

if show_yearly:
    st.subheader("Température moyenne par année")
    yearly = dff.groupby("year")[TEMP_COL].mean().reset_index()
    fig = px.line(yearly, x="year", y=TEMP_COL, markers=True)
    st.plotly_chart(fig, use_container_width=True)

if show_hot:
    st.subheader("Top 10 villes les plus chaudes")
    st.bar_chart(df.groupby("capital")[TEMP_COL].mean().sort_values(ascending=False).head(10))

if show_cold:
    st.subheader("Top 10 villes les plus froides")
    st.bar_chart(df.groupby("capital")[TEMP_COL].mean().sort_values().head(10))

if show_city:
    st.subheader("Évolution d'une ville")
    city_sel = st.selectbox("Ville :", df["capital"].dropna().unique(), key="city_plot")
    fig = px.line(df[df["capital"] == city_sel], x="date", y=TEMP_COL)
    st.plotly_chart(fig, use_container_width=True)

if show_monthly:
    st.subheader("Température moyenne par mois")
    st.line_chart(dff.groupby("month")[TEMP_COL].mean())

if show_heatmap:
    st.subheader("Top 30 villes × mois (heatmap)")
    pivot = df.pivot_table(TEMP_COL, "capital", "month", "mean")
    top30 = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).head(30).index]
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(top30, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

if show_box:
    st.subheader("Distribution par mois (boxplot)")
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(x=dff["month"], y=dff[TEMP_COL], ax=ax)
    st.pyplot(fig)

if show_precip and PRECIP_COL:
    st.subheader("Précipitations moyennes par mois")
    st.bar_chart(df.groupby("month")[PRECIP_COL].mean())

if show_hex:
    st.subheader("Densité température par année")
    fig, ax = plt.subplots()
    ax.hexbin(df["year"], df[TEMP_COL], gridsize=25)
    st.pyplot(fig)

if show_contour:
    st.subheader("Température (mois × année)")
    pivot = df.pivot_table(TEMP_COL, "year", "month", "mean")
    fig, ax = plt.subplots()
    ax.contourf(pivot.columns, pivot.index, pivot.values, levels=20)
    st.pyplot(fig)

if show_freeze:
    st.subheader("Jours ≤ 0°C par année")
    st.bar_chart((df[TEMP_COL] <= 0).groupby(df["year"]).sum())

if show_daylight and DAYLIGHT_COL:
    st.subheader("Durée du soleil par mois")
    st.line_chart(df.groupby("month")[DAYLIGHT_COL].mean())

if show_wind_heat and WIND_COL:
    st.subheader("Vitesse du vent — heatmap")
    pivot = df.pivot_table(WIND_COL, "year", "month", "mean")
    fig, ax = plt.subplots()
    sns.heatmap(pivot, cmap="Blues", ax=ax)
    st.pyplot(fig)

if show_wind_top and WIND_COL:
    st.subheader("Top 20 pays - vitesse du vent")
    country_col = "country" if "country" in df.columns else "capital"
    st.bar_chart(
        df.groupby(country_col)[WIND_COL]
        .mean()
        .sort_values(ascending=False)
        .head(20)
    )
