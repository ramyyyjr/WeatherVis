import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import requests
import io
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
# 2️⃣ Load Dataset (Google Drive Folder)
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
# 3️⃣ Dataset config
# ======================
FOLDER_ID = "1diUPY7F_xY-Cez6fzS67Km_SD1xZsONy"

NEEDED_COLS = [
    "date", "capital", "country",
    "temp_mean_c_approx", "temp_min_c_approx", "temp_max_c_approx",
    "precip_mm", "windspeed_10m", "windgusts_10m", "daylight_hours"
]

df = load_data_from_folder(FOLDER_ID, NEEDED_COLS)

if df.empty:
    st.error("❌ Dataset is empty. Check Google Drive sharing.")
    st.stop()

TEMP_COL = "temp_mean_c_approx"

# ======================
# 4️⃣ Sidebar filters
# ======================
st.sidebar.header("🎛️ Filtres")

countries = st.sidebar.multiselect(
    "🌍 Pays",
    options=sorted(df["country"].dropna().unique()),
    default=None
)

capitals = st.sidebar.multiselect(
    "🏙️ Capitales",
    options=sorted(df["capital"].dropna().unique()),
    default=None
)

if countries:
    df = df[df["country"].isin(countries)]
if capitals:
    df = df[df["capital"].isin(capitals)]

# ======================
# 5️⃣ Dataset Preview (BIGGER TABLE ✅)
# ======================
st.subheader("📋 Aperçu du Dataset")

st.dataframe(
    df.head(50),
    use_container_width=True,
    height=450,
    column_config={
        col: st.column_config.Column(width="medium")
        for col in df.columns[:15]   # 👈 SHOW 15 COLUMNS
    }
)

st.caption(f"🔢 {df.shape[0]:,} lignes × {df.shape[1]} colonnes")

# ======================
# 6️⃣ Visualization Controls
# ======================
st.subheader("📊 Visualisations")

select_all = st.checkbox("✅ Sélectionner toutes les visualisations", value=True)

if select_all:
    show_temp_trend = True
    show_temp_dist = True
    show_precip = True
    show_wind = True
else:
    show_temp_trend = st.checkbox("📈 Évolution de la température")
    show_temp_dist = st.checkbox("📊 Distribution de la température")
    show_precip = st.checkbox("🌧️ Précipitations")
    show_wind = st.checkbox("💨 Vent")

# ======================
# 7️⃣ Visualizations
# ======================
if show_temp_trend:
    st.markdown("### 📈 Évolution de la température moyenne")
    df_trend = df.groupby("date", as_index=False)[TEMP_COL].mean()
    fig = px.line(
        df_trend,
        x="date",
        y=TEMP_COL,
        labels={"date": "Date", TEMP_COL: "Température (°C)"}
    )
    st.plotly_chart(fig, use_container_width=True)

if show_temp_dist:
    st.markdown("### 📊 Distribution de la température")
    fig = px.histogram(
        df,
        x=TEMP_COL,
        nbins=40,
        labels={TEMP_COL: "Température (°C)"}
    )
    st.plotly_chart(fig, use_container_width=True)

if show_precip:
    st.markdown("### 🌧️ Précipitations moyennes par pays")
    if "precip_mm" in df.columns:
        df_precip = df.groupby("country", as_index=False)["precip_mm"].mean()
        fig = px.bar(
            df_precip,
            x="country",
            y="precip_mm",
            labels={"precip_mm": "Précipitations (mm)", "country": "Pays"}
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Colonne 'precip_mm' absente.")

if show_wind:
    st.markdown("### 💨 Vent : vitesse vs rafales")
    if "windspeed_10m" in df.columns and "windgusts_10m" in df.columns:
        fig = px.scatter(
            df,
            x="windspeed_10m",
            y="windgusts_10m",
            labels={
                "windspeed_10m": "Vitesse du vent (m/s)",
                "windgusts_10m": "Rafales (m/s)"
            }
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Colonnes de vent manquantes.")

# ======================
# 8️⃣ Footer
# ======================
st.markdown("---")
st.caption("🌍 Weather Dashboard — Streamlit | Data Science Project")
