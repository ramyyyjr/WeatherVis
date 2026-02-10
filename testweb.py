# merged_app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import requests
import io

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
# 2️⃣ Load Dataset (Google Drive)
# ======================
@st.cache_data(show_spinner=True)
def load_data(url, needed_columns=None, chunk_size=100_000):
    response = requests.get(url)
    if response.status_code != 200:
        return pd.DataFrame()

    chunks = []
    for chunk in pd.read_csv(
        io.StringIO(response.content.decode("utf-8")),
        chunksize=chunk_size,
        low_memory=False,
        on_bad_lines="skip"
    ):
        if needed_columns:
            chunk = chunk[[c for c in needed_columns if c in chunk.columns]]
        chunks.append(chunk)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)

    # Parse date
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df["quarter"] = df["date"].dt.quarter

    return df


# ======================
# 3️⃣ Dataset URL
# ======================
FILE_ID = "1Lxy0FxQ4KkM2hNWp8I6v_W_4HHnczWZs"
URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

NEEDED_COLS = [
    "date", "capital", "country",
    "temp_mean_c_approx", "temp_min_c_approx", "temp_max_c_approx",
    "precip_mm", "windspeed_10m", "windgusts_10m", "daylight_hours"
]

df = load_data(URL, NEEDED_COLS)

if df.empty:
    st.error("❌ Dataset is empty. Check Google Drive sharing permissions.")
    st.stop()

TEMP_COL = "temp_mean_c_approx"

# ======================
# 4️⃣ Sidebar Filters
# ======================
st.sidebar.header("Filtres")

cities = np.insert(df["capital"].dropna().unique(), 0, "Toutes")
city = st.sidebar.selectbox("Ville :", cities)

year_min, year_max = int(df["year"].min()), int(df["year"].max())
year_range = st.sidebar.slider("Années :", year_min, year_max, (year_min, year_max))

plot_choice = st.sidebar.selectbox("Visualisation :", [
    "Distribution des températures",
    "Température moyenne par année",
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
    "Durée du soleil par mois (polar)",
    "Vitesse du vent — heatmap",
    "Top 20 pays - vitesse du vent"
])

# ======================
# 5️⃣ Data filtering
# ======================
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
# 6️⃣ Column detection
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
# 7️⃣ Plots
# ======================
if plot_choice == "Distribution des températures":
    fig = px.histogram(dff, x=TEMP_COL, nbins=50)
    st.plotly_chart(fig, use_container_width=True)

elif plot_choice == "Température moyenne par année":
    yearly = dff.groupby("year")[TEMP_COL].mean().reset_index()
    fig = px.line(yearly, x="year", y=TEMP_COL, markers=True)
    st.plotly_chart(fig, use_container_width=True)

elif plot_choice == "Top 10 villes les plus chaudes":
    top = df.groupby("capital")[TEMP_COL].mean().sort_values(ascending=False).head(10)
    st.bar_chart(top)

elif plot_choice == "Top 10 villes les plus froides":
    top = df.groupby("capital")[TEMP_COL].mean().sort_values().head(10)
    st.bar_chart(top)

elif plot_choice == "Évolution d'une ville":
    city_sel = st.sidebar.selectbox("Choisir une ville", df["capital"].dropna().unique())
    dcity = df[df["capital"] == city_sel]
    fig = px.line(dcity, x="date", y=TEMP_COL)
    st.plotly_chart(fig, use_container_width=True)

elif plot_choice == "Température moyenne par mois":
    monthly = dff.groupby("month")[TEMP_COL].mean()
    st.line_chart(monthly)

elif plot_choice == "Top 30 villes × mois (heatmap)":
    pivot = df.pivot_table(TEMP_COL, "capital", "month", "mean")
    top30 = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).head(30).index]
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(top30, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

elif plot_choice == "Distribution par mois (boxplot)":
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(x=dff["month"], y=dff[TEMP_COL], ax=ax)
    st.pyplot(fig)

elif plot_choice == "Précipitations moyennes par mois" and PRECIP_COL:
    st.bar_chart(df.groupby("month")[PRECIP_COL].mean())

elif plot_choice == "Densité température par année (hexbin)":
    fig, ax = plt.subplots()
    hb = ax.hexbin(df["year"], df[TEMP_COL], gridsize=25)
    st.pyplot(fig)

elif plot_choice == "Température (mois × année) - contour":
    pivot = df.pivot_table(TEMP_COL, "year", "month", "mean")
    fig, ax = plt.subplots()
    cs = ax.contourf(pivot.columns, pivot.index, pivot.values, levels=20)
    st.pyplot(fig)

elif plot_choice == "Jours ≤ 0°C par année":
    freezing = (df[TEMP_COL] <= 0).groupby(df["year"]).sum()
    st.bar_chart(freezing)

elif plot_choice == "Durée du soleil par mois" and DAYLIGHT_COL:
    st.line_chart(df.groupby("month")[DAYLIGHT_COL].mean())

elif plot_choice == "Vitesse du vent — heatmap" and WIND_COL:
    pivot = df.pivot_table(WIND_COL, "year", "month", "mean")
    fig, ax = plt.subplots()
    sns.heatmap(pivot, cmap="Blues", ax=ax)
    st.pyplot(fig)

elif plot_choice == "Top 20 pays - vitesse du vent" and WIND_COL:
    country_col = "country" if "country" in df.columns else "capital"
    top20 = df.groupby(country_col)[WIND_COL].mean().sort_values(ascending=False).head(20)
    st.bar_chart(top20)
