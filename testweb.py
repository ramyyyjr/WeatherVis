# merged_app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import requests  
import io

# ----------------------
# 1️⃣ Page config
# ----------------------
st.set_page_config(
    page_title="Weather Dashboard",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🌡️ Dashboard Climat — Analyse Température et Météo")

# ----------------------
# 2️⃣ Load Dataset
# ----------------------
import pandas as pd
import streamlit as st

@st.cache_data
def load_data(url=None, needed_columns=None, chunk_size=100000):
    """
    Load a large CSV in chunks from a URL (e.g., Google Drive direct download),
    keeping only the necessary columns to save memory.
    """
    if url is None:
        # fallback: direct download from Google Drive
        file_id = "1Lxy0FxQ4KkM2hNWp8I6v_W_4HHnczWZs"
        url = f"https://drive.google.com/uc?id={file_id}&export=download"

    # Download CSV content
    response = requests.get(url)
    if response.status_code != 200:
        st.error(f"Failed to download CSV. Status code: {response.status_code}")
        return pd.DataFrame()  # return empty df
    
    # Read CSV in chunks from downloaded content
    chunks = []
    for chunk in pd.read_csv(io.StringIO(response.content.decode('utf-8')),
                             chunksize=chunk_size,
                             low_memory=False,
                             on_bad_lines='skip'):
        if needed_columns:
            chunk = chunk[[c for c in needed_columns if c in chunk.columns]]
        chunks.append(chunk)
    
    if not chunks:
        st.warning("Dataset is empty! Check CSV path or URL.")
        return pd.DataFrame()
    
    # Combine all chunks
    df = pd.concat(chunks, ignore_index=True)

    # Parse date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['quarter'] = df['date'].dt.quarter
    
    return df

# --------------------------
# USAGE EXAMPLE
# --------------------------
# Google Drive direct download link
file_id = "1Lxy0FxQ4KkM2hNWp8I6v_W_4HHnczWZs"
url = f"https://drive.google.com/uc?export=download&id={file_id}"

# Specify only the columns your dashboard uses
needed_cols = [
    "date", "capital", "country", "temp_mean_c_approx", "temp_min_c_approx", 
    "temp_max_c_approx", "precip_mm", "windspeed_10m", "windgusts_10m", "daylight_hours"
]

# Load the DataFrame
df = load_data(url, needed_columns=needed_cols)

# Quick preview
st.write(df.head())

df = load_data()
if df.empty:
    st.warning("Dataset is empty! Check CSV path.")
    st.stop()

temp_col = "temp_mean_c_approx"

# ----------------------
# 3️⃣ Sidebar Filters
# ----------------------
st.sidebar.header("Filtres globaux")

# City filter
ville_list = np.append(["Toutes"], df["capital"].dropna().unique())
ville = st.sidebar.selectbox("Choisir une ville (filtre) :", ville_list)

# Year range filter
annee_min, annee_max = int(df["year"].min()), int(df["year"].max())
annee_range = st.sidebar.slider("Sélectionner une plage d'années :", annee_min, annee_max, (annee_min, annee_max))

# Plot selection
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

# ----------------------
# 4️⃣ Filter Data Function
# ----------------------
def filter_df(df):
    d = df.copy()
    d = d[(d["year"] >= annee_range[0]) & (d["year"] <= annee_range[1])]
    if ville != "Toutes":
        d = d[d["capital"] == ville]
    return d

dff = filter_df(df)

st.subheader("Aperçu du dataset")
st.dataframe(df.head())
st.write("Colonnes disponibles :", df.columns.tolist())

# ----------------------
# 5️⃣ Helper functions to detect optional columns
# ----------------------
def find_column(keywords):
    for col in df.columns:
        for k in keywords:
            if k in col.lower():
                return col
    return None

precip_col = find_column(["precip", "precipitation", "rainfall"])
daylight_col = find_column(["daylight", "sunshine", "sun"])
windspeed_col = find_column(["windspeed", "wind_speed", "wind"])

# ----------------------
# 6️⃣ Plot Functions
# ----------------------
def plot_distribution(d):
    fig = px.histogram(d, x=temp_col, nbins=50, title="Distribution des températures")
    st.plotly_chart(fig, use_container_width=True)

def plot_yearly(d):
    yearly = d.groupby("year")[temp_col].mean().reset_index()
    fig = px.line(yearly, x="year", y=temp_col, markers=True, title="Température moyenne par année")
    st.plotly_chart(fig, use_container_width=True)

def plot_top_cities(desc=True):
    city_temp = df.groupby("capital")[temp_col].mean().sort_values(ascending=not desc).head(10)
    fig, ax = plt.subplots(figsize=(10,6))
    city_temp.plot(kind="bar", color='coral' if desc else 'skyblue', edgecolor='black', ax=ax)
    ax.set_title("Top 10 villes" + (" les plus chaudes" if desc else " les plus froides"))
    ax.set_ylabel("Température moyenne (°C)")
    plt.xticks(rotation=45, ha='right')
    st.pyplot(fig)

def plot_city_evolution(city_name):
    dcity = df[df["capital"] == city_name]
    if dcity.empty:
        st.warning(f"Ville '{city_name}' non trouvée.")
        return
    fig, ax = plt.subplots(figsize=(12,5))
    ax.plot(dcity["date"], dcity[temp_col], color='darkblue')
    ax.set_title(f"Évolution température — {city_name}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Température (°C)")
    plt.xticks(rotation=45)
    st.pyplot(fig)

def plot_monthly(d):
    monthly = d.groupby("month")[temp_col].mean()
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(monthly.index, monthly.values, marker='o', color='green')
    ax.set_xticks(range(1,13))
    ax.set_xticklabels(['Jan','Fév','Mar','Avr','Mai','Jun','Jul','Aoû','Sep','Oct','Nov','Déc'])
    ax.set_title("Température moyenne par mois")
    st.pyplot(fig)

def plot_heatmap_top30():
    pivot = df.pivot_table(values=temp_col, index="capital", columns="month", aggfunc="mean")
    top30 = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).head(30).index]
    fig, ax = plt.subplots(figsize=(14,10))
    sns.heatmap(top30, cmap="coolwarm", linewidths=0.3, cbar_kws={'label': 'Température (°C)'}, ax=ax)
    st.pyplot(fig)

def plot_boxplot(d):
    fig, ax = plt.subplots(figsize=(12,5))
    sns.boxplot(x=d["month"].astype(int), y=d[temp_col], palette="Set2", ax=ax)
    ax.set_xticks(range(0,12))
    ax.set_xticklabels(['Jan','Fév','Mar','Avr','Mai','Jun','Jul','Aoû','Sep','Oct','Nov','Déc'])
    ax.set_title("Distribution des températures par mois (boxplot)")
    st.pyplot(fig)

def plot_precipitations():
    if not precip_col:
        st.info("Aucune colonne précipitations détectée.")
        return
    monthly_precip = df.groupby("month")[precip_col].mean()
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(monthly_precip.index, monthly_precip.values, color='steelblue')
    ax.set_title("Précipitations moyennes par mois")
    st.pyplot(fig)

def plot_hexbin():
    fig, ax = plt.subplots(figsize=(12,5))
    hb = ax.hexbin(df["year"], df[temp_col], gridsize=25, cmap='YlOrRd', mincnt=1)
    fig.colorbar(hb, ax=ax, label='Nombre d\'observations')
    ax.set_xlabel('Année')
    ax.set_ylabel('Température (°C)')
    st.pyplot(fig)

def plot_contour():
    pivot = df.pivot_table(values=temp_col, index="year", columns="month", aggfunc="mean")
    if pivot.empty:
        st.info("Données insuffisantes pour le contour plot.")
        return
    fig, ax = plt.subplots(figsize=(12,7))
    contour = ax.contourf(pivot.columns, pivot.index, pivot.values, levels=20, cmap="RdYlBu_r")
    fig.colorbar(contour, ax=ax, label='Température (°C)')
    ax.set_xlabel('Mois')
    ax.set_ylabel('Année')
    st.pyplot(fig)

def plot_freezing_days():
    df['is_freezing'] = df['temp_mean_c_approx'] <= 0
    freezing = df.groupby('year')['is_freezing'].sum()
    fig, ax = plt.subplots(figsize=(12,5))
    ax.bar(freezing.index, freezing.values, color='lightblue')
    ax.set_title('Nombre de jours ≤ 0°C par année')
    st.pyplot(fig)

def plot_daylight():
    if not daylight_col:
        st.info("Aucune colonne durée du soleil détectée.")
        return
    monthly_daylight = df.groupby('month')[daylight_col].mean()
    angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111, polar=True)
    ax.bar(angles, monthly_daylight.values, width=0.5, color='gold', edgecolor='orange')
    ax.set_xticks(angles)
    ax.set_xticklabels(['Jan','Fév','Mar','Avr','Mai','Juin','Juil','Aoû','Sep','Oct','Nov','Déc'])
    st.pyplot(fig)

def plot_wind_heatmap():
    if not windspeed_col:
        st.info("Aucune colonne vitesse du vent détectée.")
        return
    pivot = df.pivot_table(values=windspeed_col, index='year', columns='month', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(12,8))
    sns.heatmap(pivot, cmap='Blues', linewidths=0.5, ax=ax)
    st.pyplot(fig)

def plot_top20_wind():
    if not windspeed_col:
        st.info("Aucune colonne vitesse du vent détectée.")
        return
    country_col = 'country' if 'country' in df.columns else ('capital' if 'capital' in df.columns else None)
    if not country_col:
        st.info("Aucune colonne pays/capital pour top 20.")
        return
    top20 = df.groupby(country_col)[windspeed_col].mean().sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(10,8))
    top20.plot(kind='barh', color='teal', edgecolor='black', ax=ax)
    ax.invert_yaxis()
    st.pyplot(fig)

# ----------------------
# 7️⃣ Dispatch plots
# ----------------------
if plot_choice == "Distribution des températures":
    plot_distribution(dff)
elif plot_choice == "Température moyenne par année":
    plot_yearly(dff)
elif plot_choice == "Top 10 villes les plus chaudes":
    plot_top_cities(desc=True)
elif plot_choice == "Top 10 villes les plus froides":
    plot_top_cities(desc=False)
elif plot_choice == "Évolution d'une ville":
    city_input = st.sidebar.selectbox("Ville pour évolution :", df['capital'].dropna().unique())
    plot_city_evolution(city_input)
elif plot_choice == "Température moyenne par mois":
    plot_monthly(dff)
elif plot_choice == "Top 30 villes × mois (heatmap)":
    plot_heatmap_top30()
elif plot_choice == "Distribution par mois (boxplot)":
    plot_boxplot(dff)
elif plot_choice == "Précipitations moyennes par mois":
    plot_precipitations()
elif plot_choice == "Densité température par année (hexbin)":
    plot_hexbin()
elif plot_choice == "Température (mois × année) - contour":
    plot_contour()
elif plot_choice == "Jours ≤ 0°C par année":
    plot_freezing_days()
elif plot_choice == "Durée du soleil par mois (polar)":
    plot_daylight()
elif plot_choice == "Vitesse du vent — heatmap":
    plot_wind_heatmap()
elif plot_choice == "Top 20 pays - vitesse du vent":
    plot_top20_wind()



