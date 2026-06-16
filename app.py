from wsgiref import headers

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats as scipy_stats
import requests
from bs4 import BeautifulSoup
from textblob import TextBlob
import re
import warnings
import os

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="Portafolio",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Fuentes y body ── */
@font-face {
    font-family: 'Helvetica';
    src: url('/app/static/Helvetica-Roman.otf') format('opentype');
}
html, body, [class*="css"] { font-family: 'Helvetica'; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] { background: #0d0d1a !important; }
section[data-testid="stSidebar"] * { color: #c9cfe0 !important; }
.sidebar-logo {
    text-align: center;
    padding: 12px 0 8px;
    font-size: 1.6rem;
    letter-spacing: .12em;
    font-family: 'Helvetica', sans-serif;
    color: #e94560 !important;
}

/* ── Tarjetas de métricas ── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #13132b 0%, #1e2a4a 100%);
    border: 1px solid #2a3a6a;
    border-radius: 12px;
    padding: 14px 20px;
}

/* ── Barra horizontal ── */
hr { border-color: #2a2a4a !important; }

/* ── Cajas de hipótesis ── */
.hypothesis-box {
    background: #13132b;
    border-left: 4px solid #e94560;
    border-radius: 6px;
    padding: 16px 20px;
    margin: 12px 0;
    color: #c9cfe0;
}

/* ── Tarjeta perfil ── */
.profile-card {
    background: linear-gradient(135deg,#0f0c29,#302b63,#24243e);
    border-radius: 16px;
    padding: 36px 40px;
    color: white;
    text-align: center;
    font-family: 'Helvetica', sans-serif;
    margin-bottom: 24px;
}
.profile-card h1 { color: #e94560; font-size: 2.4rem; margin-bottom: 4px; }
.profile-card h3 { color: #a8b2d8; font-weight: 300; }

/* ── Placeholder dashed box ── */
.dashed-box {
    border: 2px dashed #3a3a6a;
    border-radius: 12px;
    padding: 32px;
    text-align: center;
    color: #6a6a9a;
}

/* ── Tarjeta de recomendación ── */
.rec-card {
    background: #13132b;
    border: 1px solid #e94560;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 6px 0;
    color: #c9cfe0;
}

/* ── Respuesta IA ── */
.ai-response {
    background: linear-gradient(135deg,#13132b,#1e2a4a);
    border-left: 5px solid #e94560;
    border-radius: 8px;
    padding: 20px 24px;
    color: #e8e8f0;
    line-height: 1.7;
    margin-top: 16px;
}
</style>
""", unsafe_allow_html=True)


DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "rym_clean1.csv")

@st.cache_data(show_spinner="Cargando dataset…")
def load_data():
    df = pd.read_csv(DATA_PATH, index_col=0)
    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df["year"]   = df["release_date"].dt.year
    df["decade"] = (df["year"] // 10 * 10).astype("Int64")
    df["secondary_genres"] = df["secondary_genres"].fillna("")
    df["descriptors"]      = df["descriptors"].fillna("")
    df["main_genre"]       = df["primary_genres"].apply(
        lambda x: x.split(",")[0].strip() if pd.notna(x) else "Unknown"
    )
    df["content"] = (
    df["primary_genres"].fillna("").astype(str) + " " +
    df["secondary_genres"].fillna("").astype(str) + " " +
    df["descriptors"].fillna("").astype(str)).str.strip()
    
    rating_median     = df["rating_count"].median()
    df["top_rated"]   = (df["avg_rating"] >= 3.8).astype(int)
    df["popular"]     = (df["rating_count"] >= rating_median).astype(int)
    return df

df = load_data()


with st.sidebar:
    st.markdown('<div class="sidebar-logo">🎵 DS Portfolio</div>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio(
        "Menú principal",
        options=[
            "🏠 Inicio",
            "📊 Análisis Exploratorio",
            "🤖 Aprendizaje Automático",
            "🎵 Sistema de Recomendación",
            "📁 Análisis por Archivos",
            "💬 Sentimientos & Scrapping",
            "🧠 Prompts de IA",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.caption(f"**Dataset:** RateYourMusic Top 5 000")
    st.caption(f"**Registros:** {len(df):,}   |   **Campos:** {len(df.columns)}")
    st.caption(f"**Período:** {int(df['year'].min())} - {int(df['year'].max())}")


def render_home():
    st.markdown("""
    <div class="profile-card">
        <h1>Portafolio de Ciencia de Datos</h1>
        <h3>Análisis de los Top 5,000 Álbumes en RateYourMusic</h3>
        <p style="color:#6a6a9a;margin-top:6px;">
            Ingeniería en Sistemas y Redes Informáticas · Ciclo I-2026
        </p>
    </div>
    """, unsafe_allow_html=True)

    col_photo, col_info = st.columns([1, 2], gap="large")

    with col_photo:
        st.markdown("#### Fotografía")
        st.image("assets/foto.jpg")

    with col_info:
        st.markdown("#### Información Personal")

        st.markdown(f"""
            <div style="background:linear-gradient(135deg,#1e3c72,#2a5298);
                        border-radius:12px;padding:20px 24px;color:white;margin-top:8px;">
                <h2 style="color:#FFD700;margin:0 0 4px;">Edwin González </h2>
                <p style="color:#a8b2d8;margin:2px 0;">
                    Estudiante de Ingeniería en Sistemas y Redes Informáticas
                </p>
                <p style="color:#a8b2d8;margin:0;">Ciclo I-2026 · Ciencia de Datos</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div style="text-align:left;padding:16px 20px;">
            <p style="margin:0;"><i> ¿Quién soy?</i></p>
            <hr style="border-color:#2a2a4a;margin:10px 0;">
            <p style="margin:0;color:#8888aa;">
                Soy estudiante de Ingeniería en Sistemas y Redes Informáticas.
                Me gusta la música.
                Me apasiona la programación aunque no me considere como un programador todavía. Un poco de backend.
                Busco aprender y poder tener las bases para desarrollar proyectos interesantes e importantes.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("---")

    # ─── KPIs del dataset ───
    st.markdown("#### 📊 Vistazo Rápido al Dataset")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("🎵 Álbumes", f"{len(df):,}")
    k2.metric("⭐ Rating Promedio", f"{df['avg_rating'].mean():.2f}")
    k3.metric("🎸 Géneros Únicos",
              df["primary_genres"].str.split(",").explode().str.strip().nunique())
    k4.metric("📅 Años Cubiertos",
              f"{int(df['year'].min())} - {int(df['year'].max())}")

    st.markdown("#### 🏆 Top 10 Álbumes")
    top10 = df.nsmallest(10, "position")[
        ["position", "release_name", "artist_name", "avg_rating", "rating_count", "primary_genres"]
    ].rename(columns={
        "position": "#", "release_name": "Álbum", "artist_name": "Artista",
        "avg_rating": "Rating ⭐", "rating_count": "Ratings 👥", "primary_genres": "Géneros"
    })
    st.dataframe(top10, use_container_width=True, hide_index=True)


def render_eda():
    st.markdown("## 📊 Análisis Exploratorio de Datos")

    submenu = st.selectbox("Selecciona una sección:", [
        "Descripción del Dataset",
        "Descripción de los Campos",
        "Navegador del Dataset",
        "Buscador de Registros",
        "Graficador Exploratorio",
        "Hipótesis",
    ])

    if submenu == "Descripción del Dataset":
        st.markdown("### Descripción del Dataset")

        col_a, col_b = st.columns([1, 1], gap="large")
        with col_a:
            st.markdown("""
**RateYourMusic Top 5 000 Álbumes**

Este dataset recopila los 5 000 álbumes más populares de **RateYourMusic.com**,
la mayor base de datos de música mantenida por su comunidad. Incluye calificaciones,
géneros, descriptores de atmósfera y métricas de popularidad para cada lanzamiento.

- **Origen:** RateYourMusic.com (comunidad)
- **Criterio:** Ranking de popularidad ponderada
- **Período cubierto:** 1954 - 2022
- **Tipo de lanzamientos:** Álbumes de estudio
            """)

        with col_b:
            st.markdown("**Estadísticas de variables numéricas:**")
            stats = df[["avg_rating", "rating_count", "review_count", "position"]].describe().round(2)
            stats.index = ["N", "Media", "Desv. Est.", "Mín.", "Q1", "Mediana", "Q3", "Máx."]
            st.dataframe(stats, use_container_width=True)

        st.markdown("### 📌 Diccionario de Campos")
        fields = pd.DataFrame({
            "Campo": ["position","release_name","artist_name","release_date","release_type",
                      "primary_genres","secondary_genres","descriptors",
                      "avg_rating","rating_count","review_count"],
            "Tipo": ["Entero","Texto","Texto","Fecha","Texto","Texto","Texto","Texto",
                     "Decimal","Entero","Entero"],
            "Descripción": [
                "Posición en el ranking de popularidad (1 = más popular)",
                "Título del álbum",
                "Nombre del artista o banda",
                "Fecha de lanzamiento",
                "Tipo de lanzamiento (todos son 'album' en este dataset)",
                "Géneros musicales primarios asignados por la comunidad",
                "Géneros secundarios o subgéneros adicionales (puede estar vacío)",
                "Descriptores de atmósfera y características del álbum",
                "Calificación promedio ponderada (escala 0.5 – 5.0)",
                "Número total de calificaciones individuales",
                "Número total de reseñas escritas por usuarios",
            ],
        })
        st.dataframe(fields, use_container_width=True, hide_index=True)

        missing = df.isnull().sum()
        missing = missing[missing > 0]
        if len(missing):
            st.warning("⚠️ Valores faltantes detectados:")
            st.dataframe(missing.to_frame("Faltantes"), use_container_width=True)
        else:
            st.success("No hay valores faltantes en las columnas originales.")

    elif submenu == "Descripción de los Campos":
        st.markdown("### 🔍 Descripción de los Campos")

        campo = st.selectbox("Selecciona un campo:", [
            "position","release_name","artist_name","release_date","release_type",
            "primary_genres","secondary_genres","descriptors",
            "avg_rating","rating_count","review_count",
        ])

        TIPOS = {
            "position":         ("Cuantitativo - Discreto", "Posición del álbum en el ranking global de popularidad."),
            "release_name":     ("Categórico - Nominal",    "Título del álbum o lanzamiento."),
            "artist_name":      ("Categórico - Nominal",    "Nombre del artista o banda que publicó el álbum."),
            "release_date":     ("Fecha",                   "Fecha exacta de lanzamiento (año-mes-día)."),
            "release_type":     ("Categórico",              "Tipo de lanzamiento. En este dataset todos son 'album'."),
            "primary_genres":   ("Categórico - Múltiple",   "Géneros musicales primarios, separados por coma."),
            "secondary_genres": ("Categórico - Múltiple",   "Subgéneros o géneros complementarios (puede estar vacío)."),
            "descriptors":      ("Categórico - Múltiple",   "Palabras clave que describen el mood o características del álbum."),
            "avg_rating":       ("Cuantitativo - Continuo", "Calificación promedio ponderada otorgada por la comunidad (0.5–5.0)."),
            "rating_count":     ("Cuantitativo - Discreto", "Número total de calificaciones individuales recibidas."),
            "review_count":     ("Cuantitativo - Discreto", "Número total de reseñas escritas por usuarios."),
        }

        tipo, desc = TIPOS[campo]
        col_a, col_b = st.columns([1, 1])
        with col_a:
            st.info(f"**Tipo:** {tipo}")
            st.info(f"**Descripción:** {desc}")

        CUANTITATIVOS = ["position", "avg_rating", "rating_count", "review_count"]

        if campo in CUANTITATIVOS:
            with col_b:
                d = df[campo].describe().round(4)
                d.index = ["N","Media","Desv. Est.","Mín.","Q1 (25%)","Mediana","Q3 (75%)","Máx."]
                st.dataframe(d.to_frame("Valor"), use_container_width=True)

            fig = px.histogram(df, x=campo, nbins=60,
                               title=f"Distribución de «{campo}»",
                               color_discrete_sequence=["#e94560"])
            fig.update_layout(bargap=0.05)
            st.plotly_chart(fig, use_container_width=True)

        elif campo == "release_date":
            rng_col, _ = st.columns([1, 1])
            rng_col.info(f"Rango: {df['release_date'].min().date()} → {df['release_date'].max().date()}")
            fig = px.histogram(df, x="year", nbins=70,
                               title="Álbumes por Año de Lanzamiento",
                               color_discrete_sequence=["#1e3c72"])
            st.plotly_chart(fig, use_container_width=True)

        elif campo in ["primary_genres", "secondary_genres", "descriptors"]:
            vals = (df[campo].str.split(",").explode().str.strip()
                    .pipe(lambda s: s[s.notna() & (s != "")]).value_counts().head(25))
            st.markdown(f"**Top 25 valores más frecuentes:**")
            st.dataframe(vals.to_frame("Frecuencia"), use_container_width=True)
            fig = px.bar(y=vals.index, x=vals.values, orientation="h",
                         title=f"Top 25 – {campo}",
                         labels={"x": "Frecuencia", "y": campo},
                         color_discrete_sequence=["#1e3c72"])
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)

        else:
            vals = df[campo].value_counts()
            st.info(f"Valores únicos: {df[campo].nunique()}")
            st.dataframe(vals.head(20).to_frame("Frecuencia"), use_container_width=True)

    elif submenu == "Navegador del Dataset":
        st.markdown("### 🗂️ Navegador del Dataset")

        f1, f2, f3 = st.columns(3)
        with f1:
            all_genres = sorted(
                df["primary_genres"].str.split(",").explode().str.strip().dropna().unique().tolist()
            )
            genre_filter = st.multiselect("Género primario:", all_genres)
        with f2:
            y_min, y_max = int(df["year"].min()), int(df["year"].max())
            year_range = st.slider("Años:", y_min, y_max, (y_min, y_max))
        with f3:
            r_min, r_max = float(df["avg_rating"].min()), float(df["avg_rating"].max())
            rating_range = st.slider("Rating:", r_min, r_max, (r_min, r_max), step=0.01)

        fdf = df.copy()
        if genre_filter:
            mask = fdf["primary_genres"].apply(lambda x: any(g in str(x) for g in genre_filter))
            fdf = fdf[mask]
        fdf = fdf[
            fdf["year"].between(year_range[0], year_range[1]) &
            fdf["avg_rating"].between(rating_range[0], rating_range[1])
        ]

        st.info(f"Mostrando **{len(fdf):,}** registros de {len(df):,}")
        show = ["position","release_name","artist_name","year",
                "primary_genres","avg_rating","rating_count","review_count"]
        st.dataframe(
            fdf[show].rename(columns={
                "position": "#", "release_name": "Álbum", "artist_name": "Artista",
                "year": "Año", "primary_genres": "Géneros", "avg_rating": "Rating",
                "rating_count": "# Ratings", "review_count": "Reseñas",
            }),
            use_container_width=True, height=520,
        )

    elif submenu == "Buscador de Registros":
        st.markdown("### 🔎 Buscador de Registros")

        search_by = st.radio("Buscar por:", ["Nombre de álbum", "Nombre de artista", "Posición (#)"],
                             horizontal=True)

        if search_by == "Posición (#)":
            val = st.number_input("Posición:", 1, 5000, 1)
            result = df[df["position"] == val]
        elif search_by == "Nombre de álbum":
            q = st.text_input("Nombre del álbum:")
            result = df[df["release_name"].str.contains(q, case=False, na=False)] if q else pd.DataFrame()
        else:
            q = st.text_input("Nombre del artista:")
            result = df[df["artist_name"].str.contains(q, case=False, na=False)] if q else pd.DataFrame()

        if len(result) > 0:
            for _, row in result.head(6).iterrows():
                with st.expander(f"🎵 #{int(row['position'])} – {row['release_name']} · {row['artist_name']}"):
                    c1, c2, c3 = st.columns(3)
                    c1.metric("⭐ Rating", row["avg_rating"])
                    c2.metric("👥 Ratings", f"{row['rating_count']:,}")
                    c3.metric("📝 Reseñas", f"{row['review_count']:,}")
                    st.write(f"**Géneros:** {row['primary_genres']}")
                    if row["secondary_genres"]:
                        st.write(f"**Géneros secundarios:** {row['secondary_genres']}")
                    st.write(f"**Descriptores:** {row['descriptors']}")
                    st.write(f"**Lanzamiento:** {row['release_date'].date() if pd.notna(row['release_date']) else 'N/D'}")
        elif "q" in dir() and q:
            st.warning("No se encontraron resultados.")

    elif submenu == "Graficador Exploratorio":
        st.markdown("### 📈 Graficador Exploratorio")

        campo = st.selectbox("Campo a graficar:", [
            "avg_rating","rating_count","review_count","position",
            "year","decade","primary_genres","descriptors",
        ])

        CUANTITATIVOS = ["avg_rating","rating_count","review_count","position","year"]

        if campo in CUANTITATIVOS:
            t1, t2, t3 = st.tabs(["📊 Histograma","📦 Boxplot","📈 Distribución Acumulada"])
            with t1:
                fig = px.histogram(df, x=campo, nbins=55,
                                   title=f"Distribución de {campo}",
                                   color_discrete_sequence=["#1e3c72"],
                                   marginal="rug")
                st.plotly_chart(fig, use_container_width=True)
            with t2:
                fig = px.box(df, y=campo, title=f"Boxplot de {campo}",
                             color_discrete_sequence=["#e94560"])
                st.plotly_chart(fig, use_container_width=True)
            with t3:
                fig = px.ecdf(df, x=campo, title=f"Dist. Acumulada de {campo}",
                              color_discrete_sequence=["#e94560"])
                st.plotly_chart(fig, use_container_width=True)

        elif campo == "decade":
            decade_df = df[df["decade"].notna()].copy()
            dec = decade_df.groupby("decade").agg(
                N=("release_name","count"),
                Rating_Prom=("avg_rating","mean")
            ).reset_index()
            dec["Década"] = dec["decade"].astype(str) + "s"
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=dec["Década"], y=dec["N"], name="N° Álbumes",
                                 marker_color="#1e3c72"), secondary_y=False)
            fig.add_trace(go.Scatter(x=dec["Década"], y=dec["Rating_Prom"],
                                     name="Rating Promedio", mode="lines+markers",
                                     marker_color="#e94560", line_width=3), secondary_y=True)
            fig.update_layout(title="Álbumes y Rating Promedio por Década",
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
            fig.update_yaxes(title_text="N° Álbumes", secondary_y=False)
            fig.update_yaxes(title_text="Rating Promedio", secondary_y=True)
            st.plotly_chart(fig, use_container_width=True)

        elif campo in ["primary_genres", "descriptors"]:
            vals = (df[campo].str.split(",").explode().str.strip()
                    .pipe(lambda s: s[s.notna() & (s != "")]).value_counts().head(25))
            fig = px.bar(y=vals.index, x=vals.values, orientation="h",
                         title=f"Top 25 {campo}",
                         labels={"x":"Frecuencia","y": campo},
                         color=vals.values, color_continuous_scale="Blues")
            fig.update_layout(yaxis={"categoryorder":"total ascending"},
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    elif submenu == "Hipótesis":
        st.markdown("### 💡 Hipótesis")

        hip = st.selectbox("Selecciona una hipótesis:", [
            "H1: Los álbumes con más reseñas tienen calificaciones más altas",
            "H2: Los álbumes clásicos (pre-2000) superan en rating a los modernos",
        ])

        if "H1" in hip:
            st.markdown("""
            <div class="hypothesis-box">
            <b>Hipótesis 1:</b> Los álbumes que acumulan más reseñas escritas tienden a tener
            calificaciones promedio más altas, lo que sugiere que la discusión crítica activa
            de la comunidad está ligada a una mejor valoración del álbum.
            </div>
            """, unsafe_allow_html=True)

            corr_rv = df["avg_rating"].corr(df["review_count"])
            corr_rc = df["avg_rating"].corr(df["rating_count"])

            st.markdown("#### Análisis")
            c1, c2 = st.columns(2)

            with c1:
                fig = px.scatter(df.sample(800, random_state=1),
                                 x="review_count", y="avg_rating",
                                 color="avg_rating",
                                 color_continuous_scale="Viridis",
                                 opacity=0.55,
                                 trendline="ols",
                                 title="Reseñas vs Calificación (muestra 800 álbumes)",
                                 labels={"review_count":"N° Reseñas","avg_rating":"Rating"})
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                corr_matrix = df[["avg_rating","review_count","rating_count"]].corr()
                fig = px.imshow(corr_matrix, text_auto=".3f",
                                color_continuous_scale="RdBu",
                                title="Mapa de Correlación")
                st.plotly_chart(fig, use_container_width=True)

            m1, m2, m3 = st.columns(3)
            m1.metric("Corr(reseñas, rating)", f"{corr_rv:.4f}")
            m2.metric("Corr(# ratings, rating)", f"{corr_rc:.4f}")
            m3.metric("Álbumes con >500 reseñas", f"{(df['review_count']>500).sum()}")

            st.markdown("""
            **📌 Conclusión:**
            Existe una correlación positiva entre el número de reseñas y la calificación promedio,
            aunque de intensidad moderada. Los álbumes más discutidos tienden a estar mejor valorados,
            posiblemente porque son obras más icónicas que generan más opiniones críticas.
            Sin embargo, la correlación no es fuerte, indicando que la popularidad escrita no
            garantiza una alta calificación — hay álbumes muy calificados que generan pocas reseñas extensas.
            """)

        else:
            st.markdown("""
            <div class="hypothesis-box">
            <b>Hipótesis 2:</b> Los álbumes lanzados antes del año 2000 (clásicos) tienen
            calificaciones promedio estadísticamente más altas que los álbumes modernos (post-2000),
            debido al sesgo de supervivencia: solo los mejores álbumes del pasado perduran en la memoria colectiva.
            </div>
            """, unsafe_allow_html=True)

            dfv = df[df["year"].notna()].copy()
            dfv["Era"] = dfv["year"].apply(
                lambda y: "Pre-2000 (Clásicos)" if y < 2000 else "Post-2000 (Modernos)"
            )

            c1, c2 = st.columns(2)
            with c1:
                fig = px.box(dfv, x="Era", y="avg_rating",
                             color="Era", title="Rating por Era",
                             color_discrete_map={
                                 "Pre-2000 (Clásicos)": "#1e3c72",
                                 "Post-2000 (Modernos)": "#e94560",
                             })
                st.plotly_chart(fig, use_container_width=True)

            with c2:
                dec_avg = dfv.groupby("decade")["avg_rating"].mean().reset_index()
                fig = px.line(dec_avg, x="decade", y="avg_rating",
                              title="Rating Promedio por Década",
                              markers=True, color_discrete_sequence=["#1e3c72"])
                fig.update_traces(line_width=2.5, marker_size=7)
                fig.add_vline(x=2000, line_dash="dash", line_color="#e94560",
                              annotation_text="Año 2000", annotation_position="top right")
                st.plotly_chart(fig, use_container_width=True)

            pre  = dfv[dfv["year"] < 2000]["avg_rating"]
            post = dfv[dfv["year"] >= 2000]["avg_rating"]
            stat, pval = scipy_stats.mannwhitneyu(pre, post, alternative="greater")

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Media Pre-2000",  f"{pre.mean():.3f}")
            m2.metric("Media Post-2000", f"{post.mean():.3f}")
            m3.metric("Diferencia",      f"{pre.mean()-post.mean():.3f}")
            m4.metric("p-valor (Mann-Whitney)", f"{pval:.4f}")

            if pval < 0.05:
                st.success("✅ Diferencia estadísticamente significativa (p < 0.05) — Hipótesis CONFIRMADA")
            else:
                st.warning("⚠️ Diferencia no significativa estadísticamente")

            st.markdown("""
            **📌 Conclusión:**
            Los datos confirman que los álbumes previos al año 2000 tienen, en promedio,
            calificaciones más altas de forma estadísticamente significativa.
            Este resultado se explica principalmente por el **sesgo de supervivencia**:
            con el paso del tiempo, solo los álbumes más apreciados son recordados y calificados
            activamente. Los álbumes recientes conviven en un mercado más saturado y aún
            no han pasado la prueba del tiempo.
            """)


def render_ml():
    st.markdown("## Aprendizaje Automático - Clasificación")
    st.markdown(
        "Entrena modelos para **predecir si un álbum es altamente calificado o popular** "
        "a partir de variables numéricas del dataset."
    )

    col_a, col_b = st.columns([1, 1], gap="large")

    with col_a:
        algorithm = st.selectbox("🔧 Algoritmo:", [
            "Random Forest",
            "Regresión Logística",
        ])

        target_opt = st.selectbox("🎯 Variable a analizar (target):", [
            "top_rated → álbum altamente calificado (avg_rating ≥ 3.8)",
            "popular  → álbum popular (rating_count ≥ mediana)",
        ])
        target_col = "top_rated" if "top_rated" in target_opt else "popular"

    FEAT_LABELS = {
        "rating_count": "N° de calificaciones",
        "review_count": "N° de reseñas",
        "position":     "Posición en el ranking",
        "year":         "Año de lanzamiento",
    }

    with col_b:
        selected_feats = st.multiselect(
            "📐 Variables independientes (features):",
            options=list(FEAT_LABELS.keys()),
            format_func=lambda x: FEAT_LABELS[x],
            default=["review_count", "year"],
        )

        train_pct = st.slider("📊 % datos de entrenamiento:", 50, 90, 80, step=5)
        test_pct  = 100 - train_pct
        st.caption(f"Entrenamiento: **{train_pct}%**  ·  Prueba: **{test_pct}%**")

    if not selected_feats:
        st.warning("⚠️ Selecciona al menos una variable independiente.")
        return

    df_ml = df[selected_feats + [target_col]].dropna()
    X = df_ml[selected_feats]
    y = df_ml[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_pct / 100, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    Xtr_s  = scaler.fit_transform(X_train)
    Xte_s  = scaler.transform(X_test)

    if algorithm == "Random Forest":
        model = RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1)
    else:
        model = LogisticRegression(max_iter=2000, random_state=42)

    model.fit(Xtr_s, y_train)

    y_pred_tr = model.predict(Xtr_s)
    y_pred_te = model.predict(Xte_s)
    y_prob_te = model.predict_proba(Xte_s)[:, 1]

    acc_tr = accuracy_score(y_train, y_pred_tr)
    acc_te = accuracy_score(y_test,  y_pred_te)
    diff   = acc_tr - acc_te

    st.markdown("---")
    st.markdown("### 📊 Resultados del Modelo")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("🎓 Accuracy – Entrenamiento", f"{acc_tr:.4f}")
    m2.metric("🔬 Accuracy – Prueba",        f"{acc_te:.4f}")
    m3.metric("📚 Muestras Entrenamiento",    f"{len(X_train):,}")
    m4.metric("🧪 Muestras Prueba",           f"{len(X_test):,}")

    if diff > 0.1:
        st.warning(f"⚠️ Posible sobreajuste (diferencia Acc = {diff:.4f})")
    else:
        st.success(f"✅ Modelo bien generalizado (diferencia Acc = {diff:.4f})")

    if algorithm == "Regresión Logística":
        coef_df = pd.DataFrame({
            "Feature": [FEAT_LABELS[f] for f in selected_feats],
            "Coeficiente": model.coef_[0],
        }).sort_values("Coeficiente", ascending=False)
        st.markdown("**Coeficientes del modelo:**")
        fig_coef = px.bar(coef_df, x="Coeficiente", y="Feature", orientation="h",
                          title="Coeficientes – Regresión Logística",
                          color="Coeficiente", color_continuous_scale="RdBu",
                          color_continuous_midpoint=0)
        st.plotly_chart(fig_coef, use_container_width=True)
    else:
        imp_df = pd.DataFrame({
            "Feature":     [FEAT_LABELS[f] for f in selected_feats],
            "Importancia": model.feature_importances_,
        }).sort_values("Importancia", ascending=False)
        st.markdown("**Importancia de variables:**")
        fig_imp = px.bar(imp_df, x="Importancia", y="Feature", orientation="h",
                         title="Importancia de Variables – Random Forest",
                         color="Importancia", color_continuous_scale="Blues")
        fig_imp.update_layout(coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    # ─── Gráficas ───
    tab_pred, tab_cm, tab_roc = st.tabs([
        "📊 Entrenamiento vs Prueba",
        "🔲 Matriz de Confusión",
        "📈 Curva ROC",
    ])

    feat0 = selected_feats[0]
    feat0_label = FEAT_LABELS[feat0]

    with tab_pred:
        # Combine train + test for the graph requirement
        all_plot = pd.DataFrame({
            feat0:       np.concatenate([X_train[feat0].values, X_test[feat0].values]),
            "Clase Real": np.concatenate([y_train.values, y_test.values]),
            "Predicción": np.concatenate([y_pred_tr, y_pred_te]),
            "Conjunto":  ["Entrenamiento"]*len(y_train) + ["Prueba"]*len(y_test),
        })
        all_plot["Resultado"] = all_plot["Clase Real"] == all_plot["Predicción"]
        all_plot["Estado"]    = all_plot["Resultado"].map({True:"✅ Correcto", False:"❌ Incorrecto"})

        fig = px.strip(
            all_plot, x=feat0, y="Clase Real",
            color="Conjunto",
            color_discrete_map={"Entrenamiento":"#1e3c72","Prueba":"#e94560"},
            title=f"Predicciones vs Real · Feature: {feat0_label}",
            labels={feat0: feat0_label, "Clase Real": "Clase Real"},
            stripmode="overlay",
        )
        fig.update_traces(marker_size=5)
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Las formas diferencian predicciones correctas (círculo) de incorrectas (X). "
                   "Los colores distinguen entrenamiento y prueba.")

    with tab_cm:
        cm = confusion_matrix(y_test, y_pred_te)
        labels = ["No (0)", "Sí (1)"]
        fig = px.imshow(cm, text_auto=True,
                        labels=dict(x="Predicción", y="Clase Real"),
                        x=labels, y=labels,
                        color_continuous_scale="Blues",
                        title="Matriz de Confusión – Datos de Prueba")
        st.plotly_chart(fig, use_container_width=True)

        report = classification_report(y_test, y_pred_te, output_dict=True)
        rpt_df = pd.DataFrame(report).T.round(4)
        st.markdown("**Reporte de Clasificación:**")
        st.dataframe(rpt_df, use_container_width=True)

    with tab_roc:
        fpr, tpr, _ = roc_curve(y_test, y_prob_te)
        roc_auc = auc(fpr, tpr)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines",
                                 name=f"Modelo (AUC = {roc_auc:.4f})",
                                 line=dict(color="#e94560", width=2.5)))
        fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines",
                                 name="Clasificador Aleatorio",
                                 line=dict(color="gray", dash="dash", width=1.5)))
        fig.update_layout(title="Curva ROC",
                          xaxis_title="Tasa Falsos Positivos (FPR)",
                          yaxis_title="Tasa Verdaderos Positivos (TPR)")
        st.plotly_chart(fig, use_container_width=True)
        st.metric("📊 AUC-ROC", f"{roc_auc:.4f}")


# ══════════════════════════════════════════════════════════════
# OPCIÓN 4 – SISTEMA DE RECOMENDACIÓN MUSICAL
# ══════════════════════════════════════════════════════════════
def render_recommendation():
    st.markdown("## 🎵 Sistema de Recomendación Musical")
    st.markdown(
        "Recomendación basada en **similitud de contenido** (géneros + descriptores). "
        "Busca un álbum y encuentra los más parecidos en el dataset."
    )

    @st.cache_resource(show_spinner="Construyendo motor de recomendación…")
    def build_tfidf():
        tfidf   = TfidfVectorizer(max_features=600, ngram_range=(1, 2))
        corpus = df["content"].fillna("").astype(str).tolist()
        matrix  = tfidf.fit_transform(corpus)
        return matrix

    tfidf_matrix = build_tfidf()

    def get_recs(pos_int, n=10):
        sims = cosine_similarity(tfidf_matrix[pos_int:pos_int+1], tfidf_matrix)[0]
        order = np.argsort(sims)[::-1]
        order = [i for i in order if i != pos_int][:n]
        return order, sims[order]

    # ─── Búsqueda ───
    query = st.text_input("🔍 Busca un álbum por nombre:",
                          placeholder="Ej: OK Computer, Nevermind, Abbey Road…")

    if query:
        matches = df[df["release_name"].str.contains(query, case=False, na=False)]

        if matches.empty:
            st.warning("No se encontró ningún álbum con ese nombre.")
            return

        options = matches.apply(
            lambda r: f"{r['release_name']} – {r['artist_name']} (#{int(r['position'])})", axis=1
        ).tolist()
        selected = st.selectbox("Selecciona el álbum:", options)
        sel_idx  = matches.iloc[options.index(selected)].name  # DataFrame index value
        sel_pos  = df.index.get_loc(sel_idx)                   # integer position
        sel_row  = df.loc[sel_idx]

        # ─── Info del álbum seleccionado ───
        st.markdown("---")
        st.markdown("### 🎵 Álbum Seleccionado")
        ca, cb, cc = st.columns([3, 1, 1])
        with ca:
            st.markdown(f"**{sel_row['release_name']}** · *{sel_row['artist_name']}*")
            yr = int(sel_row["year"]) if pd.notna(sel_row["year"]) else "N/D"
            st.caption(f"📅 {yr}  ·  🎸 {sel_row['primary_genres']}")
            st.caption(f"🏷️ {sel_row['descriptors']}")
        cb.metric("⭐ Rating", sel_row["avg_rating"])
        cc.metric("🏆 Posición", f"#{int(sel_row['position'])}")

        n_recs = st.slider("Número de recomendaciones:", 5, 20, 10)
        rec_positions, rec_scores = get_recs(sel_pos, n=n_recs)
        rec_indices = [df.index[p] for p in rec_positions]
        recs_df = df.loc[rec_indices].copy()
        recs_df["Similitud"] = rec_scores

        st.markdown(f"### 🎯 Top {n_recs} Recomendaciones")
        for rank, (ridx, row) in enumerate(recs_df.iterrows(), start=1):
            with st.expander(
                f"#{rank} – {row['release_name']} · {row['artist_name']}  "
                f"(Similitud: {row['Similitud']:.3f})"
            ):
                r1, r2, r3 = st.columns([3, 1, 1])
                with r1:
                    st.write(f"🎸 **Géneros:** {row['primary_genres']}")
                    if row["secondary_genres"]:
                        st.write(f"🎸 **Secundarios:** {row['secondary_genres']}")
                    st.write(f"🏷️ **Descriptores:** {row['descriptors']}")
                r2.metric("⭐ Rating", row["avg_rating"])
                r3.metric("🏆 Pos.", f"#{int(row['position'])}")

        # ─── Gráfica de similitud ───
        st.markdown("### 📊 Comparativa de Similitud")
        labels = [
            f"{df.loc[df.index[p], 'release_name']} – {df.loc[df.index[p], 'artist_name']}"
            for p in rec_positions
        ]
        fig = px.bar(x=labels, y=rec_scores,
                     title=f"Similitud con «{sel_row['release_name']}»",
                     labels={"x": "Álbum Recomendado", "y": "Score de Similitud"},
                     color=rec_scores, color_continuous_scale="Blues")
        fig.update_layout(xaxis_tickangle=-40, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("💡 Escribe el nombre de un álbum para comenzar.")
        st.markdown("**Sugerencias populares:**")
        sugs = df.nsmallest(10, "position")[
            ["position","release_name","artist_name","primary_genres","avg_rating"]
        ].rename(columns={
            "position":"#","release_name":"Álbum","artist_name":"Artista",
            "primary_genres":"Géneros","avg_rating":"Rating",
        })
        st.dataframe(sugs, use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════
# OPCIÓN 5 – ANÁLISIS POR CARGA DE ARCHIVOS
# ══════════════════════════════════════════════════════════════
def render_file_analysis():
    st.markdown("## 📁 Análisis de Datos por Carga de Archivos")
    st.markdown("Carga un archivo **CSV o Excel** y explora sus datos automáticamente.")

    uploaded = st.file_uploader(
        "📂 Selecciona un archivo:",
        type=["csv","xlsx","xls"],
        help="Formatos soportados: CSV (.csv), Excel (.xlsx, .xls)",
    )

    if uploaded is None:
        st.info("👆 Carga un archivo para comenzar el análisis.")
        st.markdown("""
        **¿Qué puedes analizar?**
        - Cualquier CSV o Excel con datos tabulares
        - El sistema detecta automáticamente columnas numéricas y categóricas
        - Genera estadísticas, correlaciones y gráficos automáticamente
        """)
        return

    try:
        if uploaded.name.endswith(".csv"):
            try:
                udf = pd.read_csv(uploaded)
            except UnicodeDecodeError:
                udf = pd.read_csv(uploaded, encoding="latin-1")
        else:
            udf = pd.read_excel(uploaded)
    except Exception as e:
        st.error(f"❌ Error al cargar el archivo: {e}")
        return

    st.success(f"✅ **{uploaded.name}** cargado — {len(udf):,} filas · {len(udf.columns)} columnas")

    tab_data, tab_stats, tab_charts = st.tabs([
        "📋 Vista de Datos", "📊 Estadísticas", "📈 Gráficos",
    ])

    num_cols = udf.select_dtypes(include=np.number).columns.tolist()
    cat_cols = udf.select_dtypes(include="object").columns.tolist()

    with tab_data:
        st.markdown(f"**Columnas ({len(udf.columns)}):** {', '.join(udf.columns.tolist())}")
        st.dataframe(udf, use_container_width=True, height=440)

    with tab_stats:
        dtype_df = pd.DataFrame({
            "Tipo":       udf.dtypes.astype(str),
            "No Nulos":   udf.notnull().sum(),
            "Nulos":      udf.isnull().sum(),
            "% Nulos":    (udf.isnull().mean() * 100).round(2),
            "Únicos":     udf.nunique(),
        })
        st.dataframe(dtype_df, use_container_width=True)
        if num_cols:
            st.markdown("**Estadísticas descriptivas (numéricas):**")
            st.dataframe(udf[num_cols].describe().round(4), use_container_width=True)

    with tab_charts:
        if not num_cols and not cat_cols:
            st.warning("No se encontraron columnas graficables.")
            return

        all_cols = num_cols + cat_cols
        col_sel = st.selectbox("Columna a graficar:", all_cols)

        if col_sel in num_cols:
            chart_t = st.radio("Tipo de gráfico:", ["Histograma","Boxplot","Serie"], horizontal=True)
            if chart_t == "Histograma":
                fig = px.histogram(udf, x=col_sel, nbins=40,
                                   title=f"Distribución de {col_sel}",
                                   color_discrete_sequence=["#1e3c72"])
            elif chart_t == "Boxplot":
                fig = px.box(udf, y=col_sel, title=f"Boxplot de {col_sel}",
                             color_discrete_sequence=["#e94560"])
            else:
                fig = px.line(udf.reset_index(), x="index", y=col_sel,
                              title=f"Serie de {col_sel}",
                              color_discrete_sequence=["#1e3c72"])
            st.plotly_chart(fig, use_container_width=True)

        else:
            vc = udf[col_sel].value_counts().head(25)
            fig = px.bar(x=vc.index, y=vc.values,
                         title=f"Frecuencia de {col_sel} (top 25)",
                         labels={"x": col_sel, "y": "Frecuencia"},
                         color_discrete_sequence=["#1e3c72"])
            st.plotly_chart(fig, use_container_width=True)

        # Auto-correlation heatmap
        if len(num_cols) >= 2:
            st.markdown("### 🔗 Correlación entre Variables Numéricas")
            fig = px.imshow(udf[num_cols].corr().round(3),
                            text_auto=True, color_continuous_scale="RdBu",
                            title="Mapa de Calor de Correlaciones")
            st.plotly_chart(fig, use_container_width=True)

        # Pair plot (up to 4 numeric cols)
        if 2 <= len(num_cols) <= 6:
            st.markdown("### 🔀 Matriz de Dispersión")
            fig = px.scatter_matrix(udf[num_cols[:4]],
                                    title="Scatter Matrix – Variables Numéricas",
                                    color_discrete_sequence=["#e94560"])
            st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════
# OPCIÓN 6 – SENTIMIENTOS & SCRAPPING (PITCHFORK)
# ══════════════════════════════════════════════════════════════
def render_sentiment():
    st.markdown("## 💬 Análisis de Sentimientos – Reseñas Musicales (RSS)")
    st.markdown(
        "Extrae reseñas y noticias musicales recientes vía **feeds RSS** de blogs de música como 'Brooklyn Vegan', 'The Line of Best Fit' y 'Stereogum' "
        "y aplica análisis de sentimientos con TextBlob."
    )

    @st.cache_data(ttl=3600, show_spinner="Obteniendo reseñas vía RSS…")
    def scrape_music_rss(n: int):
        """Intenta varios feeds RSS de blogs de música; si todos fallan, datos demo."""
        headers = {"User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        )}

        feeds = [
            ("Brooklyn Vegan", "https://www.brooklynvegan.com/category/music/feed/"),
            ("The Line of Best Fit", "https://www.thelineofbestfit.com/news/feed"),
            ("Stereogum", "https://www.stereogum.com/feed/"),
        ]

        for fuente, rss_url in feeds:
            try:
                resp = requests.get(rss_url, headers=headers, timeout=12)
                soup = BeautifulSoup(resp.content, "lxml-xml")
                items = soup.find_all("item")

                reviews = []
                for it in items[:n]:
                    raw_title = it.find("title")
                    raw_desc  = it.find("description") or it.find("summary")
                    title_txt = raw_title.get_text(strip=True) if raw_title else "N/D"
                    desc_html = raw_desc.get_text(strip=True) if raw_desc else ""
                    desc_txt  = BeautifulSoup(desc_html, "html.parser").get_text(" ", strip=True)[:500]

                    if desc_txt and len(desc_txt) > 30:
                        reviews.append({
                            "Álbum": title_txt,
                            "Artista": fuente,
                            "Puntuación": "N/D",
                            "Reseña": desc_txt,
                        })

                if reviews:
                    return reviews[:n], "rss"
            except Exception:
                continue

        # ── Fallback: datos de demostración ──
        demo = [
            {"Álbum": "Bright Future", "Artista": "Adrianne Lenker",
             "Puntuación": "N/D",
             "Reseña": ("A quietly intimate record where every note feels handled with "
                        "care. The minimalist production lets the songwriting breathe "
                        "with real emotional weight.")},
            {"Álbum": "Manning Fireworks", "Artista": "MJ Lenderman",
             "Puntuación": "N/D",
             "Reseña": ("A relaxed, guitar-driven indie rock collection full of wit and "
                        "laid-back confidence. Lenderman sounds completely comfortable "
                        "in his own skin.")},
            {"Álbum": "Short n' Sweet", "Artista": "Sabrina Carpenter",
             "Puntuación": "N/D",
             "Reseña": ("The hooks are undeniable, but the record plays things safe. "
                        "Some tracks shine while others blend into the current pop "
                        "landscape without leaving much of a mark.")},
            {"Álbum": "Chromakopia", "Artista": "Tyler, the Creator",
             "Puntuación": "N/D",
             "Reseña": ("An ambitious, deeply personal project showing real growth as a "
                        "producer and lyricist. It does not always cohere, but the "
                        "highs are genuinely striking.")},
            {"Álbum": "Diamond Jubilee", "Artista": "Cindy Lee",
             "Puntuación": "N/D",
             "Reseña": ("A sprawling, hypnotic journey that rewards patience. Despite "
                        "its unconventional length, the record feels remarkably "
                        "cohesive and quietly ambitious.")},
            {"Álbum": "GNX", "Artista": "Kendrick Lamar",
             "Puntuación": "N/D",
             "Reseña": ("Kendrick remains in firm control of his craft, blending "
                        "reflection with confidence. The production is meticulous "
                        "and the writing keeps finding new angles.")},
            {"Álbum": "Imaginal Disk", "Artista": "Magdalena Bay",
             "Puntuación": "N/D",
             "Reseña": ("A maximalist, endlessly inventive synth-pop record. The duo "
                        "wraps conceptual ambition around genuinely catchy "
                        "songwriting.")},
            {"Álbum": "Wall of Eyes", "Artista": "The Smile",
             "Puntuación": "N/D",
             "Reseña": ("A more restrained outing than their debut. The arrangements "
                        "are elegant, though the album occasionally lacks the urgency "
                        "of their earlier work.")},
        ]
        return demo[:n], "demo"

    def analyze_sentiment(text: str):
        blob = TextBlob(str(text))
        pol  = blob.sentiment.polarity
        sub  = blob.sentiment.subjectivity
        lbl  = "😊 Positivo" if pol > 0.08 else ("😞 Negativo" if pol < -0.08 else "😐 Neutral")
        return pol, sub, lbl

    n_reviews = st.slider("N° de reseñas a obtener:", 4, 30, 8)

    if st.button("🔍 Obtener y Analizar Reseñas", type="primary"):
        with st.spinner("Conectando con feeds RSS de música…"):
            reviews, source = scrape_music_rss(n_reviews)

        if source == "demo":
            st.warning("⚠️ No fue posible conectar con los feeds RSS en este momento. "
                       "Se muestran datos de demostración para ilustrar el análisis.")
        else:
            st.success("✅ Reseñas obtenidas vía RSS feed")

        results = []
        for r in reviews:
            pol, sub, lbl = analyze_sentiment(r["Reseña"])
            results.append({**r, "Polaridad": pol, "Subjetividad": sub, "Sentimiento": lbl})

        rdf = pd.DataFrame(results)
        st.session_state["rss_results"] = rdf

    # ─── Mostrar resultados ───
    if "rss_results" in st.session_state:
        rdf = st.session_state["rss_results"]
        st.markdown(f"---\n### 📰 {len(rdf)} Reseñas Analizadas")

        mc1, mc2, mc3 = st.columns(3)
        mc1.metric("😊 Positivas",  (rdf["Sentimiento"] == "😊 Positivo").sum())
        mc2.metric("😐 Neutrales",  (rdf["Sentimiento"] == "😐 Neutral").sum())
        mc3.metric("😞 Negativas",  (rdf["Sentimiento"] == "😞 Negativo").sum())

        # ─── Tarjetas de reseñas ───
        st.markdown("### 📋 Reseñas y Sentimiento")
        for _, row in rdf.iterrows():
            with st.expander(
                f"{row['Sentimiento']}  ·  **{row['Álbum']}** – {row['Artista']}  "
                f"(Score: {row['Puntuación']})"
            ):
                st.markdown(f"*{row['Reseña']}*")
                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Polaridad",    f"{row['Polaridad']:.3f}")
                rc2.metric("Subjetividad", f"{row['Subjetividad']:.3f}")
                rc3.metric("Sentimiento",  row["Sentimiento"])

        # ─── Gráficas ───
        gc1, gc2 = st.columns(2)
        with gc1:
            sc = rdf["Sentimiento"].value_counts()
            fig = px.pie(values=sc.values, names=sc.index,
                         title="Distribución de Sentimientos",
                         color_discrete_sequence=["#1e3c72","#e94560","#f5a623"],
                         hole=0.4)
            st.plotly_chart(fig, use_container_width=True)

        with gc2:
            fig = px.scatter(
                rdf, x="Polaridad", y="Subjetividad",
                color="Sentimiento", text="Álbum", size_max=10,
                title="Polaridad vs Subjetividad",
                color_discrete_map={
                    "😊 Positivo": "#1e3c72",
                    "😐 Neutral":  "#f5a623",
                    "😞 Negativo": "#e94560",
                },
            )
            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.add_hline(y=0.5, line_dash="dot", line_color="gray")
            fig.update_traces(textposition="top center", textfont_size=9)
            st.plotly_chart(fig, use_container_width=True)

        # Polaridad bar
        rdf_sorted = rdf.sort_values("Polaridad")
        fig = px.bar(rdf_sorted, x="Polaridad",
                     y=rdf_sorted["Álbum"] + " – " + rdf_sorted["Artista"],
                     orientation="h",
                     color="Polaridad", color_continuous_scale="RdYlGn",
                     title="Polaridad por Álbum",
                     labels={"y": ""})
        fig.add_vline(x=0, line_color="white", line_dash="dash")
        fig.update_layout(coloraxis_showscale=False, yaxis={"categoryorder":"total ascending"})
        st.plotly_chart(fig, use_container_width=True)

def render_ai_prompts():

    # ── Base de conocimiento ──────────────────────────────────────────────
    ORIGINAL_COLS = [
        "position", "release_name", "artist_name", "release_date", "release_type",
        "primary_genres", "secondary_genres", "descriptors",
        "avg_rating", "rating_count", "review_count",
    ]

    DECADE_KEYWORDS = {
        "cincuenta": 1950, "50s": 1950,
        "sesenta": 1960, "60s": 1960,
        "setenta": 1970, "70s": 1970,
        "ochenta": 1980, "80s": 1980,
        "noventa": 1990, "90s": 1990,
        "2000s": 2000, "dosmil": 2000,
        "2010s": 2010,
        "2020s": 2020,
    }

    def responder(pregunta: str) -> str:
        p = pregunta.lower()
        for a, b in [("á","a"),("é","e"),("í","i"),("ó","o"),("ú","u"),("¿",""),("?","")]:
            p = p.replace(a, b)

        # ── Preguntas sobre el dataset ──────────────────────────────────
        if any(x in p for x in ["cuantas columnas","numero de columnas","cuantos campos","cuantas variables"]):
            return f"📋 El dataset tiene **{len(ORIGINAL_COLS)} columnas**:\n\n" + \
                   "\n".join(f"- `{c}`" for c in ORIGINAL_COLS)

        if any(x in p for x in ["cuantos registros","cuantas filas","cuantos datos","tamano","cuantos albumes tiene"]):
            return f"📊 El dataset tiene **{len(df):,} registros** (álbumes) y **{len(ORIGINAL_COLS)} columnas**."

        m = re.search(r"(media|promedio|mean)\s+(?:del?\s+campo\s+)?(\w+)", p)
        if m:
            campo = m.group(2)
            match = next((c for c in ["avg_rating","rating_count","review_count","position","year"]
                         if campo in c.lower() or c.lower() in campo), None)
            if match:
                return f"📈 La media del campo **{match}** es **{df[match].mean():.4f}**."
            return ("⚠️ No encontré ese campo numérico. Campos disponibles: "
                    "`avg_rating`, `rating_count`, `review_count`, `position`, `year`.")

        m2 = re.search(r"(maximo|mayor valor|max)\s+(?:del?\s+campo\s+)?(\w+)", p)
        if m2:
            campo = m2.group(2)
            match = next((c for c in ["avg_rating","rating_count","review_count","position","year"]
                         if campo in c.lower() or c.lower() in campo), None)
            if match:
                return f"🔺 El valor máximo del campo **{match}** es **{df[match].max()}**."

        m3 = re.search(r"(minimo|menor valor|min)\s+(?:del?\s+campo\s+)?(\w+)", p)
        if m3:
            campo = m3.group(2)
            match = next((c for c in ["avg_rating","rating_count","review_count","position","year"]
                         if campo in c.lower() or c.lower() in campo), None)
            if match:
                return f"🔻 El valor mínimo del campo **{match}** es **{df[match].min()}**."

        if any(x in p for x in ["valores unicos","categorias","opciones de release_type","tipo de lanzamiento"]):
            vals = df["release_type"].unique().tolist()
            return f"🏷️ Los valores únicos de **release_type** son:\n\n" + \
                   "\n".join(f"- `{v}`" for v in vals)

        if any(x in p for x in ["album con mas calificaciones","album mas calificado","mas ratings","album mas popular"]):
            row = df.loc[df["rating_count"].idxmax()]
            return (f"🏆 El álbum con más calificaciones es **{row['release_name']}** "
                    f"de *{row['artist_name']}* con **{int(row['rating_count']):,}** ratings "
                    f"(rating promedio: {row['avg_rating']}).")

        if any(x in p for x in ["album con mas resenas","album con mas reviews"]):
            row = df.loc[df["review_count"].idxmax()]
            return (f"📝 El álbum con más reseñas escritas es **{row['release_name']}** "
                    f"de *{row['artist_name']}* con **{int(row['review_count']):,}** reseñas.")

        if any(x in p for x in ["artista con mas albumes","artista mas frecuente","artista con mas discos"]):
            vc = df["artist_name"].value_counts().head(5)
            lines = "\n".join(f"- **{k}**: {v} álbumes" for k, v in vc.items())
            return f"🎤 Top 5 artistas con más álbumes en el dataset:\n\n{lines}"

        if any(x in p for x in ["generos mas comunes","generos hay","que generos","top generos"]):
            vc = (df["primary_genres"].str.split(",").explode().str.strip().value_counts().head(8))
            lines = "\n".join(f"- **{k}**: {v} álbumes" for k, v in vc.items())
            return f"🎸 Los 8 géneros primarios más comunes son:\n\n{lines}"

        if any(x in p for x in ["genero con mejor rating","genero con mayor rating","genero mejor calificado"]):
            g = (df.assign(g=df["primary_genres"].str.split(",").str[0].str.strip())
                 .groupby("g")["avg_rating"].mean().idxmax())
            val = (df.assign(g=df["primary_genres"].str.split(",").str[0].str.strip())
                   .groupby("g")["avg_rating"].mean().max())
            return f"🌟 El género con mayor rating promedio es **{g}** ({val:.3f})."

        if any(x in p for x in ["rango de anos","que anos cubre","periodo del dataset","desde que ano"]):
            return f"📅 El dataset cubre desde **{int(df['year'].min())}** hasta **{int(df['year'].max())}**."

        for kw, dec in DECADE_KEYWORDS.items():
            if kw in p and ("decada" in p or "anos" in p or "albumes" in p):
                sub = df[df["decade"] == dec]
                return (f"📀 Hay **{len(sub)} álbumes** de la década de **{dec}s**, "
                        f"con un rating promedio de **{sub['avg_rating'].mean():.3f}**.")

        if "correlacion" in p:
            num_cols = ["avg_rating", "rating_count", "review_count", "position"]
            corr = df[num_cols].corr()["avg_rating"].drop("avg_rating").sort_values(ascending=False)
            lines = "\n".join(f"- **{k}**: {v:.4f}" for k, v in corr.items())
            return (f"🔗 Correlación con **avg_rating** (calificación promedio):\n\n{lines}\n\n"
                    "Valores cercanos a 1 indican correlación positiva fuerte; cercanos a -1, negativa fuerte.")

        if any(x in p for x in ["estadisticas","describe","resumen general","distribucion general"]):
            cols = ["avg_rating", "rating_count", "review_count", "position", "year"]
            desc = df[cols].describe().round(2)
            return f"📊 Estadísticas descriptivas del dataset:\n\n```\n{desc.to_string()}\n```"

        # ── Conceptos de Machine Learning (sección del portafolio) ───────
        respuestas_ml = {
            "accuracy": ("✅ **Accuracy** es la proporción de predicciones correctas sobre el total.\n\n"
                        "`Accuracy = Predicciones correctas / Total`\n\n"
                        "En la sección de Aprendizaje Automático de este portafolio, se muestra el accuracy "
                        "de entrenamiento y de prueba para detectar sobreajuste."),
            "regresion logistica": ("📐 **Regresión Logística** calcula la *probabilidad* de que un álbum "
                                    "pertenezca a una clase (por ejemplo, ser 'altamente calificado').\n\n"
                                    "Usa una función sigmoide que transforma cualquier valor al rango [0, 1]. "
                                    "Si la probabilidad supera 0.5, predice la clase positiva."),
            "random forest": ("🌳 **Random Forest** combina muchos árboles de decisión entrenados con "
                              "subconjuntos distintos de datos.\n\n"
                              "Cada árbol vota y gana la mayoría. Esto reduce el sobreajuste y mejora "
                              "la precisión comparado con un solo árbol."),
            "arbol de decision": ("🌲 Un **Árbol de Decisión** divide los datos con reglas sucesivas, por ejemplo:\n\n"
                                  "- Si `review_count` > 500 → rama derecha\n- Si `avg_rating` ≥ 3.8 → clase 'top_rated'\n\n"
                                  "Random Forest es, en esencia, un conjunto de muchos árboles como este."),
            "sobreajuste": ("⚠️ **Sobreajuste (Overfitting)** ocurre cuando el modelo memoriza los datos de "
                            "entrenamiento en lugar de aprender patrones generales.\n\n"
                            "Se detecta cuando el accuracy de entrenamiento es mucho mayor al de prueba."),
            "entrenamiento": ("🎓 Dividir los datos en **entrenamiento y prueba** es esencial para evaluar "
                              "el modelo de forma honesta.\n\n"
                              "- **Entrenamiento**: el modelo aprende los patrones\n"
                              "- **Prueba**: se evalúa con datos que el modelo no vio antes\n\n"
                              "En este portafolio puedes ajustar ese porcentaje con un control deslizante."),
            "confusion": ("🔲 La **Matriz de Confusión** muestra los aciertos y errores del modelo:\n\n"
                         "- **TP**: predijo positivo y era positivo ✅\n"
                         "- **TN**: predijo negativo y era negativo ✅\n"
                         "- **FP**: predijo positivo pero era negativo ❌\n"
                         "- **FN**: predijo negativo pero era positivo ❌"),
            "curva roc": ("📈 La **Curva ROC** grafica la tasa de verdaderos positivos contra la tasa de "
                         "falsos positivos en distintos umbrales de decisión.\n\n"
                         "El **AUC** (área bajo la curva) resume el rendimiento: 1.0 es un clasificador "
                         "perfecto, 0.5 equivale a adivinar al azar."),
            "tf-idf": ("🔤 **TF-IDF** convierte texto (géneros y descriptores) en vectores numéricos, "
                      "dando más peso a palabras distintivas y menos a las muy comunes.\n\n"
                      "Se usa en el Sistema de Recomendación de este portafolio."),
            "tfidf": ("🔤 **TF-IDF** convierte texto (géneros y descriptores) en vectores numéricos, "
                     "dando más peso a palabras distintivas y menos a las muy comunes.\n\n"
                     "Se usa en el Sistema de Recomendación de este portafolio."),
            "similitud coseno": ("📐 La **Similitud de Coseno** mide el ángulo entre dos vectores de texto.\n\n"
                                "Ángulo 0° = similitud 1 (idénticos). Ángulo 90° = similitud 0 (sin relación).\n\n"
                                "Se usa para recomendar álbumes parecidos según género y descriptores."),
            "textblob": ("💬 **TextBlob** es una librería de Python para procesamiento de lenguaje natural.\n\n"
                        "Calcula **polaridad** (-1 negativo a 1 positivo) y **subjetividad** (0 objetivo a "
                        "1 subjetivo). Se usa en la sección de Análisis de Sentimientos."),
            "scraping": ("🌐 El **Web Scraping** extrae datos automáticamente de páginas web.\n\n"
                        "Esta app usa `requests` para descargar feeds RSS de blogs de música y "
                        "`BeautifulSoup` para extraer el texto de cada reseña."),
            "rss": ("📰 Un **feed RSS** es un archivo XML que los sitios web publican con su contenido "
                   "más reciente en formato estructurado.\n\n"
                   "Es más estable para scraping que el HTML normal, porque rara vez cambia su formato."),
            "clasificacion": ("🤖 La **Clasificación** es una técnica de Aprendizaje Automático que predice "
                              "una categoría (no un número continuo).\n\n"
                              "En este portafolio se usa para predecir si un álbum es 'altamente calificado' "
                              "o 'popular', usando Random Forest o Regresión Logística."),
        }

        for clave, respuesta in respuestas_ml.items():
            if clave in p:
                return respuesta

        # ── Descripción general del dataset ───────────────────────────────
        if any(x in p for x in ["dataset","datos","que contiene","de que trata"]):
            return (f"🎵 El dataset **RateYourMusic Top 5000** tiene **{len(df):,} registros** y "
                    f"**{len(ORIGINAL_COLS)} columnas**.\n\n"
                    "Contiene los álbumes más populares de la comunidad de RateYourMusic.com: "
                    "nombre, artista, fecha de lanzamiento, géneros, descriptores de atmósfera, "
                    "calificación promedio, número de calificaciones y de reseñas.\n\n"
                    f"Cubre el período de **{int(df['year'].min())}** a **{int(df['year'].max())}**.")

        return ("🤔 No encontré una respuesta para esa pregunta. Prueba con:\n\n"
                "- *¿Cuántas columnas tiene el dataset?*\n"
                "- *¿Cuál es la media del avg_rating?*\n"
                "- *¿Cuál es el álbum con más calificaciones?*\n"
                "- *¿Qué géneros son los más comunes?*\n"
                "- *¿Qué es el accuracy?*\n"
                "- *¿Qué es Random Forest?*\n"
                "- *Correlación con avg_rating*")

    # ── Vista ────────────────────────────────────────────────────────────
    SUGERENCIAS = [
        "¿Cuántas columnas tiene el dataset?",
        "¿Cuál es la media del avg_rating?",
        "¿Cuál es el álbum con más calificaciones?",
        "¿Qué géneros son los más comunes?",
        "¿Qué es el accuracy en clasificación?",
        "¿Cuál es el artista con más álbumes?",
        "¿Qué es Random Forest?",
        "Correlación con avg_rating",
        "¿Cuántos álbumes son de los 90s?",
        "¿Qué es la matriz de confusión?",
        "¿Cómo funciona el sistema de recomendación?",
        "¿Qué es TextBlob?",
    ]

    st.markdown("""
    <div style='background:linear-gradient(135deg,#1e3c72,#e94560);
                padding:1.8rem;border-radius:14px;color:white;margin-bottom:1.5rem;
                box-shadow:0 4px 18px rgba(233,69,96,0.25);'>
        <h2 style='margin:0;'>🧠 Interfaz IA — Asistente del Portafolio</h2>
        <p style='margin:0.4rem 0 0 0;opacity:0.9;font-size:0.95rem;'>
            Haz preguntas sobre el dataset de RateYourMusic o sobre conceptos de Ciencia de Datos.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = []

    # Sugerencias como chips
    if not st.session_state.ai_messages:
        st.markdown("**💡 Preguntas que puedes hacer:**")
        cols = st.columns(2)
        for i, sugg in enumerate(SUGERENCIAS):
            with cols[i % 2]:
                if st.button(sugg, key=f"ai_s_{i}", use_container_width=True):
                    st.session_state.ai_messages.append({"role": "user", "content": sugg})
                    resp = responder(sugg)
                    st.session_state.ai_messages.append({"role": "assistant", "content": resp})
                    st.rerun()
        st.markdown("---")

    # Historial de mensajes
    for msg in st.session_state.ai_messages:
        if msg["role"] == "user":
            st.markdown(f"""
            <div style='display:flex;justify-content:flex-end;margin-bottom:0.6rem;'>
                <div style='background:#e94560;color:white;padding:0.8rem 1.1rem;
                            border-radius:16px 16px 4px 16px;max-width:75%;
                            font-size:0.94rem;box-shadow:0 2px 8px rgba(233,69,96,0.3);'>
                    {msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='display:flex;justify-content:flex-start;margin-bottom:0.6rem;'>
                <div style='background:#13132b;color:#e8e8f0;padding:0.8rem 1.1rem;
                            border-radius:16px 16px 16px 4px;max-width:80%;
                            font-size:0.94rem;border:1px solid #2a3a6a;'>
                    {msg["content"]}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Input
    prompt = st.chat_input("Escribe tu pregunta sobre el dataset o Ciencia de Datos…")
    if prompt:
        st.session_state.ai_messages.append({"role": "user", "content": prompt})
        resp = responder(prompt)
        st.session_state.ai_messages.append({"role": "assistant", "content": resp})
        st.rerun()

    # Limpiar
    if st.session_state.ai_messages:
        st.markdown("---")
        if st.button("🗑️ Limpiar conversación", use_container_width=False):
            st.session_state.ai_messages = []
            st.rerun()

    # Ayuda
    with st.expander("❓ Preguntas de ejemplo"):
        st.markdown("""
        **Sobre el dataset:**
        - ¿Cuántas columnas tiene el dataset?
        - ¿Cuántos registros tiene el dataset?
        - ¿Cuál es la media del avg_rating?
        - ¿Cuál es el máximo de rating_count?
        - ¿Cuáles son los valores de release_type?
        - ¿Cuál es el álbum con más calificaciones?
        - ¿Qué artista tiene más álbumes?
        - ¿Qué géneros son los más comunes?
        - ¿Qué género tiene mejor rating promedio?
        - ¿Cuántos álbumes son de los 90s?
        - Correlación con avg_rating
        - Estadísticas del dataset

        **Sobre Machine Learning:**
        - ¿Qué es el accuracy?
        - ¿Qué es la regresión logística?
        - ¿Qué es Random Forest?
        - ¿Qué es el sobreajuste?
        - ¿Qué es la matriz de confusión?
        - ¿Qué es la curva ROC?

        **Sobre la aplicación:**
        - ¿Cómo funciona el sistema de recomendación?
        - ¿Qué es TF-IDF?
        - ¿Qué es la similitud de coseno?
        - ¿Cómo funciona el scraping?
        - ¿Qué es TextBlob?
        """)

# ══════════════════════════════════════════════════════════════
# ENRUTADOR PRINCIPAL
# ══════════════════════════════════════════════════════════════
if   page == "🏠 Inicio":                render_home()
elif page == "📊 Análisis Exploratorio": render_eda()
elif page == "🤖 Aprendizaje Automático": render_ml()
elif page == "🎵 Sistema de Recomendación": render_recommendation()
elif page == "📁 Análisis por Archivos": render_file_analysis()
elif page == "💬 Sentimientos & Scrapping": render_sentiment()
elif page == "🧠 Prompts de IA":         render_ai_prompts()
