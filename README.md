# 🎵 Portafolio de Ciencia de Datos — RateYourMusic Top 5 000

Aplicación Streamlit desarrollada para el curso **Técnica Electiva I – Ciencia de Datos**,
Ingeniería en Sistemas y Redes Informáticas, Ciclo I-2026.

---

## 📁 Estructura del proyecto

```
portfolio/
├── app.py                  ← Aplicación principal
├── requirements.txt        ← Dependencias Python
├── .streamlit/
│   └── config.toml         ← Tema visual oscuro
├── data/
│   └── rym_clean1.csv      ← Dataset (debes incluirlo)
└── assets/                 ← (Opcional) foto y video local
    └── foto.jpg
```

---

## 🚀 Cómo desplegar en Streamlit Cloud

### 1. Crear repositorio en GitHub

1. Ve a [github.com](https://github.com) y crea un repositorio **público** (ej. `portafolio-ds`).
2. Sube todos los archivos de esta carpeta manteniendo la estructura.
   - **Importante:** incluye la carpeta `data/` con el archivo `rym_clean1.csv`.

```bash
git init
git add .
git commit -m "Portafolio DS - Ciencia de Datos Ciclo I-2026"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/portafolio-ds.git
git push -u origin main
```

### 2. Conectar con Streamlit Cloud

1. Ve a [share.streamlit.io](https://share.streamlit.io) e inicia sesión con tu cuenta de GitHub.
2. Haz clic en **"New app"**.
3. Selecciona tu repositorio, rama `main` y archivo principal `app.py`.
4. Haz clic en **"Deploy"** y espera ~2 minutos.

### 3. Configurar la API Key de IA (Opción 7)

Para que la opción **Prompts de IA** funcione sin ingresar la clave manualmente:

1. En tu app de Streamlit Cloud, ve a **Settings → Secrets**.
2. Agrega el siguiente contenido:

```toml
ANTHROPIC_API_KEY = "sk-ant-TU_CLAVE_AQUI"
```

Obtén tu clave en [console.anthropic.com](https://console.anthropic.com).

---

## ✏️ Personalización de la página de Inicio

Edita `app.py` en la función `render_home()` para:

### Agregar tu fotografía
```python
# Reemplaza el bloque "dashed-box" de foto por:
st.image("assets/foto.jpg", use_column_width=True, caption="Tu Nombre")
```

### Agregar tu video
```python
# Para video en YouTube:
st.video("https://youtu.be/TU_VIDEO_ID")

# Para video local (sube el archivo a assets/):
st.video("assets/demo.mp4")
```

### Agregar tu resumen personal
Reemplaza el bloque `dashed-box` del resumen con:
```python
st.markdown("""
Tu resumen personal de 3 a 5 líneas aquí...
""")
```

---

## 📋 Funcionalidades incluidas

| Opción | Descripción |
|--------|-------------|
| 🏠 Inicio | Portafolio profesional con foto, video y resumen |
| 📊 Análisis Exploratorio | Descripción, navegador, buscador, graficador, hipótesis |
| 🤖 Aprendizaje Automático | Clasificación con Random Forest y Regresión Logística |
| 🎵 Sistema de Recomendación | Recomendación musical por similitud de contenido (TF-IDF) |
| 📁 Análisis por Archivos | Carga CSV/Excel y genera gráficos automáticamente |
| 💬 Sentimientos & Scrapping | Extrae reseñas de Pitchfork y analiza sentimientos |
| 🧠 Prompts de IA | Consultas en lenguaje natural sobre el dataset con Claude |

---

## 🔧 Ejecución local (opcional)

```bash
# Instalar dependencias
pip install -r requirements.txt

# Descargar datos de TextBlob (primera vez)
python -m textblob.download_corpora

# Ejecutar la aplicación
streamlit run app.py
```

---

## 📌 Notas importantes

- **Opción 6 (Pitchfork):** Si el scraping falla por restricciones del servidor,
  la app muestra automáticamente datos de demostración para ilustrar el análisis.
- **Opción 7 (IA):** Requiere API Key de Anthropic. Configúrala en Streamlit Secrets
  o ingrésala directamente en la barra lateral de la app.
- El archivo `rym_clean1.csv` **debe estar en la carpeta `data/`** para que la app funcione.
