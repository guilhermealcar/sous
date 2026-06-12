import os
import streamlit as st
from PIL import Image
import folium
from streamlit_folium import st_folium
from streamlit_geolocation import streamlit_geolocation
from ultralytics import YOLO

# ==================================================
# CONFIG
# ==================================================
st.set_page_config(
    page_title="• sous",
    page_icon="⚪",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Inicializa o Session State para o GPS não sumir ao recarregar a página
if "lat" not in st.session_state:
    st.session_state["lat"] = None
if "lon" not in st.session_state:
    st.session_state["lon"] = None

# ==================================================
# MODEL
# ==================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(BASE_DIR, "runs", "detect", "sous_v2_augmented", "weights", "best.pt")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"Arquivo de pesos não encontrado em: {MODEL_PATH}. Aguarde o fim do treinamento.")
        st.stop()
    return YOLO(MODEL_PATH)

model = load_model()

# ==================================================
# CSS
# ==================================================
st.markdown("""
<style>
[data-testid="stSidebar"]{ display:none; }
[data-testid="stHeader"]{ background:transparent; }
[data-testid="stAppViewContainer"]{ background:#0B0F14; }
html, body, [class*="css"]{ font-family: Inter, sans-serif; }

.logo-text{
    text-align:center;
    color:white;
    font-size:72px;
    font-weight:300;
    letter-spacing:-2px;
    margin-top:30px;
}
.subtitle{
    text-align:center;
    color:#8D99AE;
    font-size:18px;
    margin-bottom:30px;
}
.card{
    border:1px solid rgba(255,255,255,0.08);
    border-radius:18px;
    padding:20px;
    background:#111827;
}
.metric{
    padding:12px;
    border-radius:12px;
    background:#161F2D;
    margin-bottom:10px;
    color:white;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# HEADER
# ==================================================
st.markdown("<div class='logo-text'>• sous</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Smart Outdoor Urban Cleaning System<br>Waste detection powered by AI</div>", unsafe_allow_html=True)

# ==================================================
# GEOLOCATION
# ==================================================
st.markdown("## Passo 1: Permita a Localização")
st.write("Clique no botão abaixo para capturar o GPS antes de enviar a foto.")

location = streamlit_geolocation()

if location and location.get("latitude") and location.get("longitude"):
    st.session_state["lat"] = location.get("latitude")
    st.session_state["lon"] = location.get("longitude")

# Resgata os valores do estado da sessão
lat = st.session_state["lat"]
lon = st.session_state["lon"]

if lat and lon:
    st.success("✅ Localização capturada com sucesso!")

# ==================================================
# IMAGE INPUT
# ==================================================
st.markdown("## Passo 2: Capture a Imagem")

camera_image = st.camera_input("", label_visibility="collapsed")

st.markdown("<div style='text-align:center;color:#8D99AE;margin-top:15px'>Upload for testing</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

image_source = uploaded_file if uploaded_file else camera_image

# ==================================================
# DETECTION
# ==================================================
if image_source:
    image = Image.open(image_source).convert("RGB")

    with st.spinner("Running detection..."):
        results = model.predict(
            source=image,
            conf=0.45, 
            verbose=False
        )

    result = results[0]
    
    # CORREÇÃO: YOLO gera imagem em BGR. Precisamos inverter para RGB.
    annotated_image_bgr = result.plot()
    annotated_image_rgb = annotated_image_bgr[..., ::-1] 

    counts = {
        "plastic": 0, "metal": 0, "glass": 0, 
        "paper": 0, "cigarette": 0, "other": 0
    }
    confidences = []

    if result.boxes is not None:
        for cls, conf in zip(result.boxes.cls, result.boxes.conf):
            class_id = int(cls)
            class_name = result.names[class_id]

            if class_name.lower() in counts:
                counts[class_name.lower()] += 1
            confidences.append(float(conf))

    left, right = st.columns([2, 1])

    with left:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Detected Waste")
        st.image(annotated_image_rgb, use_container_width=True) # Usa a imagem RGB corrigida
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.subheader("Detection Summary")

        total_objects = sum(counts.values())
        avg_conf = (sum(confidences) / len(confidences)) if confidences else 0

        st.markdown(f"""
        <div class='metric'>Plastic: {counts['plastic']}</div>
        <div class='metric'>Metal: {counts['metal']}</div>
        <div class='metric'>Glass: {counts['glass']}</div>
        <div class='metric'>Paper: {counts['paper']}</div>
        <div class='metric'>Cigarette: {counts['cigarette']}</div>
        <div class='metric'>Other: {counts['other']}</div>
        <div class='metric' style='border: 1px solid #3B82F6;'><b>Total: {total_objects}</b></div>
        <div class='metric'>Confidence: {avg_conf:.2%}</div>
        """, unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ==================================================
    # MAP
    # ==================================================
    st.subheader("Location Map")

    if lat is not None and lon is not None:
        m = folium.Map(
            location=[lat, lon],
            zoom_start=18,
            tiles="CartoDB dark_matter"
        )

        folium.CircleMarker(
            [lat, lon],
            radius=8,
            fill=True,
            color="#EF4444", # Vermelho para dar contraste com o fundo escuro
            fill_color="#EF4444",
            fill_opacity=0.7,
            tooltip="Waste location"
        ).add_to(m)

        # CORREÇÃO: use_container_width para responsividade total
        st_folium(m, height=450, use_container_width=True)

        st.info(f"Latitude: {lat:.6f} | Longitude: {lon:.6f}")
    else:
        st.warning("Location permission not granted. Please click the GPS button in Step 1 before capturing the image.")