import streamlit as st
import cv2
import numpy as np
import easyocr
from ultralytics import YOLO
import re
from PIL import Image

# Configuracion de la pagina - Estetica limpia
st.set_page_config(page_title="Plataforma ANPR v2.0", layout="wide")

# CSS personalizado para paleta pastel rosa y azul
st.markdown("""
    <style>
    .main {
        background-color: #fdfdfd;
    }
    .stButton>button {
        background-color: #B5D8EB;
        color: #1A252F;
        border-radius: 10px;
        border: none;
        font-weight: bold;
        padding: 10px 24px;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #F8D1E0;
        transform: scale(1.02);
    }
    .header-style {
        background: linear-gradient(135deg, #B5D8EB 0%, #F8D1E0 100%);
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    h1 { color: #2C3E50 !important; font-family: 'Inter', sans-serif; }
    h3 { color: #5D6D7E !important; }
    </style>
    """, unsafe_allow_html=True)

# Cache para no recargar modelos en cada clic
@st.cache_resource
def load_models():
    # Uso el modelo medium como aprendi en clase con Miguel para equilibrar
    model = YOLO("yolov8m.pt")
    reader = easyocr.Reader(["en"], gpu=False)
    return model, reader

model_vehicles, reader = load_models()
VEHICLE_CLASSES = [2, 3, 5, 7]

def clean_plate_text(text):
    text = re.sub(r"[^A-Za-z0-9]", "", text)
    return text.upper()

def process_frame(img_np):
    results = model_vehicles(img_np, conf=0.25, verbose=False)
    plate_text = "No identificada"
    
    for result in results:
        for box in result.boxes:
            if int(box.cls[0]) in VEHICLE_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                # Recorte del vehiculo para buscar matricula
                roi = img_np[y1:y2, x1:x2]
                
                # Preprocesado clasico
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                enhanced = cv2.createCLAHE(clipLimit=2.0).apply(gray)
                
                # OCR
                ocr_results = reader.readtext(enhanced)
                for (_, text, conf) in ocr_results:
                    cleaned = clean_plate_text(text)
                    if len(cleaned) >= 4:
                        plate_text = cleaned
                        # Dibujo el resultado en la imagen
                        cv2.rectangle(img_np, (x1, y1), (x2, y2), (181, 216, 235), 4)
                        cv2.putText(img_np, plate_text, (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (248, 209, 224), 3)
                        break
    return img_np, plate_text

# Layout de la aplicacion
st.markdown('<div class="header-style"><h1>Sistema de Analisis de Matriculas</h1><p>Motor de vision computacional en entorno de produccion</p></div>', unsafe_allow_html=True)

tab1, tab2 = st.tabs(["Terminal de Escaneo", "Especificaciones Tecnicas"])

with tab1:
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Modulo de entrada")
        uploaded_file = st.file_uploader("Cargue una imagen de vehiculo (JPG, PNG)", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            st.image(image, caption="Imagen original cargada", use_column_width=True)
            
            if st.button("Ejecutar Escaneo Profesional"):
                with st.spinner("Procesando pipeline multimodelo..."):
                    res_img, plate = process_frame(img_array)
                    
                with col2:
                    st.subheader("Panel de resultados")
                    st.image(res_img, caption="Deteccion y segmentacion aplicada", use_column_width=True)
                    st.success(f"Matricula extraida: {plate}")
                    st.info("El sistema ha verificado los datos mediante filtrado alfanumerico.")

with tab2:
    st.markdown("""
    ### Arquitectura del Sistema
    Este MVP profesional utiliza un pipeline de tres etapas:
    1. **Deteccion Espacial:** Localizacion del vehiculo mediante YOLOv8m (ajustado para velocidad/precision).
    2. **Optimizacion de Imagen:** Ecualizacion adaptativa de histograma para mitigar reflejos y sombras.
    3. **Motor OCR:** Extraccion mediante redes neuronales convolucionales y recurrentes (EasyOCR).
    """)