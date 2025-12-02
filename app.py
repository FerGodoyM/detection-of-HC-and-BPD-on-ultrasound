"""
Aplicación Web Educativa para Medición de HC y BPD en Ecografías Fetales
Para estudiantes de obstetricia
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
import os
import random
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import math

app = Flask(__name__)

# Configuración de rutas
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "DATASET HC18")
TRAINING_SET = os.path.join(DATASET_DIR, "training_set")
TEST_SET = os.path.join(DATASET_DIR, "test_set")
MODEL_PATH = os.path.join(BASE_DIR, "unet_hc18_best.h5")

# Cargar modelo U-Net
model = None

def cargar_modelo():
    """Carga el modelo U-Net"""
    global model
    if model is None:
        print("Cargando modelo U-Net...")
        model = keras.models.load_model(MODEL_PATH)
        print("Modelo cargado exitosamente")
    return model

def obtener_imagenes_disponibles():
    """Obtiene lista de imágenes disponibles para entrenamiento"""
    imagenes = []
    
    # Cargar datos de entrenamiento con HC real
    csv_training = os.path.join(DATASET_DIR, "training_set_pixel_size_and_HC.csv")
    if os.path.exists(csv_training):
        df = pd.read_csv(csv_training)
        for _, row in df.iterrows():
            filename = row['filename']
            img_path = os.path.join(TRAINING_SET, filename)
            if os.path.exists(img_path):
                imagenes.append({
                    'filename': filename,
                    'path': img_path,
                    'pixel_size': row['pixel size(mm)'],
                    'hc_real': row['head circumference (mm)'],
                    'set': 'training'
                })
    
    return imagenes

# Cache de imágenes
IMAGENES_CACHE = None

def get_imagenes():
    global IMAGENES_CACHE
    if IMAGENES_CACHE is None:
        IMAGENES_CACHE = obtener_imagenes_disponibles()
    return IMAGENES_CACHE

def calcular_hc_ramanujan(a, b):
    """Calcula la circunferencia usando la fórmula de Ramanujan"""
    h = ((a - b) ** 2) / ((a + b) ** 2)
    return math.pi * (a + b) * (1 + (3 * h) / (10 + math.sqrt(4 - 3 * h)))

def predecir_con_ia(imagen_path, pixel_size):
    """Realiza la predicción con el modelo U-Net"""
    modelo = cargar_modelo()
    
    # Cargar y preprocesar imagen
    img = cv2.imread(imagen_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    
    h_orig, w_orig = img.shape
    img_resized = cv2.resize(img, (256, 256))
    img_norm = img_resized.astype(np.float32) / 255.0
    img_input = img_norm.reshape(1, 256, 256, 1)
    
    # Predecir
    pred = modelo.predict(img_input, verbose=0)[0, :, :, 0]
    
    # Post-procesar
    pred_binary = (pred > 0.5).astype(np.uint8) * 255
    
    # Operaciones morfológicas
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    pred_clean = cv2.morphologyEx(pred_binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    pred_clean = cv2.morphologyEx(pred_clean, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(pred_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Usar el contorno más grande
    contour = max(contours, key=cv2.contourArea)
    
    if len(contour) < 5:
        return None
    
    # Ajustar elipse
    ellipse = cv2.fitEllipse(contour)
    center, axes, angle = ellipse
    
    # Escalar al tamaño original
    scale_x = w_orig / 256
    scale_y = h_orig / 256
    
    center_orig = (center[0] * scale_x, center[1] * scale_y)
    axes_orig = (axes[0] * scale_x, axes[1] * scale_y)
    
    # Calcular HC
    a = max(axes_orig) / 2 * pixel_size
    b = min(axes_orig) / 2 * pixel_size
    hc_mm = calcular_hc_ramanujan(a, b)
    
    # Calcular BPD (diámetro menor de la elipse)
    bpd_mm = min(axes_orig) * pixel_size
    
    return {
        'center': center_orig,
        'axes': axes_orig,
        'angle': angle,
        'hc_mm': hc_mm,
        'bpd_mm': bpd_mm
    }

@app.route('/')
def index():
    """Página principal"""
    return render_template('index.html')

@app.route('/api/imagen-aleatoria')
def imagen_aleatoria():
    """Devuelve una imagen aleatoria para practicar"""
    imagenes = get_imagenes()
    if not imagenes:
        return jsonify({'error': 'No hay imágenes disponibles'}), 404
    
    img_data = random.choice(imagenes)
    
    return jsonify({
        'filename': img_data['filename'],
        'pixel_size': img_data['pixel_size'],
        'hc_real': img_data['hc_real'],
        'set': img_data['set']
    })

@app.route('/api/imagen/<path:filename>')
def servir_imagen(filename):
    """Sirve una imagen del dataset"""
    # Buscar en training_set
    img_path = os.path.join(TRAINING_SET, filename)
    if os.path.exists(img_path):
        return send_from_directory(TRAINING_SET, filename)
    
    # Buscar en test_set
    img_path = os.path.join(TEST_SET, filename)
    if os.path.exists(img_path):
        return send_from_directory(TEST_SET, filename)
    
    return jsonify({'error': 'Imagen no encontrada'}), 404

@app.route('/api/calcular-medidas', methods=['POST'])
def calcular_medidas():
    """Calcula HC y BPD a partir de la elipse dibujada por el usuario"""
    data = request.json
    
    center_x = data.get('centerX')
    center_y = data.get('centerY')
    axis_a = data.get('axisA')  # Semi-eje mayor
    axis_b = data.get('axisB')  # Semi-eje menor
    pixel_size = data.get('pixelSize')
    
    if not all([center_x, center_y, axis_a, axis_b, pixel_size]):
        return jsonify({'error': 'Faltan parámetros'}), 400
    
    # Convertir a mm
    a_mm = axis_a * pixel_size
    b_mm = axis_b * pixel_size
    
    # Calcular HC usando Ramanujan
    hc_mm = calcular_hc_ramanujan(a_mm, b_mm)
    
    # BPD es el diámetro menor
    bpd_mm = min(axis_a, axis_b) * 2 * pixel_size
    
    return jsonify({
        'hc_mm': round(hc_mm, 2),
        'bpd_mm': round(bpd_mm, 2)
    })

@app.route('/api/calcular-bpd', methods=['POST'])
def calcular_bpd():
    """Calcula BPD a partir de la línea dibujada por el usuario"""
    data = request.json
    
    x1 = data.get('x1')
    y1 = data.get('y1')
    x2 = data.get('x2')
    y2 = data.get('y2')
    pixel_size = data.get('pixelSize')
    
    if not all([x1 is not None, y1 is not None, x2 is not None, y2 is not None, pixel_size]):
        return jsonify({'error': 'Faltan parámetros'}), 400
    
    # Calcular longitud en píxeles
    longitud_px = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Convertir a mm
    bpd_mm = longitud_px * pixel_size
    
    return jsonify({
        'bpd_mm': round(bpd_mm, 2)
    })

@app.route('/api/prediccion-ia', methods=['POST'])
def prediccion_ia():
    """Obtiene la predicción de la IA para una imagen"""
    try:
        data = request.json
        print(f"[IA] Recibido: {data}")
        
        filename = data.get('filename')
        pixel_size = data.get('pixelSize')
        
        if not filename or not pixel_size:
            print(f"[IA] Error: Faltan parámetros - filename={filename}, pixelSize={pixel_size}")
            return jsonify({'error': 'Faltan parámetros'}), 400
        
        # Buscar imagen
        img_path = os.path.join(TRAINING_SET, filename)
        if not os.path.exists(img_path):
            img_path = os.path.join(TEST_SET, filename)
        
        if not os.path.exists(img_path):
            print(f"[IA] Error: Imagen no encontrada - {filename}")
            return jsonify({'error': 'Imagen no encontrada'}), 404
        
        print(f"[IA] Procesando imagen: {img_path}")
        resultado = predecir_con_ia(img_path, pixel_size)
        
        if resultado is None:
            print(f"[IA] Error: La predicción devolvió None")
            return jsonify({'error': 'No se pudo realizar la predicción'}), 500
        
        print(f"[IA] Resultado: HC={resultado['hc_mm']:.2f}mm")
        return jsonify({
            'center': {'x': resultado['center'][0], 'y': resultado['center'][1]},
            'axes': {'a': resultado['axes'][0] / 2, 'b': resultado['axes'][1] / 2},
            'angle': resultado['angle'],
            'hc_mm': round(resultado['hc_mm'], 2),
            'bpd_mm': round(resultado['bpd_mm'], 2)
        })
    except Exception as e:
        print(f"[IA] Excepción: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("=" * 60)
    print("🏥 APLICACIÓN EDUCATIVA DE OBSTETRICIA")
    print("   Medición de HC y BPD en Ecografías Fetales")
    print("=" * 60)
    
    # Pre-cargar modelo
    cargar_modelo()
    
    # Verificar imágenes
    imagenes = get_imagenes()
    print(f"📁 Imágenes disponibles: {len(imagenes)}")
    
    print("\n🌐 Iniciando servidor web...")
    print("   Abrir en navegador: http://localhost:5000")
    print("=" * 60)
    
    app.run(debug=True, host='0.0.0.0', port=5000)
