# ==============================================================
# VERSIÓN CORREGIDA - VALIDACIÓN ADAPTATIVA
# ==============================================================

import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from scipy import ndimage
from skimage import morphology
import os
import matplotlib.pyplot as plt

# Rutas base
BASE_DIR = r"C:\Users\ferna\OneDrive\Escritorio\PROYECTO OBSTETRICIA"
DATASET_DIR = os.path.join(BASE_DIR, "DATASET HC18")
TRAIN_DIR = os.path.join(DATASET_DIR, "training_set")
TEST_DIR = os.path.join(DATASET_DIR, "test_set")
MODEL_PATH = os.path.join(BASE_DIR, "unet_hc18_best.h5")

# Cargar modelo una sola vez (manejo de excepciones por si falta)
try:
    model = tf.keras.models.load_model(MODEL_PATH)
except Exception as e:
    print(f"⚠️ No se pudo cargar el modelo en '{MODEL_PATH}': {e}")
    model = None

# Cargar CSVs (cache en variables globales)
_train_df = None
_test_df = None

def _load_csv_data():
    global _train_df, _test_df
    if _train_df is None:
        train_csv = os.path.join(DATASET_DIR, "training_set_pixel_size_and_HC.csv")
        if os.path.isfile(train_csv):
            _train_df = pd.read_csv(train_csv)
        else:
            _train_df = pd.DataFrame(columns=["filename","pixel size(mm)","head circumference (mm)"])
    if _test_df is None:
        test_csv = os.path.join(DATASET_DIR, "test_set_pixel_size.csv")
        if os.path.isfile(test_csv):
            _test_df = pd.read_csv(test_csv)
        else:
            _test_df = pd.DataFrame(columns=["filename","pixel size(mm)"])

_load_csv_data()

def _select_row(df, img_id):
    """Selecciona fila más apropiada para un id (preferencia *_HC.png exacto)."""
    # Primero coincidencia exacta id_HC.png
    exact = df[df['filename'] == f"{img_id}_HC.png"]
    if not exact.empty:
        return exact.iloc[0]
    # Luego filas que contienen id (duplicados _2HC etc.)
    subset = df[df['filename'].str.startswith(f"{img_id}")]
    if not subset.empty:
        # Si hay varias, tomar promedio de numeric columns
        if 'head circumference (mm)' in subset.columns:
            avg_pixel = subset['pixel size(mm)'].astype(float).mean()
            avg_hc = subset['head circumference (mm)'].astype(float).mean()
            return pd.Series({
                'filename': f"{img_id}_HC.png",
                'pixel size(mm)': avg_pixel,
                'head circumference (mm)': avg_hc
            })
        else:
            avg_pixel = subset['pixel size(mm)'].astype(float).mean()
            return pd.Series({
                'filename': f"{img_id}_HC.png",
                'pixel size(mm)': avg_pixel
            })
    return None

def get_HC_real(img_id: str):
    """Devuelve HC real (mm) si está disponible en el CSV de entrenamiento."""
    row = _select_row(_train_df, img_id)
    if row is None:
        return None
    return float(row.get('head circumference (mm)', np.nan)) if 'head circumference (mm)' in row.index else None

def load_sample_with_scaling(img_id: str, target_size: int = 256):
    """Carga imagen asociada al id y ajusta escalado devolviendo pixel_size corregido.

    Retorna: img_norm (float32 [target_size,target_size]), pixel_size_corr (mm/px), orig_w, orig_h
    """
    # Buscar fila en train primero, luego test
    row = _select_row(_train_df, img_id)
    source_df = 'train'
    if row is None:
        row = _select_row(_test_df, img_id)
        source_df = 'test'
    if row is None:
        raise FileNotFoundError(f"No hay metadatos para id {img_id}")

    filename = row['filename']
    pixel_size = float(row['pixel size(mm)'])

    # Construir ruta
    possible_paths = [
        os.path.join(TRAIN_DIR, filename),
        os.path.join(TEST_DIR, filename)
    ]
    img_path = None
    for p in possible_paths:
        if os.path.isfile(p):
            img_path = p
            break
    if img_path is None:
        raise FileNotFoundError(f"Imagen '{filename}' no encontrada en train/test set")

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Error leyendo imagen {img_path}")

    orig_h, orig_w = img.shape

    # Redimensionar manteniendo proporción (interpolación área)
    resized = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)

    # Ajustar pixel size por factor de escala promedio
    scale_x = orig_w / target_size
    scale_y = orig_h / target_size
    scale_avg = (scale_x + scale_y) / 2.0
    pixel_size_corr = pixel_size * scale_avg

    # Normalizar igual que entrenamiento (0-1)
    resized_norm = (resized.astype(np.float32) / 255.0)

    return resized_norm, pixel_size_corr, orig_w, orig_h

# ==============================================================
# ETAPA 1: POST-PROCESAMIENTO MEJORADO
# ==============================================================

def refinar_prediccion_unet_v2(pred_raw, umbral_confianza=0.3):
    """
    Post-procesamiento optimizado para HC18
    """
    pred_norm = cv2.normalize(pred_raw, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # Umbralización más permisiva
    _, binary = cv2.threshold(pred_norm, int(umbral_confianza * 255), 255, cv2.THRESH_BINARY)

    # Morfología progresiva
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))

    # Cerrar gaps pequeños
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_small, iterations=2)

    # Cerrar gaps medianos
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_medium, iterations=2)

    # Cerrar gaps grandes (para fragmentación severa)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_large, iterations=2)

    # Rellenar huecos
    binary_filled = ndimage.binary_fill_holes(binary).astype(np.uint8) * 255

    # Limpiar ruido pequeño
    binary_clean = morphology.remove_small_objects(
        binary_filled.astype(bool),
        min_size=300  # Más permisivo
    ).astype(np.uint8) * 255

    # Suavizado suave
    binary_smooth = cv2.GaussianBlur(binary_clean, (5, 5), 1)
    _, binary_final = cv2.threshold(binary_smooth, 127, 255, cv2.THRESH_BINARY)

    return binary_final, pred_norm

# ==============================================================
# ETAPA 2: DETECCIÓN CON VALIDACIÓN ADAPTATIVA
# ==============================================================

def detectar_contorno_adaptativo(binary_mask, pred_norm, HC_esperado=None, debug=True):
    """
    Detección con umbrales adaptativos según el tamaño esperado
    """
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    if len(contours) == 0:
        if debug:
            print("   ⚠️ No se encontraron contornos")
        return None, None

    img_area = binary_mask.shape[0] * binary_mask.shape[1]

    if debug:
        print(f"   📊 Contornos encontrados: {len(contours)}")

    # Filtro adaptativo
    def validar_contorno_adaptativo(contorno, img_shape):
        """
        Validación más flexible basada en el dataset HC18
        """
        if len(contorno) < 10:
            return False, {}

        area = cv2.contourArea(contorno)

        # Rango adaptativo de área (10-70% de la imagen)
        area_ratio = area / img_area

        if area_ratio < 0.08 or area_ratio > 0.75:
            if debug:
                print(f"      ✗ Área fuera de rango: {area_ratio:.3f}")
            return False, {}

        try:
            if len(contorno) < 5:
                return False, {}

            ellipse = cv2.fitEllipse(contorno)
            (cx, cy), (major, minor), angle = ellipse

            # Aspect ratio más permisivo (1.0 - 2.0)
            aspect_ratio = max(major, minor) / (min(major, minor) + 1e-6)
            if aspect_ratio > 2.5:  # Muy permisivo
                if debug:
                    print(f"      ✗ AR muy alta: {aspect_ratio:.2f}")
                return False, {}

            # Circularidad más permisiva
            perimeter = cv2.arcLength(contorno, True)
            if perimeter == 0:
                return False, {}

            circularity = 4 * np.pi * area / (perimeter ** 2)

            # Umbral MUY bajo para casos fragmentados
            if circularity < 0.40:  # Antes era 0.65
                if debug:
                    print(f"      ✗ Circularidad baja: {circularity:.3f}")
                return False, {}

            # Posición más permisiva
            img_center = np.array([img_shape[1]/2, img_shape[0]/2])
            ellipse_center = np.array([cx, cy])
            center_distance = np.linalg.norm(ellipse_center - img_center)
            center_ratio = center_distance / max(img_shape)

            # Score simplificado
            score = (
                circularity * 0.4 +
                area_ratio * 0.3 +
                (1.0 - min(center_ratio, 0.5) / 0.5) * 0.2 +
                (1.0 / aspect_ratio) * 0.1  # Favorece círculos
            )

            metrics = {
                'area': area,
                'area_ratio': area_ratio,
                'circularity': circularity,
                'aspect_ratio': aspect_ratio,
                'center_ratio': center_ratio,
                'score': score,
                'ellipse': ellipse,
                'center': (cx, cy),
                'axes': (major, minor)
            }

            if debug:
                print(f"      ✓ Candidato válido: Circ={circularity:.3f}, AR={aspect_ratio:.2f}, "
                      f"Area={area_ratio:.3f}, Score={score:.3f}")

            return True, metrics

        except Exception as e:
            if debug:
                print(f"      ✗ Error en validación: {e}")
            return False, {}

    # Evaluar todos los contornos
    candidatos = []

    for i, cnt in enumerate(contours):
        if debug:
            print(f"   🔍 Evaluando contorno {i+1}/{len(contours)}...")

        es_valido, metrics = validar_contorno_adaptativo(cnt, binary_mask.shape)

        if es_valido:
            candidatos.append((cnt, metrics))

    if not candidatos:
        if debug:
            print("   ⚠️ Ningún contorno pasó validación (intentando modo de emergencia)")

        # MODO DE EMERGENCIA: Tomar el contorno más grande sin validar
        if len(contours) > 0:
            contorno_mayor = max(contours, key=cv2.contourArea)

            if len(contorno_mayor) >= 5:
                try:
                    ellipse_emergency = cv2.fitEllipse(contorno_mayor)

                    metrics_emergency = {
                        'area': cv2.contourArea(contorno_mayor),
                        'area_ratio': cv2.contourArea(contorno_mayor) / img_area,
                        'circularity': 0.0,
                        'aspect_ratio': 0.0,
                        'center_ratio': 0.0,
                        'score': 0.0,
                        'ellipse': ellipse_emergency,
                        'modo': 'emergencia'
                    }

                    if debug:
                        print("   🚨 Usando modo de emergencia (contorno más grande)")

                    return contorno_mayor, metrics_emergency
                except:
                    pass

        return None, None

    # Seleccionar mejor candidato
    mejor_contorno, mejores_metrics = max(candidatos, key=lambda x: x[1]['score'])

    if debug:
        print(f"   ✅ Mejor contorno seleccionado: Score={mejores_metrics['score']:.3f}")

    return mejor_contorno, mejores_metrics

# ==============================================================
# ETAPA 3: CÁLCULO DE HC ROBUSTO
# ==============================================================

def calcular_HC_robusto(contorno, ellipse, pixel_size, HC_esperado=None):
    """
    Cálculo con múltiples métodos y selección inteligente
    """
    (cx, cy), (major, minor), angle = ellipse

    # Asegurar que major > minor
    if major < minor:
        major, minor = minor, major

    # Conversión a mm
    a_mm = (major / 2) * pixel_size
    b_mm = (minor / 2) * pixel_size

    # Método 1: Ramanujan (estándar clínico)
    h = ((a_mm - b_mm)**2) / ((a_mm + b_mm)**2 + 1e-10)
    HC_ramanujan = np.pi * (a_mm + b_mm) * (1 + (3*h) / (10 + np.sqrt(4 - 3*h) + 1e-10))

    # Método 2: Perímetro del contorno
    perimetro_px = cv2.arcLength(contorno, True)
    HC_contorno = perimetro_px * pixel_size

    # Método 3: Aproximación exacta
    HC_exact = np.pi * (a_mm + b_mm) * (1 + h/4 + h**2/64 + h**3/256)

    # Selección inteligente
    # Si el contorno es muy fragmentado, confiar más en la elipse
    circularity = 4 * np.pi * cv2.contourArea(contorno) / (perimetro_px**2 + 1e-10)

    if circularity > 0.80:
        # Contorno limpio -> promediar
        HC_final = (HC_ramanujan * 0.5 + HC_contorno * 0.3 + HC_exact * 0.2)
    elif circularity > 0.60:
        # Contorno aceptable -> confiar más en Ramanujan
        HC_final = (HC_ramanujan * 0.7 + HC_contorno * 0.2 + HC_exact * 0.1)
    else:
        # Contorno fragmentado -> solo Ramanujan
        HC_final = HC_ramanujan

    # BPD y OFD
    BPD_mm = minor * pixel_size
    OFD_mm = major * pixel_size

    stats = {
        'HC_mm': HC_final,
        'HC_ramanujan': HC_ramanujan,
        'HC_contorno': HC_contorno,
        'HC_exact': HC_exact,
        'BPD_mm': BPD_mm,
        'OFD_mm': OFD_mm,
        'aspect_ratio': major / minor,
        'circularity': circularity,      
        'metodo_usado': 'mixto' if circularity > 0.60 else 'ramanujan'
    }

    if HC_esperado:
        error_abs = abs(HC_final - HC_esperado)
        error_pct = (error_abs / HC_esperado) * 100
        stats['error_abs'] = error_abs
        stats['error_pct'] = error_pct

        emoji = '✅' if error_pct < 5 else '✓' if error_pct < 10 else '⚠️'
        print(f"   {emoji} HC={HC_final:.2f}mm (Real={HC_esperado:.2f}mm, Error={error_pct:.2f}%)")
        print(f"   📊 Ramanujan={HC_ramanujan:.2f}, Contorno={HC_contorno:.2f}, Exacto={HC_exact:.2f}")

    return HC_final, BPD_mm, stats

# ==============================================================
# PIPELINE COMPLETO OPTIMIZADO
# ==============================================================

def medir_HC_pipeline_optimizado(pred_raw, pixel_size, HC_esperado=None, visualizar=False, debug=True):
    """
    Pipeline optimizado con fallbacks
    """
    if debug:
        print(f"\n{'─'*70}")
        print("🔬 PIPELINE OPTIMIZADO - GRADO MÉDICO")
        print(f"{'─'*70}")

    # Etapa 1: Post-procesamiento
    if debug:
        print("📍 Etapa 1: Post-procesamiento de predicción...")

    binary_mask, pred_norm = refinar_prediccion_unet_v2(pred_raw, umbral_confianza=0.3)

    # Etapa 2: Detección de contorno
    if debug:
        print("📍 Etapa 2: Detección de contorno...")

    contorno, metrics = detectar_contorno_adaptativo(binary_mask, pred_norm, HC_esperado, debug=debug)

    if contorno is None:
        print("❌ FALLO: No se pudo detectar contorno válido")

        if visualizar:
            # Mostrar diagnóstico
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            axes[0].imshow(pred_raw, cmap='hot')
            axes[0].set_title('Predicción Original')
            axes[0].axis('off')

            axes[1].imshow(pred_norm, cmap='gray')
            axes[1].set_title('Normalizada')
            axes[1].axis('off')

            axes[2].imshow(binary_mask, cmap='gray')
            axes[2].set_title('Máscara Binaria (sin contornos válidos)')
            axes[2].axis('off')

            plt.tight_layout()
            plt.show()

        return None, None, None, None

    # Etapa 3: Ajuste de elipse
    if debug:
        print("📍 Etapa 3: Ajuste de elipse...")

    ellipse = metrics.get('ellipse')

    if ellipse is None:
        ellipse = cv2.fitEllipse(contorno)

    # Etapa 4: Cálculo de HC
    if debug:
        print("📍 Etapa 4: Cálculo de HC...")

    HC_mm, BPD_mm, stats = calcular_HC_robusto(contorno, ellipse, pixel_size, HC_esperado)

    # Combinar métricas
    stats.update(metrics)

    # Visualización
    if visualizar:
        visualizar_pipeline_detallado(pred_raw, pred_norm, binary_mask, contorno,
                                       ellipse, stats, HC_esperado)

    if debug:
        print(f"{'─'*70}\n")

    return HC_mm, BPD_mm, ellipse, stats

# ==============================================================
# VISUALIZACIÓN DETALLADA
# ==============================================================

def visualizar_pipeline_detallado(pred_raw, pred_norm, binary_mask, contorno,
                                   ellipse, stats, HC_esperado=None):
    """
    Visualización completa del pipeline
    """
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

    # Fila 1: Procesamiento
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(pred_raw, cmap='hot')
    ax1.set_title('1. Predicción U-Net', fontsize=10, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(pred_norm, cmap='gray')
    ax2.set_title('2. Normalizada', fontsize=10, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(binary_mask, cmap='gray')
    ax3.set_title('3. Máscara Binaria', fontsize=10, fontweight='bold')
    ax3.axis('off')

    # Contorno detectado
    ax4 = fig.add_subplot(gs[0, 3])
    cont_img = cv2.cvtColor(pred_norm, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(cont_img, [contorno], -1, (0, 255, 0), 2)
    ax4.imshow(cont_img)
    ax4.set_title('4. Contorno Detectado', fontsize=10, fontweight='bold')
    ax4.axis('off')

    # Fila 2: Resultados
    ax5 = fig.add_subplot(gs[1, 0])
    result_img = cv2.cvtColor(pred_norm, cv2.COLOR_GRAY2BGR)
    cv2.ellipse(result_img, ellipse, (0, 255, 255), 2)
    (cx, cy), _, _ = ellipse
    cv2.circle(result_img, (int(cx), int(cy)), 3, (255, 0, 0), -1)
    ax5.imshow(result_img)
    ax5.set_title('5. Elipse Ajustada', fontsize=10, fontweight='bold')
    ax5.axis('off')

    # Overlay
    ax6 = fig.add_subplot(gs[1, 1])
    overlay = cv2.addWeighted(
        cv2.cvtColor(pred_norm, cv2.COLOR_GRAY2BGR),
        0.7,
        cv2.cvtColor(binary_mask, cv2.COLOR_GRAY2BGR),
        0.3,
        0
    )
    cv2.ellipse(overlay, ellipse, (0, 255, 255), 2)
    ax6.imshow(overlay)
    ax6.set_title('6. Overlay', fontsize=10, fontweight='bold')
    ax6.axis('off')

    # Resultado final anotado
    ax7 = fig.add_subplot(gs[1, 2:])
    final_img = cv2.cvtColor((pred_norm * 0.7).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    cv2.ellipse(final_img, ellipse, (0, 255, 255), 3)
    cv2.circle(final_img, (int(cx), int(cy)), 5, (255, 0, 0), -1)

    # Anotaciones
    y_pos = 30
    anotaciones = [
        f"HC: {stats['HC_mm']:.2f} mm",
        f"BPD: {stats['BPD_mm']:.2f} mm",
        f"OFD: {stats['OFD_mm']:.2f} mm",
    ]

    if HC_esperado:
        color = (0, 255, 0) if stats['error_pct'] < 5 else \
                (0, 255, 255) if stats['error_pct'] < 10 else (0, 0, 255)
        anotaciones.extend([
            f"Real: {HC_esperado:.2f} mm",
            f"Error: {stats['error_pct']:.2f}%"
        ])
    else:
        color = (0, 255, 255)

    for texto in anotaciones:
        cv2.putText(final_img, texto, (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        y_pos += 35

    ax7.imshow(final_img)
    ax7.set_title('7. Resultado Final', fontsize=10, fontweight='bold')
    ax7.axis('off')

    # Fila 3: Métricas
    ax8 = fig.add_subplot(gs[2, :2])
    ax8.axis('off')

    metricas_texto = f"""
    MÉTRICAS DE CALIDAD:
    ══════════════════════════════
    • Circularidad: {stats.get('circularity', 0):.3f}
    • Aspect Ratio: {stats.get('aspect_ratio', 0):.2f}
    • Área (ratio): {stats.get('area_ratio', 0):.3f}
    • Quality Score: {stats.get('score', 0):.3f}

    MÉTODOS DE CÁLCULO:
    ══════════════════════════════
    • Ramanujan: {stats.get('HC_ramanujan', 0):.2f} mm
    • Contorno: {stats.get('HC_contorno', 0):.2f} mm
    • Exacto: {stats.get('HC_exact', 0):.2f} mm
    • Método usado: {stats.get('metodo_usado', 'N/A')}
    """

    ax8.text(0.1, 0.5, metricas_texto, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Gráfico de comparación
    if HC_esperado:
        ax9 = fig.add_subplot(gs[2, 2:])

        categorias = ['Predicho', 'Real']
        valores = [stats['HC_mm'], HC_esperado]
        colores = ['#00CED1', '#32CD32']

        bars = ax9.bar(categorias, valores, color=colores, alpha=0.7, edgecolor='black', linewidth=2)

        # Anotar valores
        for bar, valor in zip(bars, valores):
            height = bar.get_height()
            ax9.text(bar.get_x() + bar.get_width()/2., height,
                     f'{valor:.1f} mm',
                     ha='center', va='bottom', fontsize=12, fontweight='bold')

        ax9.set_ylabel('HC (mm)', fontsize=11, fontweight='bold')
        ax9.set_title(f'Comparación (Error: {stats["error_pct"]:.2f}%)',
                      fontsize=10, fontweight='bold')
        ax9.grid(axis='y', alpha=0.3)
        ax9.set_ylim(0, max(valores) * 1.2)

    plt.suptitle('PIPELINE DE MEDICIÓN DE HC - GRADO MÉDICO',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.show()

# ==============================================================
# FUNCIÓN DE PROCESAMIENTO
# ==============================================================

def procesar_con_pipeline_optimizado(image_names, visualizar_todos=False):
    """
    Procesa múltiples imágenes con el pipeline optimizado
    """
    resultados = []

    for img_name in image_names:
        print(f"\n{'='*70}")
        print(f"📁 PROCESANDO: {img_name}")
        print('='*70)

        try:
            # Cargar imagen
            img, pixel_size, orig_w, orig_h = load_sample_with_scaling(img_name)
            HC_real = get_HC_real(img_name)

            print(f"📐 Dimensiones: {orig_w}x{orig_h} | Pixel: {pixel_size:.6f}mm")
            if HC_real:
                print(f"🎯 HC Real: {HC_real:.2f}mm")

            # Predicción
            pred = model.predict(np.expand_dims(img, axis=(0, -1)), verbose=0)[0].squeeze()

            # Pipeline
            HC_pred, BPD_pred, ellipse, stats = medir_HC_pipeline_optimizado(
                pred, pixel_size, HC_esperado=HC_real,
                visualizar=visualizar_todos, debug=True
            )

            if HC_pred and HC_real:
                resultados.append({
                    'nombre': img_name,
                    'HC_real': HC_real,
                    'HC_pred': HC_pred,
                    'BPD_pred': BPD_pred,
                    'stats': stats,
                    'ellipse': ellipse,
                    'img': img,
                    'pred': pred
                })

                print(f"✅ ÉXITO: HC predicho = {HC_pred:.2f}mm")

            elif HC_pred:
                print(f"⚠️ PARCIAL: HC predicho = {HC_pred:.2f}mm (sin ground truth)")
            else:
                print(f"❌ FALLO: No se pudo procesar la imagen")

        except Exception as e:
            print(f"❌ ERROR CRÍTICO: {str(e)}")
            import traceback
            traceback.print_exc()

    return resultados

# ==============================================================
# EJECUCIÓN
# ==============================================================

def main():
    # Bloque de salida decorativa (con fallback ascii si encoding falla)
    try:
        print("\n" + "🚀 "*20)
        print("INICIANDO PROCESAMIENTO CON PIPELINE OPTIMIZADO")
        print("🚀 "*20 + "\n")
    except UnicodeEncodeError:
        print("\n" + "ROCKET "*10)
        print("INICIANDO PROCESAMIENTO CON PIPELINE OPTIMIZADO")
        print("ROCKET "*10 + "\n")

    imagenes_test = ["045", "050", "300", "202"]
    return procesar_con_pipeline_optimizado(imagenes_test, visualizar_todos=True)

if __name__ == "__main__":
    resultados = main()

# ==============================================================
# RESUMEN
# ==============================================================

if __name__ == "__main__":
    if resultados:
        print("\n" + "="*110)
        print("RESULTADOS FINALES - PIPELINE OPTIMIZADO")
        print("="*110)
        print(f"{'Img':<8} {'HC Real':<10} {'HC Pred':<10} {'Error(mm)':<12} {'Error(%)':<10} {'Circ':<8} {'Método':<10}")
        print("-"*110)

        errores = []
        for res in resultados:
            emoji = 'OK' if res['stats']['error_pct'] < 5 else 'MED' if res['stats']['error_pct'] < 10 else 'WARN'
            print(f"{res['nombre']:<8} {res['HC_real']:<10.2f} {res['HC_pred']:<10.2f} "
                  f"{res['stats']['error_abs']:<12.2f} {res['stats']['error_pct']:<10.2f} "
                  f"{res['stats'].get('circularity', 0):<8.3f} "
                  f"{res['stats'].get('metodo_usado', 'N/A'):<10} {emoji}")
            errores.append(res['stats']['error_pct'])

        print("="*110)
        print(f"\nESTADÍSTICAS GLOBALES:")
        print(f"   • Procesadas exitosamente: {len(resultados)}/{len(imagenes_test)}")
        print(f"   • Error promedio: {np.mean(errores):.2f}%")
        print(f"   • Error mediano: {np.median(errores):.2f}%")
        print(f"   • Desviación estándar: {np.std(errores):.2f}%")
        print(f"   • Rango: [{np.min(errores):.2f}% - {np.max(errores):.2f}%]")

        exitos_5 = sum(1 for e in errores if e < 5)
        exitos_10 = sum(1 for e in errores if e < 10)

        print(f"\nPRECISIÓN:")
        print(f"   • Error < 5%: {exitos_5}/{len(errores)} ({exitos_5/len(errores)*100:.1f}%)")
        print(f"   • Error < 10%: {exitos_10}/{len(errores)} ({exitos_10/len(errores)*100:.1f}%)")

        if np.mean(errores) < 5:
            print(f"\nPRECISIÓN DE GRADO MÉDICO ALCANZADA")
        elif np.mean(errores) < 10:
            print(f"\nOBJETIVO CUMPLIDO (Error promedio < 10%)")
        else:
            print(f"\nProgreso: {100 - np.mean(errores):.1f}% de precisión")

        print("="*110)
    else:
        print("\nNo se obtuvieron resultados. Revisar diagnóstico...")