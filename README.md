# Detección de HC y BPD en Ecografías con IA

Proyecto para segmentar la cabeza fetal en imágenes de ultrasonido y medir perímetro cefálico (HC) y diámetros BPD/OFD mediante un modelo U-Net y un pipeline de post-procesamiento robusto.

## Características
- U-Net ligera para segmentación de cabeza fetal.
- Post-procesado adaptativo con morfología y ajuste de elipse.
- Cálculo robusto de HC usando Ramanujan, perímetro de contorno y aproximación mejorada.
- Uso de metadatos de pixel size del dataset HC18 para convertir píxeles a milímetros.

## Requisitos
- Python 3.11 (probado)
- Paquetes principales: TensorFlow, NumPy, OpenCV, Pandas, SciPy, scikit-image, Matplotlib
- Ver `requirements.txt` para instalar dependencias.

Instalación (PowerShell):
```powershell
# Crear y activar entorno virtual
python -m venv env
./env/Scripts/Activate.ps1

# Instalar dependencias
python -m pip install -r requirements.txt
```

## Dataset
  - `training_set/` y `test_set/` con imágenes `*_HC.png`
  - `training_set_pixel_size_and_HC.csv` (pixel size y HC real)
  - `test_set_pixel_size.csv` (pixel size)

Nota: El repositorio ignora por defecto los archivos de dataset y binarios mediante `.gitignore`.

## Entrenamiento
El script `entrenamiento.py` entrena la U-Net ligera con el conjunto de entrenamiento.

Ejecutar:
```powershell
python entrenamiento.py
```
Salida: guarda el mejor modelo en `unet_hc18_best.h5`.

### Pesos entrenados
- Archivo: `unet_hc18_best.h5`
- Ubicación: raíz del proyecto (mismo nivel que los scripts)
- Uso: `calculo_HC.py` intentará cargarlo automáticamente.

Nota: Por políticas de tamaño y buenas prácticas, no se versiona el archivo `.h5` en el repositorio público. Puedes generarlo ejecutando `entrenamiento.py` o colocar tu propio `.h5` con el mismo nombre en la raíz.

## Inferencia y Medición
El script `calculo_HC.py` carga el modelo y aplica el pipeline optimizado para medir HC/BPD.

  - `load_sample_with_scaling(id)`: carga imagen `id` y corrige pixel size tras resize.
  - `get_HC_real(id)`: obtiene HC real del CSV (si disponible).
  - `medir_HC_pipeline_optimizado(pred, pixel_size, HC_esperado, visualizar, debug)`: pipeline de post-proceso y medición.

Ejecutar sobre ejemplos:
```powershell
python calculo_HC.py
```
El script procesa por defecto `045`, `050`, `300`, `202` y muestra visualizaciones (si están habilitadas).

### Créditos del dataset
Este proyecto utiliza el dataset HC18: Head Circumference Challenge. Más información y créditos en:
https://hc18.grand-challenge.org
Por favor, respeta los términos de uso y cita adecuadamente a los autores del desafío HC18.

## Estructura del Proyecto
- `calculo_HC.py`: Pipeline de medición (post-proceso, contorno, elipse, cálculo HC/BPD) y carga de metadatos.
- `entrenamiento.py`: Entrenamiento de U-Net ligera.
- `requirements.txt`: Dependencias.
- `unet_hc18_best.h5`: Pesos del modelo (generados tras entrenamiento; ignorados en git).
- `DATASET HC18/`: Dataset HC18 local (ignorado en git).
- `.gitignore`: Exclusiones para dataset, entorno virtual y binarios.

## Notas de Compatibilidad
- En consolas con codificación CP1252 (Windows), el script evita errores por emojis en `print` usando alternativas ASCII al ejecutar como script.
- En CPU, TensorFlow puede mostrar avisos de oneDNN y optimizaciones; son esperados.

## Resultado Esperado
Para imágenes con buena predicción, el pipeline selecciona el mejor contorno/elipse y calcula HC cercano al real (error medio objetivo < 10%). Se reportan métricas de calidad (circularidad, aspect ratio, área relativa, score) y comparativas.

## Contribuciones
Pull requests bienvenidos. Por favor, no subas datasets ni pesos del modelo; comparte solo código y configuraciones.
