# Configuración del Entorno Local

He configurado un entorno virtual de Python para ejecutar tu proyecto de Obstetricia.

## Archivos creados
- `requirements.txt`: Lista de todas las librerías necesarias (TensorFlow, NumPy, OpenCV, etc.).
- `setup_env.bat`: Script para automatizar la creación del entorno e instalación de librerías.

## Instrucciones

### 1. Entorno Virtual
He creado un entorno virtual llamado `env` (en lugar de `venv` para evitar conflictos) utilizando **Python 3.11**.

Si necesitas recrear el entorno en el futuro, simplemente ejecuta el archivo `setup_env.bat` haciendo doble clic sobre él.

### 2. Ejecutar el código
Para ejecutar tus scripts, primero debes activar el entorno virtual.

**Desde la terminal (PowerShell/CMD):**
```bash
env\Scripts\activate
python calculo_HC.py
```

**En VS Code:**
Asegúrate de seleccionar el intérprete de Python que está dentro de la carpeta `env`.
1. Presiona `Ctrl + Shift + P`
2. Escribe "Python: Select Interpreter"
3. Selecciona la opción que dice `Python 3.11.x ('env': venv)`

## Notas sobre Librerías
- Se ha instalado `numpy==1.26.4` específicamente para mantener compatibilidad con TensorFlow.
- Se han incluido todas las librerías detectadas en tus scripts (`calculo_HC.py` y `entrenamiento.py`).
