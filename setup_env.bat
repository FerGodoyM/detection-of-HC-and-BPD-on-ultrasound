@echo off
echo ==================================================
echo Configurando entorno para PROYECTO OBSTETRICIA
echo ==================================================

:: Verificar si el lanzador py esta disponible y buscar Python 3.11
py -3.11 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python 3.11 no encontrado. Se requiere Python 3.10 o 3.11 para TensorFlow.
    echo Por favor instala Python 3.11.
    pause
    exit /b 1
)

:: Crear entorno virtual usando Python 3.11
if not exist "env" (
    echo Creando entorno virtual con Python 3.11...
    py -3.11 -m venv env
) else (
    echo El entorno virtual ya existe.
)

:: Activar entorno e instalar dependencias
echo Activando entorno e instalando librerias...
call env\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ==================================================
echo Instalacion completada exitosamente!
echo ==================================================
echo Para activar el entorno manualmente usa: env\Scripts\activate
echo.
pause
