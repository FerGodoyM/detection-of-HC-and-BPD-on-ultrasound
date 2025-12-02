/**
 * Aplicación Principal - Módulo de inicialización y eventos
 */

import { AppState } from './state.js';
import { API } from './api.js';
import { initCanvas, loadImage, redraw, getMousePos, clearDrawings, getHandleAtPosition } from './canvas.js';
import { startEllipse, moveEllipse, endEllipse, addLinePoint, cancelLine } from './tools.js';
import { 
    initUI, 
    updateStep, 
    setActiveTool, 
    updateHCMeasurements, 
    updateBPDMeasurements, 
    showError,
    updateImageInfo,
    resetMeasurements 
} from './ui.js';

/**
 * Carga una nueva imagen
 */
async function loadNewImage() {
    try {
        const data = await API.getRandomImage();
        
        AppState.imageName = data.nombre;
        AppState.pixelSize = data.pixel_size;
        AppState.realHC = data.hc_real;
        
        await loadImage(data.imagen);
        
        // Reset estado
        clearDrawings();
        resetMeasurements();
        updateStep(1);
        updateImageInfo(data.nombre, data.pixel_size);
        
        // Mostrar HC real
        updateHCMeasurements(null, null, data.hc_real);
        
    } catch (error) {
        console.error('Error cargando imagen:', error);
        alert('Error al cargar la imagen. Intenta de nuevo.');
    }
}

/**
 * Procesa la elipse del usuario
 */
async function processUserEllipse() {
    if (!AppState.ellipse) return;
    
    try {
        // Calcular HC del usuario
        const result = await API.calculateMeasures(AppState.ellipse, AppState.pixelSize);
        
        // Obtener predicción IA
        const iaResult = await API.getIAPrediction(AppState.imageName, AppState.pixelSize);
        
        // Guardar elipse de IA
        if (iaResult.axes) {
            AppState.iaEllipse = {
                centerX: iaResult.center.x,
                centerY: iaResult.center.y,
                radiusX: iaResult.axes.a,
                radiusY: iaResult.axes.b,
                rotation: (iaResult.angle || 0) * Math.PI / 180
            };
            redraw();
        }
        
        // Actualizar mediciones
        updateHCMeasurements(result.hc_mm, iaResult.hc_mm, AppState.realHC);
        
        // Calcular y mostrar error
        if (AppState.realHC) {
            const error = Math.abs(result.hc_mm - AppState.realHC) / AppState.realHC * 100;
            showError(error);
        }
        
        // Avanzar al paso 2
        updateStep(2);
        
    } catch (error) {
        console.error('Error procesando elipse:', error);
    }
}

/**
 * Procesa la línea BPD del usuario
 */
async function processUserLine() {
    if (!AppState.line) return;
    
    try {
        const result = await API.calculateBPD(AppState.line, AppState.pixelSize);
        
        updateBPDMeasurements(result.bpd_mm, null, '--');
        
        // Avanzar al paso 3
        updateStep(3);
        
    } catch (error) {
        console.error('Error procesando línea:', error);
    }
}

/**
 * Inicializa los event listeners del canvas
 */
function initCanvasEvents(canvas) {
    // Mouse down
    canvas.addEventListener('mousedown', (e) => {
        const pos = getMousePos(e);
        
        if (AppState.currentTool === 'ellipse') {
            startEllipse(pos.x, pos.y);
        } else if (AppState.currentTool === 'line') {
            const line = addLinePoint(pos.x, pos.y);
            // Ya no procesa automáticamente, espera al botón Siguiente
        }
    });
    
    // Mouse move
    canvas.addEventListener('mousemove', (e) => {
        const pos = getMousePos(e);
        
        // Cambiar cursor si está sobre un handle
        if (AppState.currentTool === 'ellipse') {
            const handle = getHandleAtPosition(pos.x, pos.y);
            canvas.style.cursor = handle ? 'pointer' : 'crosshair';
        }
        
        if (AppState.currentTool === 'ellipse') {
            moveEllipse(pos.x, pos.y);
        }
    });
    
    // Mouse up
    canvas.addEventListener('mouseup', () => {
        if (AppState.currentTool === 'ellipse') {
            endEllipse();
            // Ya no procesa automáticamente, espera al botón Siguiente
        }
    });
    
    // Mouse leave
    canvas.addEventListener('mouseleave', () => {
        if (AppState.isDrawingEllipse) {
            endEllipse();
        }
    });
    
    // Tecla Escape para cancelar
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            cancelLine();
        }
    });
}

/**
 * Inicializa los event listeners de los botones
 */
function initButtonEvents() {
    // Botón nueva imagen
    document.getElementById('btnNewImage').addEventListener('click', loadNewImage);
    
    // Botón siguiente
    document.getElementById('btnNext').addEventListener('click', async () => {
        console.log('Siguiente clicked. Step:', AppState.currentStep, 'Ellipse:', AppState.ellipse, 'Line:', AppState.line);
        
        if (AppState.currentStep === 1 && AppState.ellipse) {
            await processUserEllipse();
        } else if (AppState.currentStep === 2 && AppState.line) {
            await processUserLine();
        } else if (AppState.currentStep === 1 && !AppState.ellipse) {
            alert('Primero dibuja una elipse alrededor de la cabeza fetal');
        } else if (AppState.currentStep === 2 && !AppState.line) {
            alert('Dibuja una línea para medir el BPD');
        }
    });
    
    // Botones de herramientas
    document.getElementById('toolEllipse').addEventListener('click', () => {
        setActiveTool('ellipse');
    });
    
    document.getElementById('toolLine').addEventListener('click', () => {
        setActiveTool('line');
    });
    
    document.getElementById('toolClear').addEventListener('click', () => {
        clearDrawings();
        resetMeasurements();
        updateStep(1);
    });
    
    document.getElementById('toolUndo').addEventListener('click', () => {
        if (AppState.line) {
            AppState.line = null;
            updateStep(2);
        } else if (AppState.ellipse) {
            AppState.ellipse = null;
            AppState.iaEllipse = null;
            updateStep(1);
        }
        redraw();
        resetMeasurements();
        updateHCMeasurements(null, null, AppState.realHC);
    });
}

/**
 * Inicialización de la aplicación
 */
function init() {
    // Inicializar módulos
    const canvas = document.getElementById('mainCanvas');
    initCanvas(canvas);
    initUI();
    
    // Configurar eventos
    initCanvasEvents(canvas);
    initButtonEvents();
    
    // Cargar primera imagen
    loadNewImage();
}

// Iniciar cuando el DOM esté listo
document.addEventListener('DOMContentLoaded', init);
