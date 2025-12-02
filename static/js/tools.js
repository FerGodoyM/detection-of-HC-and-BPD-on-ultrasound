/**
 * Módulo de Herramientas de Dibujo
 */

import { AppState, Config } from './state.js';
import { getCanvas, getCanvasScale, redraw, getHandleAtPosition, getMousePos } from './canvas.js';

/**
 * Maneja el inicio del dibujo de elipse
 */
function startEllipse(x, y) {
    // Verificar si se está sobre un handle
    const handle = getHandleAtPosition(x, y);
    if (handle && AppState.ellipse) {
        AppState.selectedHandle = handle;
        AppState.isDraggingHandle = true;
        return;
    }
    
    // Iniciar nueva elipse
    AppState.isDrawingEllipse = true;
    AppState.ellipseStart = { x, y };
    AppState.ellipse = null;
}

/**
 * Maneja el movimiento durante el dibujo de elipse
 */
function moveEllipse(x, y) {
    if (AppState.isDraggingHandle && AppState.selectedHandle) {
        updateEllipseHandle(x, y);
        return;
    }
    
    if (!AppState.isDrawingEllipse || !AppState.ellipseStart) return;
    
    // Calcular elipse desde esquina a esquina
    const centerX = (AppState.ellipseStart.x + x) / 2;
    const centerY = (AppState.ellipseStart.y + y) / 2;
    const radiusX = Math.abs(x - AppState.ellipseStart.x) / 2;
    const radiusY = Math.abs(y - AppState.ellipseStart.y) / 2;
    
    if (radiusX > 5 && radiusY > 5) {
        AppState.ellipse = {
            centerX,
            centerY,
            radiusX,
            radiusY,
            rotation: 0
        };
        redraw();
    }
}

/**
 * Maneja el fin del dibujo de elipse
 */
function endEllipse() {
    AppState.isDrawingEllipse = false;
    AppState.ellipseStart = null;
    AppState.isDraggingHandle = false;
    AppState.selectedHandle = null;
    
    return AppState.ellipse;
}

/**
 * Actualiza la elipse según el handle que se está arrastrando
 */
function updateEllipseHandle(x, y) {
    if (!AppState.ellipse || !AppState.selectedHandle) return;
    
    const handle = AppState.selectedHandle;
    const ellipse = AppState.ellipse;
    
    switch (handle.type) {
        case 'center':
            ellipse.centerX = x;
            ellipse.centerY = y;
            break;
            
        case 'right':
        case 'left':
            const distX = Math.sqrt(
                (x - ellipse.centerX) ** 2 + 
                (y - ellipse.centerY) ** 2
            );
            ellipse.radiusX = Math.max(10, distX);
            break;
            
        case 'top':
        case 'bottom':
            const distY = Math.sqrt(
                (x - ellipse.centerX) ** 2 + 
                (y - ellipse.centerY) ** 2
            );
            ellipse.radiusY = Math.max(10, distY);
            break;
            
        case 'rotate':
            ellipse.rotation = Math.atan2(
                y - ellipse.centerY,
                x - ellipse.centerX
            );
            break;
    }
    
    redraw();
}

/**
 * Agrega un punto para la línea BPD
 */
function addLinePoint(x, y) {
    AppState.linePoints.push({ x, y });
    
    if (AppState.linePoints.length === 2) {
        AppState.line = {
            x1: AppState.linePoints[0].x,
            y1: AppState.linePoints[0].y,
            x2: AppState.linePoints[1].x,
            y2: AppState.linePoints[1].y
        };
        AppState.linePoints = [];
        redraw();
        return AppState.line;
    }
    
    // Dibujar punto temporal
    const { ctx } = getCanvas();
    const scale = getCanvasScale();
    ctx.fillStyle = Config.colors.bpdLine;
    ctx.beginPath();
    ctx.arc(x * scale, y * scale, 5, 0, 2 * Math.PI);
    ctx.fill();
    
    return null;
}

/**
 * Cancela el dibujo de línea
 */
function cancelLine() {
    AppState.linePoints = [];
    redraw();
}

export { 
    startEllipse, 
    moveEllipse, 
    endEllipse, 
    addLinePoint,
    cancelLine 
};
