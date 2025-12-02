/**
 * Módulo de Canvas - Dibujo y renderizado
 */

import { AppState, Config } from './state.js';

let canvas, ctx;
let canvasScale = 1;

/**
 * Inicializa el canvas
 */
function initCanvas(canvasElement) {
    canvas = canvasElement;
    ctx = canvas.getContext('2d');
}

/**
 * Obtiene el canvas y contexto
 */
function getCanvas() {
    return { canvas, ctx };
}

/**
 * Obtiene la escala actual del canvas
 */
function getCanvasScale() {
    return canvasScale;
}

/**
 * Carga una imagen en el canvas
 */
function loadImage(imageData) {
    return new Promise((resolve) => {
        const img = new Image();
        img.onload = () => {
            // Calcular escala para ajustar al contenedor
            const container = canvas.parentElement;
            const containerRect = container.getBoundingClientRect();
            
            const scaleX = containerRect.width / img.width;
            const scaleY = containerRect.height / img.height;
            canvasScale = Math.min(scaleX, scaleY, 1);
            
            canvas.width = img.width * canvasScale;
            canvas.height = img.height * canvasScale;
            
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            
            AppState.currentImage = img;
            resolve();
        };
        img.src = 'data:image/png;base64,' + imageData;
    });
}

/**
 * Redibuja todo el canvas
 */
function redraw() {
    if (!AppState.currentImage) return;
    
    // Limpiar y dibujar imagen
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.drawImage(AppState.currentImage, 0, 0, canvas.width, canvas.height);
    
    // Dibujar elipse de IA (si existe)
    if (AppState.iaEllipse) {
        drawEllipse(AppState.iaEllipse, Config.colors.iaEllipse, true);
    }
    
    // Dibujar elipse del usuario (si existe)
    if (AppState.ellipse) {
        drawEllipse(AppState.ellipse, Config.colors.userEllipse);
        drawHandles(AppState.ellipse);
    }
    
    // Dibujar línea BPD (si existe)
    if (AppState.line) {
        drawLine(AppState.line, Config.colors.bpdLine);
    }
}

/**
 * Dibuja una elipse
 */
function drawEllipse(ellipse, color, dashed = false) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = Config.lineWidth;
    
    if (dashed) {
        ctx.setLineDash([5, 5]);
    }
    
    ctx.beginPath();
    ctx.ellipse(
        ellipse.centerX * canvasScale,
        ellipse.centerY * canvasScale,
        ellipse.radiusX * canvasScale,
        ellipse.radiusY * canvasScale,
        ellipse.rotation || 0,
        0,
        2 * Math.PI
    );
    ctx.stroke();
    ctx.restore();
}

/**
 * Dibuja los handles de la elipse
 */
function drawHandles(ellipse) {
    const handles = getEllipseHandles(ellipse);
    
    handles.forEach(handle => {
        ctx.beginPath();
        
        if (handle.type === 'center') {
            // Handle del centro (círculo)
            ctx.fillStyle = Config.colors.handleCenter;
            ctx.arc(handle.x * canvasScale, handle.y * canvasScale, Config.handleSize, 0, 2 * Math.PI);
            ctx.fill();
        } else if (handle.type === 'rotate') {
            // Handle de rotación (círculo con símbolo)
            ctx.fillStyle = '#ef4444';
            ctx.arc(handle.x * canvasScale, handle.y * canvasScale, Config.handleSize, 0, 2 * Math.PI);
            ctx.fill();
            ctx.fillStyle = 'white';
            ctx.font = '10px Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText('↻', handle.x * canvasScale, handle.y * canvasScale);
        } else {
            // Handles de esquina/borde (cuadrados)
            ctx.fillStyle = Config.colors.handle;
            ctx.fillRect(
                handle.x * canvasScale - Config.handleSize / 2,
                handle.y * canvasScale - Config.handleSize / 2,
                Config.handleSize,
                Config.handleSize
            );
        }
    });
}

/**
 * Obtiene las posiciones de los handles de una elipse
 */
function getEllipseHandles(ellipse) {
    const cos = Math.cos(ellipse.rotation || 0);
    const sin = Math.sin(ellipse.rotation || 0);
    
    return [
        { type: 'center', x: ellipse.centerX, y: ellipse.centerY },
        { type: 'right', x: ellipse.centerX + ellipse.radiusX * cos, y: ellipse.centerY + ellipse.radiusX * sin },
        { type: 'left', x: ellipse.centerX - ellipse.radiusX * cos, y: ellipse.centerY - ellipse.radiusX * sin },
        { type: 'top', x: ellipse.centerX - ellipse.radiusY * sin, y: ellipse.centerY + ellipse.radiusY * cos },
        { type: 'bottom', x: ellipse.centerX + ellipse.radiusY * sin, y: ellipse.centerY - ellipse.radiusY * cos },
        { 
            type: 'rotate', 
            x: ellipse.centerX + (ellipse.radiusX + 25) * cos, 
            y: ellipse.centerY + (ellipse.radiusX + 25) * sin 
        }
    ];
}

/**
 * Detecta si el mouse está sobre un handle
 */
function getHandleAtPosition(x, y) {
    if (!AppState.ellipse) return null;
    
    const handles = getEllipseHandles(AppState.ellipse);
    const threshold = Config.handleSize * 1.5 / canvasScale;
    
    for (const handle of handles) {
        const dist = Math.sqrt((x - handle.x) ** 2 + (y - handle.y) ** 2);
        if (dist < threshold) {
            return handle;
        }
    }
    return null;
}

/**
 * Dibuja una línea
 */
function drawLine(line, color) {
    ctx.save();
    ctx.strokeStyle = color;
    ctx.lineWidth = Config.lineWidth;
    ctx.setLineDash([]);
    
    ctx.beginPath();
    ctx.moveTo(line.x1 * canvasScale, line.y1 * canvasScale);
    ctx.lineTo(line.x2 * canvasScale, line.y2 * canvasScale);
    ctx.stroke();
    
    // Dibujar puntos en los extremos
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.arc(line.x1 * canvasScale, line.y1 * canvasScale, 5, 0, 2 * Math.PI);
    ctx.fill();
    ctx.beginPath();
    ctx.arc(line.x2 * canvasScale, line.y2 * canvasScale, 5, 0, 2 * Math.PI);
    ctx.fill();
    
    ctx.restore();
}

/**
 * Obtiene las coordenadas del mouse relativas al canvas (sin escala)
 */
function getMousePos(e) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: (e.clientX - rect.left) / canvasScale,
        y: (e.clientY - rect.top) / canvasScale
    };
}

/**
 * Limpia los dibujos del usuario
 */
function clearDrawings() {
    AppState.ellipse = null;
    AppState.line = null;
    AppState.linePoints = [];
    AppState.iaEllipse = null;
    redraw();
}

export { 
    initCanvas, 
    getCanvas,
    getCanvasScale,
    loadImage, 
    redraw, 
    drawEllipse,
    drawLine,
    getEllipseHandles,
    getHandleAtPosition,
    getMousePos,
    clearDrawings 
};
