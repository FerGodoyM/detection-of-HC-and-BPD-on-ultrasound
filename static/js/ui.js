/**
 * Módulo de UI - Actualización de interfaz
 */

import { AppState } from './state.js';

// Referencias a elementos del DOM
let elements = {};

/**
 * Inicializa las referencias a elementos del DOM
 */
function initUI() {
    elements = {
        // Steps
        step1: document.getElementById('step1'),
        step2: document.getElementById('step2'),
        step3: document.getElementById('step3'),
        
        // Instructions
        instructions: document.getElementById('instructions'),
        
        // Measurements
        userHC: document.getElementById('userHC'),
        iaHC: document.getElementById('iaHC'),
        realHC: document.getElementById('realHC'),
        userBPD: document.getElementById('userBPD'),
        iaBPD: document.getElementById('iaBPD'),
        realBPD: document.getElementById('realBPD'),
        
        // Error display
        errorDisplay: document.getElementById('errorDisplay'),
        errorValue: document.getElementById('errorValue'),
        
        // Toolbar
        toolEllipse: document.getElementById('toolEllipse'),
        toolLine: document.getElementById('toolLine'),
        
        // Image info
        imageInfo: document.getElementById('imageInfo')
    };
}

/**
 * Actualiza el paso actual
 */
function updateStep(step) {
    AppState.currentStep = step;
    
    // Actualizar clases de pasos
    [elements.step1, elements.step2, elements.step3].forEach((el, i) => {
        el.classList.remove('active', 'completed');
        if (i + 1 < step) {
            el.classList.add('completed');
        } else if (i + 1 === step) {
            el.classList.add('active');
        }
    });
    
    // Actualizar instrucciones
    const instructions = {
        1: `<strong>Paso 1:</strong> Dibuja una elipse alrededor de la cabeza fetal.<br>
            • Arrastra desde una esquina a la opuesta<br>
            • Usa los handles □ para ajustar tamaño<br>
            • Usa ● para mover, 🔄 para rotar`,
        2: `<strong>Paso 2:</strong> Dibuja la línea del BPD (Diámetro Biparietal).<br>
            • Haz clic en un punto del borde<br>
            • Haz clic en el punto opuesto`,
        3: `<strong>¡Completado!</strong> Revisa tus mediciones comparadas con la IA y el valor real.`
    };
    
    elements.instructions.innerHTML = instructions[step] || '';
    
    // Actualizar herramienta activa
    if (step === 1) {
        setActiveTool('ellipse');
    } else if (step === 2) {
        setActiveTool('line');
    }
}

/**
 * Establece la herramienta activa
 */
function setActiveTool(tool) {
    AppState.currentTool = tool;
    
    elements.toolEllipse.classList.toggle('active', tool === 'ellipse');
    elements.toolLine.classList.toggle('active', tool === 'line');
}

/**
 * Actualiza las mediciones HC
 */
function updateHCMeasurements(userHC, iaHC, realHC) {
    elements.userHC.textContent = userHC ? userHC.toFixed(1) : '--';
    elements.iaHC.textContent = iaHC ? iaHC.toFixed(1) : '--';
    elements.realHC.textContent = realHC ? realHC.toFixed(1) : '--';
}

/**
 * Actualiza las mediciones BPD
 */
function updateBPDMeasurements(userBPD, iaBPD, realBPD) {
    elements.userBPD.textContent = userBPD ? userBPD.toFixed(1) : '--';
    elements.iaBPD.textContent = iaBPD ? iaBPD.toFixed(1) : '--';
    elements.realBPD.textContent = realBPD || '--';
}

/**
 * Muestra el error del usuario
 */
function showError(errorPercent) {
    elements.errorDisplay.classList.add('show');
    elements.errorValue.textContent = errorPercent.toFixed(1) + '%';
    
    // Determinar clase según el error
    elements.errorDisplay.classList.remove('good', 'warning', 'bad');
    if (errorPercent < 5) {
        elements.errorDisplay.classList.add('good');
    } else if (errorPercent < 10) {
        elements.errorDisplay.classList.add('warning');
    } else {
        elements.errorDisplay.classList.add('bad');
    }
}

/**
 * Oculta el error
 */
function hideError() {
    elements.errorDisplay.classList.remove('show');
}

/**
 * Actualiza la información de la imagen
 */
function updateImageInfo(name, pixelSize) {
    elements.imageInfo.textContent = `${name} | ${pixelSize.toFixed(4)} mm/px`;
}

/**
 * Resetea todas las mediciones
 */
function resetMeasurements() {
    elements.userHC.textContent = '--';
    elements.iaHC.textContent = '--';
    elements.realHC.textContent = '--';
    elements.userBPD.textContent = '--';
    elements.iaBPD.textContent = '--';
    elements.realBPD.textContent = '--';
    hideError();
}

export { 
    initUI, 
    updateStep, 
    setActiveTool,
    updateHCMeasurements, 
    updateBPDMeasurements, 
    showError,
    hideError,
    updateImageInfo,
    resetMeasurements 
};
