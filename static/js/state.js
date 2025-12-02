/**
 * Estado global de la aplicación
 */
const AppState = {
    // Herramienta actual
    currentTool: 'ellipse',
    
    // Datos de la imagen actual
    currentImage: null,
    imageName: '',
    pixelSize: 1,
    realHC: null,
    
    // Elipse del usuario
    ellipse: null,
    isDrawingEllipse: false,
    ellipseStart: null,
    
    // Handles de la elipse
    selectedHandle: null,
    isDraggingHandle: false,
    
    // Línea BPD
    line: null,
    linePoints: [],
    isDrawingLine: false,
    
    // Predicción IA
    iaEllipse: null,
    
    // Paso actual (1, 2, 3)
    currentStep: 1
};

/**
 * Configuración de colores y estilos
 */
const Config = {
    colors: {
        userEllipse: '#3b82f6',
        iaEllipse: '#8b5cf6',
        bpdLine: '#10b981',
        handle: '#ffffff',
        handleCenter: '#f59e0b'
    },
    handleSize: 8,
    lineWidth: 2
};

export { AppState, Config };
