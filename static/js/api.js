/**
 * Módulo de API - Comunicación con el backend
 */

const API = {
    /**
     * Obtiene una imagen aleatoria del dataset
     */
    async getRandomImage() {
        const response = await fetch('/api/imagen-aleatoria');
        if (!response.ok) throw new Error('Error al cargar imagen');
        const data = await response.json();
        
        // Cargar la imagen como base64
        const imgResponse = await fetch(`/api/imagen/${data.filename}`);
        if (!imgResponse.ok) throw new Error('Error al cargar imagen');
        const blob = await imgResponse.blob();
        
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => {
                resolve({
                    nombre: data.filename,
                    pixel_size: data.pixel_size,
                    hc_real: data.hc_real,
                    imagen: reader.result.split(',')[1] // Solo el base64
                });
            };
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    },

    /**
     * Calcula las medidas HC basadas en la elipse del usuario
     */
    async calculateMeasures(ellipseData, pixelSize) {
        const response = await fetch('/api/calcular-medidas', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                centerX: ellipseData.centerX,
                centerY: ellipseData.centerY,
                axisA: ellipseData.radiusX,
                axisB: ellipseData.radiusY,
                pixelSize: pixelSize
            })
        });
        if (!response.ok) throw new Error('Error al calcular medidas');
        return response.json();
    },

    /**
     * Calcula el BPD basado en la línea del usuario
     */
    async calculateBPD(lineData, pixelSize) {
        const response = await fetch('/api/calcular-bpd', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                x1: lineData.x1,
                y1: lineData.y1,
                x2: lineData.x2,
                y2: lineData.y2,
                pixelSize: pixelSize
            })
        });
        if (!response.ok) throw new Error('Error al calcular BPD');
        return response.json();
    },

    /**
     * Obtiene la predicción de la IA para la imagen actual
     */
    async getIAPrediction(imageName, pixelSize) {
        const response = await fetch('/api/prediccion-ia', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ 
                filename: imageName,
                pixelSize: pixelSize
            })
        });
        if (!response.ok) throw new Error('Error en predicción IA');
        return response.json();
    }
};

export { API };
