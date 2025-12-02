/**
 * Aplicación de Medición Fetal - HC & BPD
 * Código organizado en secciones
 */

(function() {
    'use strict';

    // ============================================================
    // ESTADO GLOBAL
    // ============================================================
    const AppState = {
        currentTool: 'ellipse',
        currentImage: null,
        imageName: '',
        pixelSize: 1,
        realHC: null,
        ellipse: null,
        isDrawingEllipse: false,
        ellipseStart: null,
        selectedHandle: null,
        isDraggingHandle: false,
        line: null,
        linePoints: [],
        iaEllipse: null,
        currentStep: 1
    };

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

    // ============================================================
    // CANVAS Y DIBUJO
    // ============================================================
    let canvas, ctx;
    let canvasScale = 1;

    function initCanvas() {
        canvas = document.getElementById('mainCanvas');
        ctx = canvas.getContext('2d');
    }

    function getMousePos(e) {
        const rect = canvas.getBoundingClientRect();
        return {
            x: (e.clientX - rect.left) / canvasScale,
            y: (e.clientY - rect.top) / canvasScale
        };
    }

    function loadImage(imageData) {
        return new Promise((resolve) => {
            const img = new Image();
            img.onload = () => {
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

    function redraw() {
        if (!AppState.currentImage) return;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.drawImage(AppState.currentImage, 0, 0, canvas.width, canvas.height);
        
        if (AppState.iaEllipse) {
            drawEllipse(AppState.iaEllipse, Config.colors.iaEllipse, true);
        }
        
        if (AppState.ellipse) {
            drawEllipse(AppState.ellipse, Config.colors.userEllipse);
            drawHandles(AppState.ellipse);
        }
        
        if (AppState.line) {
            drawLine(AppState.line, Config.colors.bpdLine);
        }
    }

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

    function drawHandles(ellipse) {
        const handles = getEllipseHandles(ellipse);
        
        handles.forEach(handle => {
            ctx.beginPath();
            
            if (handle.type === 'center') {
                ctx.fillStyle = Config.colors.handleCenter;
                ctx.arc(handle.x * canvasScale, handle.y * canvasScale, Config.handleSize, 0, 2 * Math.PI);
                ctx.fill();
            } else if (handle.type === 'rotate') {
                ctx.fillStyle = '#ef4444';
                ctx.arc(handle.x * canvasScale, handle.y * canvasScale, Config.handleSize, 0, 2 * Math.PI);
                ctx.fill();
                ctx.fillStyle = 'white';
                ctx.font = '10px Arial';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText('↻', handle.x * canvasScale, handle.y * canvasScale);
            } else {
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

    function drawLine(line, color) {
        ctx.save();
        ctx.strokeStyle = color;
        ctx.lineWidth = Config.lineWidth;
        ctx.setLineDash([]);
        
        ctx.beginPath();
        ctx.moveTo(line.x1 * canvasScale, line.y1 * canvasScale);
        ctx.lineTo(line.x2 * canvasScale, line.y2 * canvasScale);
        ctx.stroke();
        
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(line.x1 * canvasScale, line.y1 * canvasScale, 5, 0, 2 * Math.PI);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(line.x2 * canvasScale, line.y2 * canvasScale, 5, 0, 2 * Math.PI);
        ctx.fill();
        
        ctx.restore();
    }

    function clearDrawings() {
        AppState.ellipse = null;
        AppState.line = null;
        AppState.linePoints = [];
        AppState.iaEllipse = null;
        redraw();
    }

    // ============================================================
    // HERRAMIENTAS DE DIBUJO
    // ============================================================
    function startEllipse(x, y) {
        const handle = getHandleAtPosition(x, y);
        if (handle && AppState.ellipse) {
            AppState.selectedHandle = handle;
            AppState.isDraggingHandle = true;
            return;
        }
        
        AppState.isDrawingEllipse = true;
        AppState.ellipseStart = { x, y };
        AppState.ellipse = null;
    }

    function moveEllipse(x, y) {
        if (AppState.isDraggingHandle && AppState.selectedHandle) {
            updateEllipseHandle(x, y);
            return;
        }
        
        if (!AppState.isDrawingEllipse || !AppState.ellipseStart) return;
        
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

    function endEllipse() {
        AppState.isDrawingEllipse = false;
        AppState.ellipseStart = null;
        AppState.isDraggingHandle = false;
        AppState.selectedHandle = null;
        return AppState.ellipse;
    }

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
                const distX = Math.sqrt((x - ellipse.centerX) ** 2 + (y - ellipse.centerY) ** 2);
                ellipse.radiusX = Math.max(10, distX);
                break;
            case 'top':
            case 'bottom':
                const distY = Math.sqrt((x - ellipse.centerX) ** 2 + (y - ellipse.centerY) ** 2);
                ellipse.radiusY = Math.max(10, distY);
                break;
            case 'rotate':
                ellipse.rotation = Math.atan2(y - ellipse.centerY, x - ellipse.centerX);
                break;
        }
        
        redraw();
    }

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
        ctx.fillStyle = Config.colors.bpdLine;
        ctx.beginPath();
        ctx.arc(x * canvasScale, y * canvasScale, 5, 0, 2 * Math.PI);
        ctx.fill();
        
        return null;
    }

    // ============================================================
    // API
    // ============================================================
    async function apiGetRandomImage() {
        const response = await fetch('/api/imagen-aleatoria');
        if (!response.ok) throw new Error('Error al cargar imagen');
        const data = await response.json();
        
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
                    imagen: reader.result.split(',')[1]
                });
            };
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    }

    async function apiCalculateMeasures(ellipseData, pixelSize) {
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
    }

    async function apiCalculateBPD(lineData, pixelSize) {
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
    }

    async function apiGetIAPrediction(imageName, pixelSize) {
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

    // ============================================================
    // UI
    // ============================================================
    const elements = {};

    function initUI() {
        elements.step1 = document.getElementById('step1');
        elements.step2 = document.getElementById('step2');
        elements.step3 = document.getElementById('step3');
        elements.instructions = document.getElementById('instructions');
        elements.userHC = document.getElementById('userHC');
        elements.iaHC = document.getElementById('iaHC');
        elements.realHC = document.getElementById('realHC');
        elements.userBPD = document.getElementById('userBPD');
        elements.iaBPD = document.getElementById('iaBPD');
        elements.realBPD = document.getElementById('realBPD');
        elements.errorDisplay = document.getElementById('errorDisplay');
        elements.errorValue = document.getElementById('errorValue');
        elements.toolEllipse = document.getElementById('toolEllipse');
        elements.toolLine = document.getElementById('toolLine');
        elements.imageInfo = document.getElementById('imageInfo');
    }

    function updateStep(step) {
        AppState.currentStep = step;
        
        [elements.step1, elements.step2, elements.step3].forEach((el, i) => {
            el.classList.remove('active', 'completed');
            if (i + 1 < step) {
                el.classList.add('completed');
            } else if (i + 1 === step) {
                el.classList.add('active');
            }
        });
        
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
        
        if (step === 1) {
            setActiveTool('ellipse');
        } else if (step === 2) {
            setActiveTool('line');
        }
    }

    function setActiveTool(tool) {
        AppState.currentTool = tool;
        elements.toolEllipse.classList.toggle('active', tool === 'ellipse');
        elements.toolLine.classList.toggle('active', tool === 'line');
    }

    function updateHCMeasurements(userHC, iaHC, realHC) {
        elements.userHC.textContent = userHC ? userHC.toFixed(1) : '--';
        elements.iaHC.textContent = iaHC ? iaHC.toFixed(1) : '--';
        elements.realHC.textContent = realHC ? realHC.toFixed(1) : '--';
    }

    function updateBPDMeasurements(userBPD, iaBPD, realBPD) {
        elements.userBPD.textContent = userBPD ? userBPD.toFixed(1) : '--';
        elements.iaBPD.textContent = iaBPD ? iaBPD.toFixed(1) : '--';
        elements.realBPD.textContent = realBPD || '--';
    }

    function showError(errorPercent) {
        elements.errorDisplay.classList.add('show');
        elements.errorValue.textContent = errorPercent.toFixed(1) + '%';
        
        elements.errorDisplay.classList.remove('good', 'warning', 'bad');
        if (errorPercent < 5) {
            elements.errorDisplay.classList.add('good');
        } else if (errorPercent < 10) {
            elements.errorDisplay.classList.add('warning');
        } else {
            elements.errorDisplay.classList.add('bad');
        }
    }

    function hideError() {
        elements.errorDisplay.classList.remove('show');
    }

    function updateImageInfo(name, pixelSize) {
        elements.imageInfo.textContent = `${name} | ${pixelSize.toFixed(4)} mm/px`;
    }

    function resetMeasurements() {
        elements.userHC.textContent = '--';
        elements.iaHC.textContent = '--';
        elements.realHC.textContent = '--';
        elements.userBPD.textContent = '--';
        elements.iaBPD.textContent = '--';
        elements.realBPD.textContent = '--';
        hideError();
    }

    // ============================================================
    // LÓGICA PRINCIPAL
    // ============================================================
    async function loadNewImage() {
        try {
            const data = await apiGetRandomImage();
            
            AppState.imageName = data.nombre;
            AppState.pixelSize = data.pixel_size;
            AppState.realHC = data.hc_real;
            
            await loadImage(data.imagen);
            
            clearDrawings();
            resetMeasurements();
            updateStep(1);
            updateImageInfo(data.nombre, data.pixel_size);
            updateHCMeasurements(null, null, data.hc_real);
            
        } catch (error) {
            console.error('Error cargando imagen:', error);
            alert('Error al cargar la imagen. Intenta de nuevo.');
        }
    }

    async function processUserEllipse() {
        if (!AppState.ellipse) {
            alert('Primero dibuja una elipse alrededor de la cabeza fetal');
            return;
        }
        
        try {
            console.log('Procesando elipse...', AppState.ellipse);
            
            // Calcular HC del usuario
            const result = await apiCalculateMeasures(AppState.ellipse, AppState.pixelSize);
            console.log('Resultado HC:', result);
            
            // Intentar obtener predicción IA (pero no bloquear si falla)
            let iaResult = null;
            try {
                iaResult = await apiGetIAPrediction(AppState.imageName, AppState.pixelSize);
                console.log('Resultado IA:', iaResult);
                
                if (iaResult && iaResult.axes) {
                    AppState.iaEllipse = {
                        centerX: iaResult.center.x,
                        centerY: iaResult.center.y,
                        radiusX: iaResult.axes.a,
                        radiusY: iaResult.axes.b,
                        rotation: (iaResult.angle || 0) * Math.PI / 180
                    };
                    redraw();
                }
            } catch (iaError) {
                console.warn('No se pudo obtener predicción IA:', iaError);
            }
            
            // Actualizar mediciones
            updateHCMeasurements(
                result.hc_mm, 
                iaResult ? iaResult.hc_mm : null, 
                AppState.realHC
            );
            
            // Calcular y mostrar error del usuario vs real
            if (AppState.realHC) {
                const error = Math.abs(result.hc_mm - AppState.realHC) / AppState.realHC * 100;
                showError(error);
            }
            
            // Avanzar al paso 2
            console.log('Avanzando al paso 2');
            updateStep(2);
            
        } catch (error) {
            console.error('Error procesando elipse:', error);
            alert('Error al calcular medidas: ' + error.message);
        }
    }

    async function processUserLine() {
        if (!AppState.line) {
            alert('Dibuja una línea para medir el BPD');
            return;
        }
        
        try {
            const result = await apiCalculateBPD(AppState.line, AppState.pixelSize);
            updateBPDMeasurements(result.bpd_mm, null, '--');
            updateStep(3);
            
        } catch (error) {
            console.error('Error procesando línea:', error);
            alert('Error al procesar la línea: ' + error.message);
        }
    }

    // ============================================================
    // EVENTOS
    // ============================================================
    function initCanvasEvents() {
        canvas.addEventListener('mousedown', (e) => {
            const pos = getMousePos(e);
            
            if (AppState.currentTool === 'ellipse') {
                startEllipse(pos.x, pos.y);
            } else if (AppState.currentTool === 'line') {
                addLinePoint(pos.x, pos.y);
            }
        });
        
        canvas.addEventListener('mousemove', (e) => {
            const pos = getMousePos(e);
            
            if (AppState.currentTool === 'ellipse') {
                const handle = getHandleAtPosition(pos.x, pos.y);
                canvas.style.cursor = handle ? 'pointer' : 'crosshair';
                moveEllipse(pos.x, pos.y);
            }
        });
        
        canvas.addEventListener('mouseup', () => {
            if (AppState.currentTool === 'ellipse') {
                endEllipse();
            }
        });
        
        canvas.addEventListener('mouseleave', () => {
            if (AppState.isDrawingEllipse) {
                endEllipse();
            }
        });
        
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape') {
                AppState.linePoints = [];
                redraw();
            }
        });
    }

    function initButtonEvents() {
        document.getElementById('btnNewImage').addEventListener('click', loadNewImage);
        
        document.getElementById('btnNext').addEventListener('click', async () => {
            console.log('Botón Siguiente clickeado. Paso actual:', AppState.currentStep);
            
            if (AppState.currentStep === 1) {
                await processUserEllipse();
            } else if (AppState.currentStep === 2) {
                await processUserLine();
            }
        });
        
        document.getElementById('toolEllipse').addEventListener('click', () => setActiveTool('ellipse'));
        document.getElementById('toolLine').addEventListener('click', () => setActiveTool('line'));
        
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

    // ============================================================
    // INICIALIZACIÓN
    // ============================================================
    function init() {
        console.log('Inicializando aplicación...');
        initCanvas();
        initUI();
        initCanvasEvents();
        initButtonEvents();
        loadNewImage();
        console.log('Aplicación inicializada');
    }

    // Iniciar cuando el DOM esté listo
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

})();
