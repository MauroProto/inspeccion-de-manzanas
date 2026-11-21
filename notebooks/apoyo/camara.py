"""Captura la cámara del navegador y muestra la predicción de cada cuadro en Colab."""

import json
import time

import cv2
import numpy as np
from IPython.display import display
from inspeccion import sha256
from video import dibujar_cuadro


def probar_camara(inspeccion, fuente, salida, nombre_camara="", clasificar=True, cuadros_de_prueba=80):
    cuadros_de_calentamiento = 10
    resolucion_ideal = 640
    import base64
    from google.colab.output import eval_js
    from IPython.display import Javascript

    display(Javascript("""
    (() => {
    window.cerrarManzanas?.();
    const c = {};
    window.iniciarManzanas = async function(nombre, ancho) {
      c.stream = await navigator.mediaDevices.getUserMedia({
        video: {width: {ideal: ancho}}, audio: false
      });
      if (nombre) {
        const dispositivos = await navigator.mediaDevices.enumerateDevices();
        const camaras = dispositivos.filter(d => d.kind === 'videoinput');
        const elegida = camaras.find(d => d.label.toLowerCase().includes(nombre.toLowerCase()));
        if (!elegida) throw new Error('Cámaras disponibles: ' + camaras.map(d => d.label).join(', '));
        c.stream.getTracks().forEach(t => t.stop());
        c.stream = await navigator.mediaDevices.getUserMedia({
          video: {deviceId: {exact: elegida.deviceId}, width: {ideal: ancho}}, audio: false
        });
      }
      c.video = document.createElement('video');
      c.video.playsInline = true;
      c.video.muted = true;
      c.video.srcObject = c.stream;
      await c.video.play();
      c.canvas = document.createElement('canvas');
      c.canvas.width = c.video.videoWidth;
      c.canvas.height = c.video.videoHeight;
      c.salida = document.createElement('img');
      c.salida.width = c.canvas.width;
      c.salida.alt = 'Imagen procesada';
      c.boton = document.createElement('button');
      c.boton.textContent = 'Detener cámara';
      c.boton.onclick = window.cerrarManzanas;
      document.body.append(c.boton, c.salida);
      c.activa = true;
      return [c.canvas.width, c.canvas.height];
    };
    window.leerManzanas = async function() {
      if (!c.activa) return null;
      if (document.hidden) throw new Error('Mantené visible la pestaña durante la prueba.');
      await new Promise((resolver, rechazar) => {
        const limite = setTimeout(() => {
          if (c.activa) rechazar(new Error('La cámara dejó de entregar cuadros.'));
          else resolver();
        }, 3000);
        c.video.requestVideoFrameCallback(() => { clearTimeout(limite); resolver(); });
      });
      if (!c.activa) return null;
      c.toma = performance.now();
      c.canvas.getContext('2d').drawImage(c.video, 0, 0);
      return c.canvas.toDataURL('image/jpeg', 0.8);
    };
    window.mostrarManzanas = async function(jpeg) {
      if (!c.activa) return null;
      c.salida.src = jpeg;
      await c.salida.decode();
      await new Promise((resolver, rechazar) => {
        const limite = setTimeout(() => rechazar(new Error('La vista previa dejó de actualizarse.')), 3000);
        requestAnimationFrame(() => requestAnimationFrame(() => { clearTimeout(limite); resolver(); }));
      });
      return c.activa ? performance.now() - c.toma : null;
    };
    window.cerrarManzanas = function() {
      c.activa = false;
      if (c.stream) c.stream.getTracks().forEach(t => t.stop());
      if (c.video) {
        c.video.pause();
        c.video.srcObject = null;
      }
      c.boton?.remove();
    };
    window.addEventListener('pagehide', window.cerrarManzanas, {once: true});
    })();
    """))

    if cuadros_de_prueba < 1 or cuadros_de_calentamiento < 0:
        raise ValueError('Revisá la cantidad de cuadros.')
    inspeccion.reiniciar_seguimiento()

    latencias, tiempos_modelos, cantidades = [], [], []
    inicio_medicion = None
    duracion_medicion = 0.0
    error_camara = None
    try:
        resolucion = eval_js(f'iniciarManzanas({json.dumps(nombre_camara)}, {int(resolucion_ideal)})', timeout_sec=90)
        for i in range(cuadros_de_calentamiento + cuadros_de_prueba):
            if i == cuadros_de_calentamiento:
                inspeccion.reiniciar_seguimiento()
                inicio_medicion = time.perf_counter()
            foto = eval_js('leerManzanas()', timeout_sec=15)
            if foto is None:
                break
            jpeg = base64.b64decode(foto.split(',')[1])
            imagen = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            if imagen is None:
                raise ValueError('El navegador no devolvió un cuadro JPEG válido.')
            t0 = time.perf_counter()
            predicciones = inspeccion.procesar_lote([imagen], clasificar=clasificar)[0]
            dibujada = dibujar_cuadro(imagen, predicciones, inspeccion.umbral_danio, fuente, clasificar)
            tiempo_modelos = (time.perf_counter() - t0) * 1000
            ok, archivo = cv2.imencode('.jpg', dibujada, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ok:
                raise RuntimeError('No se pudo convertir la imagen procesada a JPEG.')
            url = 'data:image/jpeg;base64,' + base64.b64encode(archivo).decode()
            latencia = eval_js(f'mostrarManzanas({json.dumps(url)})', timeout_sec=15)
            if latencia is None:
                break
            if inicio_medicion is not None:
                latencias.append(latencia)
                tiempos_modelos.append(tiempo_modelos)
                cantidades.append(len(predicciones))
                duracion_medicion = time.perf_counter() - inicio_medicion
    except Exception as error:
        error_camara = str(error)
    finally:
        try:
            eval_js('cerrarManzanas()', timeout_sec=5)
        except Exception as error:
            print('Revisá el indicador de cámara del navegador.', error)

    if latencias:
        medidas_camara = {
            'cuadros': len(latencias), 'cuadros_solicitados': cuadros_de_prueba,
            'completa': len(latencias) == cuadros_de_prueba, 'resolucion': resolucion,
            'dispositivo': inspeccion.dispositivo, 'clasificar': clasificar,
            'sha256_pesos': {nombre: sha256(ruta) for nombre, ruta in inspeccion.archivos.items()}, 'error': error_camara,
            'cuadros_con_detecciones': sum(n > 0 for n in cantidades),
            'detecciones_por_cuadro': float(np.mean(cantidades)),
            'fps': len(latencias) / duracion_medicion,
            'cuadro_disponible_a_presentacion_estimada_p50_ms': float(np.median(latencias)),
            'cuadro_disponible_a_presentacion_estimada_p95_ms': float(np.percentile(latencias, 95)),
            'modelos_y_dibujo_p50_ms': float(np.median(tiempos_modelos)),
            'modelos_y_dibujo_p95_ms': float(np.percentile(tiempos_modelos, 95)),
        }
        print(json.dumps(medidas_camara, indent=2))
        (salida / 'camara_tiempos.json').write_text(json.dumps(medidas_camara, indent=2))
    if error_camara:
        raise RuntimeError(error_camara)
    if not latencias:
        print('La cámara se detuvo antes de la primera muestra.')
