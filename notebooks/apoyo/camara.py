"""Captura la cámara del navegador y muestra la predicción de cada cuadro en Colab."""

import base64
import json
import time
from pathlib import Path

import cv2
import numpy as np
from IPython.display import Javascript, display
from ultralytics import YOLO

from coral import etiqueta_manzana
from video import dibujar_cuadro

# Nuestro YOLO se entrenó solo con fotos de manzanas y nunca vio una cara.
# En la cámara, una caja se muestra solo si YOLO11n preentrenado en COCO,
# que también conoce personas, ve una manzana en el mismo lugar.
MANZANA_EN_COCO = 47


def superposicion(caja, otra):
    """Área común dividida por el área que cubren las dos cajas (IoU)."""
    x1, y1 = max(caja[0], otra[0]), max(caja[1], otra[1])
    x2, y2 = min(caja[2], otra[2]), min(caja[3], otra[3])
    comun = max(0, x2 - x1) * max(0, y2 - y1)
    total = ((caja[2] - caja[0]) * (caja[3] - caja[1])
             + (otra[2] - otra[0]) * (otra[3] - otra[1]) - comun)
    return comun / total if total > 0 else 0


def solo_manzanas(imagen, cajas, coco, dispositivo):
    """Deja las cajas que el YOLO de COCO también reconoce como manzana en este cuadro."""
    if not cajas:
        return []
    resultado = coco.predict(imagen, imgsz=640, conf=0.10, classes=[MANZANA_EN_COCO],
                             device=dispositivo, verbose=False)[0]
    manzanas_coco = resultado.boxes.xyxy.tolist()
    elegidas = []
    for caja in cajas:
        coordenadas = (caja['x1'], caja['y1'], caja['x2'], caja['y2'])
        if any(superposicion(coordenadas, otra) >= 0.3 for otra in manzanas_coco):
            elegidas.append(caja)
    return elegidas


def preparar_camara():
    interfaz = Path(__file__).with_suffix('.js').read_text(encoding='utf-8')
    display(Javascript(interfaz))


def decodificar_cuadro(foto):
    jpeg = base64.b64decode(foto.split(',', 1)[1])
    imagen = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if imagen is None:
        raise ValueError('No se pudo leer el cuadro enviado por la cámara.')
    return imagen


def ver_en_vivo(inspeccion, nombre_camara=''):
    """Detecta y clasifica hasta que se pulse Detener cámara."""
    from google.colab.output import eval_js

    inspeccion.reiniciar_seguimiento()
    coco = YOLO('yolo11n.pt')
    preparar_camara()
    try:
        iniciar = f'iniciarManzanas({json.dumps(nombre_camara)}, 640, true)'
        foto = eval_js(iniciar + '.then(leerManzanas)', timeout_sec=90)
        while foto:
            imagen = decodificar_cuadro(foto)
            cajas = inspeccion.procesar_lote([imagen], clasificar=True)[0]
            cajas = solo_manzanas(imagen, cajas, coco, inspeccion.dispositivo)
            marcas = []
            for caja in cajas:
                texto, color = etiqueta_manzana(caja, inspeccion.umbral_danio)
                coordenadas = [round(caja[k]) for k in ('x1', 'y1', 'x2', 'y2')]
                marcas.append({'caja': coordenadas, 'texto': texto, 'color': color})
            foto = eval_js(f'siguienteManzanas({json.dumps(marcas)})', timeout_sec=15)
    finally:
        try:
            eval_js('window.cerrarManzanas?.()', timeout_sec=5)
        except Exception:
            print('Si la cámara sigue encendida, cerrá la pestaña de Colab.')


def probar_camara(inspeccion, fuente, salida, nombre_camara="", clasificar=True, cuadros_de_prueba=80):
    cuadros_de_calentamiento = 10
    resolucion_ideal = 640
    from google.colab.output import eval_js
    from inspeccion import sha256

    preparar_camara()

    if cuadros_de_prueba < 1 or cuadros_de_calentamiento < 0:
        raise ValueError('Revisá la cantidad de cuadros.')
    inspeccion.reiniciar_seguimiento()
    coco = YOLO('yolo11n.pt')

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
            imagen = decodificar_cuadro(foto)
            t0 = time.perf_counter()
            predicciones = inspeccion.procesar_lote([imagen], clasificar=clasificar)[0]
            predicciones = solo_manzanas(imagen, predicciones, coco, inspeccion.dispositivo)
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
