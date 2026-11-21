"""Compara las detecciones y los cruces con las anotaciones de la cinta."""

import csv
import hashlib
import json
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import __version__

from seguimiento import ContadorCruces, SeguimientoConByteTrack

TOLERANCIA_CUADROS = 3
TOLERANCIA_PIXELES = 20


def iou(caja_a, caja_b):
    ax1, ay1, ax2, ay2 = caja_a
    bx1, by1, bx2, by2 = caja_b
    interseccion = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(0, min(ay2, by2) - max(ay1, by1))
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - interseccion
    return interseccion / union if union else 0


def dentro(caja, zona):
    x1, y1, x2, y2 = caja
    a, b, c, d = zona
    return a <= x1 < x2 <= c and b <= y1 < y2 <= d


def contar(referencias, predicciones, ignoradas, zona):
    pares = sorted(((iou(real, predicha), r, p)
                    for r, real in enumerate(referencias) for p, predicha in enumerate(predicciones)),
                   reverse=True)
    usadas_reales, usadas_predichas = set(), set()
    for coincidencia, r, p in pares:
        if coincidencia < .50:
            break
        if r not in usadas_reales and p not in usadas_predichas:
            usadas_reales.add(r)
            usadas_predichas.add(p)
    falsas = sum(p not in usadas_predichas and dentro(caja, zona)
                 and all(iou(caja, otra) < .30 for otra in ignoradas)
                 for p, caja in enumerate(predicciones))
    return len(usadas_reales), falsas, len(referencias) - len(usadas_reales)


def comparar(referencia, predicciones):
    """Empareja eventos por tiempo y altura de cruce. Cada evento entra una sola vez."""
    n, m = len(referencia), len(predicciones)
    imposible = 3 * (min(n, m) + 1)
    costos = np.full((n, m), imposible, dtype=float)
    for i, real in enumerate(referencia):
        for j, pred in enumerate(predicciones):
            tiempo = abs(real['cuadro_estimado'] - pred['cuadro_estimado'])
            altura = abs(real['x'] - pred['x'])
            if tiempo <= TOLERANCIA_CUADROS and altura <= TOLERANCIA_PIXELES:
                costos[i, j] = tiempo / TOLERANCIA_CUADROS + altura / TOLERANCIA_PIXELES

    pares = []
    for i, j in zip(*linear_sum_assignment(costos)):
        if costos[i, j] < imposible:
            real, pred = referencia[i], predicciones[j]
            pares.append({'id_referencia': real['id'], 'id_predicho': pred['id'],
                          'cuadro_referencia': real['cuadro_estimado'],
                          'cuadro_predicho': pred['cuadro_estimado'],
                          'y_referencia': real['x'], 'y_predicho': pred['x']})
    return {
        'referencia': n, 'predicciones': m, 'coincidencias': len(pares),
        'omitidos': n - len(pares), 'sin_referencia': m - len(pares),
        'referencias_con_varias_opciones': int(((costos < imposible).sum(axis=1) > 1).sum()),
        'pares': pares,
    }


def huella(ruta):
    with ruta.open('rb') as archivo:
        return hashlib.file_digest(archivo, 'sha256').hexdigest()


def preparar_referencia(datos):
    if datos['anotacion_completa'] is not True or not datos['anotador'] or not datos['linea']:
        raise ValueError('Primero completá la referencia visual y fijá la línea en el manifiesto.')
    ancho, alto = datos['resolucion_anotacion']
    inicio, fin = datos['ventana']
    if not 0 <= inicio <= fin < datos['cuadros']:
        raise ValueError('La ventana está fuera del video.')

    linea = datos['linea']
    eje = linea['eje']
    if eje not in ('x', 'y') or linea['sentido'] not in ('creciente', 'decreciente'):
        raise ValueError('Elegí eje x o y y un sentido creciente o decreciente.')
    limite, transversal = (ancho, alto) if eje == 'x' else (alto, ancho)
    if not (0 < linea['posicion'] < limite and 0 <= linea['desde'] < linea['hasta'] <= transversal):
        raise ValueError('La línea o sus extremos están fuera de la resolución anotada.')

    eventos = []
    identificadores = set()
    for pasada in datos['pasadas']:
        identificador, cuadro, punto = pasada['pasada'], pasada['cuadro'], pasada['punto']
        if not identificador or identificador in identificadores:
            raise ValueError('Las pasadas anotadas deben tener nombres distintos.')
        if not (isinstance(cuadro, int) and inicio <= cuadro <= fin
                and linea['desde'] <= punto <= linea['hasta']):
            raise ValueError(f'La pasada {identificador} está fuera de la línea o ventana.')
        eventos.append({'id': identificador, 'cuadro_estimado': cuadro, 'x': punto})
        identificadores.add(identificador)
    return eventos


def contar_predicciones(ruta, datos, resolucion_cajas):
    linea = datos['linea']
    creciente = linea['sentido'] == 'creciente'
    contador = ContadorCruces(y=linea['posicion'], x_desde=linea['desde'], x_hasta=linea['hasta'],
                             sentido='abajo' if creciente else 'arriba', margen=4, max_salto=6)

    ancho, alto = datos['resolucion_anotacion']
    ancho_cajas, alto_cajas = resolucion_cajas
    if min(ancho_cajas, alto_cajas) <= 0:
        raise ValueError('Indicá la resolución en la que están las cajas.')
    escala_x, escala_y = ancho / ancho_cajas, alto / alto_cajas
    with ruta.open(encoding='utf-8-sig', newline='') as archivo:
        filas = csv.DictReader(archivo)
        columnas = {'cuadro', 'id', 'x1', 'y1', 'x2', 'y2'}
        if not columnas.issubset(filas.fieldnames or []):
            raise ValueError('El CSV necesita cuadros, IDs y coordenadas de las cajas.')
        for fila in filas:
            cuadro, identificador = int(fila['cuadro']), int(fila['id'])
            if not 0 <= cuadro < datos['cuadros']:
                raise ValueError(f'El cuadro {cuadro} del CSV está fuera del video.')
            x1, y1, x2, y2 = (float(fila[campo]) for campo in ('x1', 'y1', 'x2', 'y2'))
            if not (0 <= x1 < x2 <= ancho_cajas and 0 <= y1 < y2 <= alto_cajas):
                raise ValueError(f'Caja inválida en el cuadro {cuadro}.')
            x1, x2, y1, y2 = x1 * escala_x, x2 * escala_x, y1 * escala_y, y2 * escala_y
            caja = (y1, x1, y2, x2) if linea['eje'] == 'x' else (x1, y1, x2, y2)
            contador.actualizar(cuadro, identificador, caja)

    inicio, fin = datos['ventana']
    return [e for e in contador.cruces if inicio <= e['cuadro_estimado'] <= fin]


def verificar(manifiesto, predicciones, video, cuadros_procesados, resolucion_cajas):
    datos = json.loads(manifiesto.read_text(encoding='utf-8'))
    referencia = preparar_referencia(datos)
    if cuadros_procesados != datos['cuadros']:
        raise ValueError('La corrida debe procesar todos los cuadros del video.')
    if huella(video) != datos['sha256_video']:
        raise ValueError('El video no coincide con el original anotado.')
    eventos = contar_predicciones(predicciones, datos, resolucion_cajas)
    return {
        'video': video.name, 'sha256_video': datos['sha256_video'],
        'sha256_manifiesto': huella(manifiesto), 'sha256_predicciones': huella(predicciones),
        'ventana': datos['ventana'], 'linea': datos['linea'],
        'tolerancia_cuadros': TOLERANCIA_CUADROS,
        'tolerancia_pixeles_en_linea': TOLERANCIA_PIXELES,
        **comparar(referencia, eventos),
    }


def escribir_csv(ruta, columnas, filas):
    with ruta.open('w', encoding='utf-8', newline='') as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)


def evaluar(video, manifiesto, modelo, pesos, clase, destino):
    seguidor = SeguimientoConByteTrack()
    captura = cv2.VideoCapture(str(video))
    if not captura.isOpened():
        raise ValueError(f'No se pudo leer {video.name}.')

    inicio = time.perf_counter()
    cuadro = 0
    cajas = []
    try:
        while True:
            lote = []
            for _ in range(6):
                ok, imagen = captura.read()
                if not ok:
                    break
                lote.append(imagen)
            if not lote:
                break
            predicciones = modelo.predict(lote, imgsz=1280, conf=.10, iou=.70, max_det=300,
                                           classes=[clase], device=0, verbose=False)
            if len(predicciones) != len(lote):
                raise RuntimeError('Faltan predicciones para algunos cuadros.')
            for prediccion in predicciones:
                cajas.extend(seguidor.actualizar(cuadro, prediccion.boxes, contar=False))
                cuadro += 1
    finally:
        captura.release()

    destino.mkdir(parents=True)
    ruta_cajas = destino / 'cajas.csv'
    escribir_csv(ruta_cajas, ['cuadro', 'id', 'x1', 'y1', 'x2', 'y2', 'score_detector'], cajas)
    medidas = verificar(manifiesto, ruta_cajas, video, cuadro, (3840, 2160))
    medidas.update(imgsz=1280, conf=.10, max_det=300, iou_nms=.70, cuadros_procesados=cuadro,
                   segundos_archivo=time.perf_counter()-inicio, pesos=pesos.name,
                   sha256_pesos=huella(pesos), ultralytics=__version__, cajas=len(cajas),
                   cajas_con_id=sum(f['id'] >= 0 for f in cajas))
    (destino / 'evaluacion.json').write_text(json.dumps(medidas, indent=2, ensure_ascii=False))
    print(destino.name, cuadro, len(cajas), medidas['coincidencias'],
          medidas['omitidos'], medidas['sin_referencia'], flush=True)
