"""Dibuja las predicciones y compara cada pasada con su referencia."""

from collections import Counter

import cv2
import numpy as np
from PIL import Image, ImageDraw


def dibujar(imagen, cajas, umbral, fuente):
    if not cajas:
        return imagen
    alto, ancho = imagen.shape[:2]
    lienzo = Image.fromarray(cv2.cvtColor(imagen, cv2.COLOR_BGR2RGB))
    lapiz = ImageDraw.Draw(lienzo)
    for caja in cajas:
        x1, y1, x2, y2 = [round(caja[k]) for k in ('x1', 'y1', 'x2', 'y2')]
        puntaje = caja['p_danio']
        estado = 'Estado por revisar'
        if puntaje is not None:
            estado = 'Con daño visible' if puntaje >= umbral else 'Sin daño visible'
        borde = (239, 158, 110) if puntaje is not None and puntaje >= umbral else (129, 207, 227)
        lapiz.rectangle((x1, y1, x2, y2), outline=borde, width=2)

        variedad = caja['variedad']
        texto = f'{variedad} (estimada) · {estado}' if variedad else estado
        izq, arriba, der, abajo = lapiz.textbbox((0, 0), texto, font=fuente)
        ancho_texto, alto_texto = der - izq + 14, abajo - arriba + 8
        x = max(0, min(x1, ancho - ancho_texto))
        y = y1 - alto_texto - 2
        if y < 0:
            y = max(0, min(y2 + 2, alto - alto_texto))
        lapiz.rectangle((x, y, x + ancho_texto, y + alto_texto), fill=(25, 29, 37))
        lapiz.text((x + 7 - izq, y + 4 - arriba), texto, font=fuente, fill='white')
    return cv2.cvtColor(np.asarray(lienzo), cv2.COLOR_RGB2BGR)


def evaluar_pasadas(referencias, predichas, umbral):
    resultados = []
    for fila in referencias:
        desde, hasta = int(fila['cuadro_desde']), int(fila['cuadro_hasta'])
        cuadros = [i for i in range(desde, hasta + 1) if i % 2 == 0]
        puntajes = [predichas[i]['p_danio'] for i in cuadros if i in predichas
                    and predichas[i]['p_danio'] is not None]
        promedio = sum(puntajes) / len(puntajes) if len(puntajes) >= len(cuadros) / 2 else None
        estado = None if promedio is None else ('defectuosa' if promedio >= umbral else 'no defectuosa')
        resultados.append({'pasada': fila['manzana'], 'real': fila['estado'], 'prediccion': estado,
                           'puntaje': promedio, 'cuadros_con_decision': len(puntajes),
                           'cuadros_referencia': len(cuadros)})
    resumen = {}
    for clase in ('defectuosa', 'no defectuosa'):
        elegidas = [f for f in resultados if f['real'] == clase]
        resumen[clase] = {'total': len(elegidas), 'predicciones':
                          dict(Counter(f['prediccion'] or 'sin_decision' for f in elegidas))}
    return resumen, resultados
