"""Procesa los cuadros en orden y escribe un MP4 a la resolución del original."""

import math
import subprocess
from pathlib import Path

import cv2

from coral import dibujar


def dibujar_cuadro(imagen, cajas, umbral, fuente, clasificar):
    if clasificar:
        return dibujar(imagen, cajas, umbral, fuente)
    vista = imagen.copy()
    for caja in cajas:
        x1, y1, x2, y2 = [round(caja[k]) for k in ('x1', 'y1', 'x2', 'y2')]
        cv2.rectangle(vista, (x1, y1), (x2, y2), (227, 207, 129), 2)
    return vista


def procesar_video(origen, destino, anotar):
    """anotar recibe el número de cuadro y su imagen BGR."""
    destino = Path(destino)
    if Path(origen).resolve() == destino.resolve():
        raise ValueError('Elegí un archivo de salida distinto del original.')
    temporal = destino.with_name(destino.stem + '.tmp.mp4')
    captura = cv2.VideoCapture(str(origen))
    try:
        ok, imagen = captura.read()
        if not ok:
            raise ValueError(f'No se pudo leer el primer cuadro de {origen}.')
        fps = captura.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError('El archivo no indica cuántos cuadros por segundo tiene.')
        alto, ancho = imagen.shape[:2]
        comando = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                   '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{ancho}x{alto}',
                   '-r', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
                   '-vf', 'pad=ceil(iw/2)*2:ceil(ih/2)*2', '-crf', '18',
                   '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(temporal)]
        with subprocess.Popen(comando, stdin=subprocess.PIPE) as escritor:
            with escritor.stdin:
                cuadro = 0
                while ok:
                    escritor.stdin.write(anotar(cuadro, imagen).tobytes())
                    cuadro += 1
                    ok, imagen = captura.read()
            if escritor.wait() != 0:
                raise RuntimeError('FFmpeg no pudo terminar el video.')
        temporal.replace(destino)
    finally:
        captura.release()
        temporal.unlink(missing_ok=True)
