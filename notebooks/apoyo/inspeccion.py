"""Detecta, sigue y clasifica manzanas de un archivo o una cámara."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from ultralytics import YOLO

from seguimiento import SeguimientoConByteTrack


def ventanas_piel(foto):
    ancho, alto = foto.size
    for proporcion in (.25, .34):
        lado = max(2, round(min(ancho, alto) * proporcion))
        for x in (.20, .35, .50, .65, .80):
            for y in (.20, .35, .50, .65, .80):
                if ((x-.5)/.5)**2 + ((y-.5)/.5)**2 <= .58:
                    x1, y1 = round(ancho*x-lado/2), round(alto*y-lado/2)
                    yield x1, y1, x1+lado, y1+lado


def cargar_cnn(ruta, dispositivo):
    if not ruta.is_file():
        raise FileNotFoundError(f'No se encontró el modelo {ruta}.')
    datos = torch.load(ruta, map_location='cpu', weights_only=True)
    red = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(),
        nn.Linear(1024, 64), nn.ReLU(), nn.Dropout(.5), nn.Linear(64, len(datos['clases'])),
    )
    red.load_state_dict(datos['pesos'])
    return red.to(dispositivo).eval(), datos


def sha256(ruta):
    with Path(ruta).open('rb') as archivo:
        return hashlib.file_digest(archivo, 'sha256').hexdigest()


class Inspeccion:
    def __init__(self, pesos_yolo, pesos_variedad=None, pesos_estado=None,
                 dispositivo='cuda', imgsz=640, conf=.35, max_det=10, *, pesos_confirmacion=None):
        pesos_yolo = Path(pesos_yolo)
        if not pesos_yolo.is_file():
            raise FileNotFoundError(f'No se encontró el detector {pesos_yolo}.')
        self.yolo = YOLO(pesos_yolo)
        if self.yolo.names != {0: 'manzana'}:
            raise ValueError('Usá el detector entrenado para la clase manzana.')
        self.dispositivo = dispositivo
        self.imgsz, self.conf, self.max_det = imgsz, conf, max_det
        self.archivos = {'yolo': pesos_yolo}
        self.red_variedad = None
        self.red_estado = None
        self.red_confirmacion = None
        self.umbral_danio = .5

        if pesos_variedad is not None:
            ruta = Path(pesos_variedad)
            self.red_variedad, datos = cargar_cnn(ruta, dispositivo)
            self.variedades = datos['clases']
            self.preparar_variedad = transforms.Compose([
                transforms.Resize((datos['tamano'], datos['tamano'])),
                transforms.ToTensor(),
                transforms.Normalize(datos['media'], datos['desvio']),
            ])
            self.archivos['variedad'] = ruta

        if pesos_estado is not None:
            ruta = Path(pesos_estado)
            self.red_estado, datos = cargar_cnn(ruta, dispositivo)
            if set(datos['clases']) != {'Defectuosa', 'No defectuosa'} or datos['tamano'] != 64:
                raise ValueError('Usá el modelo de estado entrenado con parches RGB de 64 × 64.')
            if datos.get('normalizacion_ejes', (1, 2, 3)) != (1, 2, 3):
                raise ValueError('El modelo de estado debe centrar cada parche con una sola media RGB.')
            self.indice_danio = datos['clases'].index('Defectuosa')
            self.umbral_danio = float(datos.get('umbral', .5))
            self.archivos['estado'] = ruta
        if pesos_confirmacion is not None:
            ruta = Path(pesos_confirmacion)
            self.red_confirmacion, datos = cargar_cnn(ruta, dispositivo)
            if (self.red_estado is None or self.umbral_danio != .5 or datos['tamano'] != 64
                    or tuple(datos.get('normalizacion_ejes', (1, 2, 3))) != (1, 2, 3)
                    or set(datos['clases']) != {'Defectuosa', 'No defectuosa'}):
                raise ValueError('El acuerdo requiere dos CNN de estado compatibles y umbral 0,50.')
            self.indice_confirmacion = datos['clases'].index('Defectuosa')
            self.archivos['confirmacion'] = ruta
        self.reiniciar_seguimiento()

    def reiniciar_seguimiento(self):
        self.seguimiento = SeguimientoConByteTrack()
        self.cuadro = 0

    def clasificar_variedad(self, fotos, filas):
        lote = torch.stack([self.preparar_variedad(foto) for foto in fotos])
        puntajes, indices = self.red_variedad(lote.to(self.dispositivo)).softmax(1).max(1)
        for fila, indice, puntaje in zip(filas, indices.tolist(), puntajes.tolist()):
            fila['variedad'] = self.variedades[indice]
            fila['p_variedad'] = puntaje

    def clasificar_estado(self, fotos, filas):
        parches, cantidades = [], []
        for foto in fotos:
            cajas = list(ventanas_piel(foto))
            cantidades.append(len(cajas))
            for caja in cajas:
                parche = foto.crop(caja).resize((64, 64), Image.Resampling.BILINEAR)
                parches.append(torch.from_numpy(np.array(parche)).permute(2, 0, 1))
        probabilidades, confirmaciones = [], []
        for lote in torch.stack(parches).split(96):
            lote = lote.to(self.dispositivo).float().div(255)
            lote = (lote - lote.mean((1, 2, 3), keepdim=True)).mul(2)
            probabilidades.append(self.red_estado(lote).softmax(1)[:, self.indice_danio].cpu())
            if self.red_confirmacion is not None:
                confirmaciones.append(self.red_confirmacion(lote).softmax(1)[:, self.indice_confirmacion].cpu())
        puntajes = [p.topk(3).values.mean().item() for p in torch.cat(probabilidades).split(cantidades)]
        if confirmaciones:
            # El daño queda confirmado cuando ambas CNN superan el mismo umbral.
            segundo = [p.topk(3).values.mean().item() for p in torch.cat(confirmaciones).split(cantidades)]
            puntajes = [min(primero, otro) for primero, otro in zip(puntajes, segundo, strict=True)]
        for fila, puntaje in zip(filas, puntajes, strict=True):
            fila['p_danio'] = puntaje

    @torch.inference_mode()
    def procesar_lote(self, imagenes, *, clasificar=True, imgsz=None, conf=None, max_det=None):
        """Recibe cuadros BGR consecutivos y devuelve sus cajas en píxeles originales."""
        if not imagenes:
            return []
        predicciones = self.yolo.predict(
            imagenes, imgsz=self.imgsz if imgsz is None else imgsz,
            conf=self.conf if conf is None else conf, max_det=self.max_det if max_det is None else max_det,
            iou=.7, device=self.dispositivo, verbose=False,
        )
        if len(predicciones) != len(imagenes):
            raise RuntimeError('YOLO no devolvió una salida por cuadro.')
        por_cuadro, fotos, filas_recortadas = [], [], []
        usar_cnn = clasificar and (self.red_estado is not None or self.red_variedad is not None)
        for imagen, resultado in zip(imagenes, predicciones):
            filas = self.seguimiento.actualizar(self.cuadro, resultado.boxes, contar=False)
            self.cuadro += 1
            por_cuadro.append(filas)
            alto, ancho = imagen.shape[:2]
            for fila in filas:
                fila.update(variedad=None, p_variedad=None, p_danio=None)
                if usar_cnn:
                    x1 = max(0, int(fila['x1']))
                    y1 = max(0, int(fila['y1']))
                    x2 = min(ancho, int(fila['x2']))
                    y2 = min(alto, int(fila['y2']))
                    if x1 < x2 and y1 < y2:
                        rgb = cv2.cvtColor(imagen[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
                        fotos.append(Image.fromarray(rgb))
                        filas_recortadas.append(fila)
        if fotos and self.red_variedad is not None:
            self.clasificar_variedad(fotos, filas_recortadas)
        if fotos and self.red_estado is not None:
            self.clasificar_estado(fotos, filas_recortadas)
        return por_cuadro
