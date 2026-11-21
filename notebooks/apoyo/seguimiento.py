"""Identifica las manzanas entre cuadros y cuenta sus cruces."""


class ContadorCruces:
    def __init__(self, y=200, x_desde=320, x_hasta=800, sentido='abajo', margen=4, max_salto=12):
        if x_desde >= x_hasta or sentido not in ('arriba', 'abajo') or margen <= 0 or max_salto < 1:
            raise ValueError('Revisá la línea, el sentido y el margen del conteo.')
        self.y = y
        self.x_desde, self.x_hasta = x_desde, x_hasta
        self.sentido = 1 if sentido == 'abajo' else -1
        self.margen, self.max_salto = margen, max_salto
        self.cruces = []
        self._contados = set()
        self._antes = {}
        self._ultimo_cuadro = {}
        self._cuadro = -1

    @property
    def total(self):
        return len(self.cruces)

    def actualizar(self, cuadro, identificador, caja):
        """Recibe una caja en coordenadas del video. Devuelve el cruce si se confirma."""
        if cuadro < self._cuadro:
            raise ValueError('Procesá los cuadros en orden.')
        self._cuadro = cuadro
        if identificador < 0 or identificador in self._contados:
            return None

        ultimo = self._ultimo_cuadro.get(identificador)
        if ultimo == cuadro:
            raise ValueError(f'El ID {identificador} aparece dos veces en el cuadro {cuadro}.')
        if ultimo is not None and cuadro - ultimo > self.max_salto:
            self._antes.pop(identificador, None)
        self._ultimo_cuadro[identificador] = cuadro
        x1, y1, x2, y2 = caja
        if not (x1 < x2 and y1 < y2):
            raise ValueError(f'Caja inválida en el cuadro {cuadro}.')
        x, centro_y = (x1 + x2) / 2, (y1 + y2) / 2
        distancia = self.sentido * (centro_y - self.y)

        if distancia <= -self.margen:
            self._antes[identificador] = (cuadro, x, distancia)
            return None
        if distancia < self.margen:
            return None
        antes = self._antes.pop(identificador, None)
        if antes is None:
            return None

        cuadro_anterior, x_anterior, distancia_anterior = antes
        fraccion = -distancia_anterior / (distancia - distancia_anterior)
        x_cruce = x_anterior + fraccion * (x - x_anterior)
        if not self.x_desde <= x_cruce <= self.x_hasta:
            return None

        cruce = {
            'id': identificador, 'cuadro': cuadro,
            'cuadro_estimado': round(cuadro_anterior + fraccion * (cuadro - cuadro_anterior)),
            'x': x_cruce,
        }
        self._contados.add(identificador)
        self.cruces.append(cruce)
        return cruce


class SeguimientoConByteTrack:
    def __init__(self, contador=None):
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, YAML
        from ultralytics.utils.checks import check_yaml

        configuracion = IterableSimpleNamespace(**YAML.load(check_yaml('bytetrack.yaml')))
        self.tracker = BYTETracker(configuracion)
        self.contador = ContadorCruces() if contador is None else contador
        self._cuadro = None

    def actualizar(self, cuadro, cajas_yolo, contar=True):
        """Usa las cajas y sus puntajes originales. Llamar también en cuadros vacíos."""
        if self._cuadro is not None and cuadro != self._cuadro + 1:
            raise ValueError('Pasá cuadros consecutivos, incluso cuando YOLO no encuentre manzanas.')
        self._cuadro = cuadro
        cajas = cajas_yolo.cpu().numpy()
        trayectorias = self.tracker.update(cajas)
        ids = {int(fila[-1]): int(fila[4]) for fila in trayectorias}
        filas = []
        for i, (coordenadas, confianza) in enumerate(zip(cajas.xyxy, cajas.conf)):
            identificador = ids.get(i, -1)
            caja = tuple(float(c) for c in coordenadas)
            if contar:
                self.contador.actualizar(cuadro, identificador, caja)
            x1, y1, x2, y2 = caja
            filas.append(dict(cuadro=cuadro, id=identificador, x1=x1, y1=y1,
                              x2=x2, y2=y2, score_detector=float(confianza)))
        return filas
