from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from scipy.fft import dctn
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import StratifiedGroupKFold

repo = Path(__file__).resolve().parents[1]
raiz = repo / 'datos/originales/estado'
filas, huellas, vistos = [], [], set()
for archivo in sorted(raiz.glob('*/*')):
    imagen = Image.open(archivo).convert('RGB')
    huella = hashlib.sha256(imagen.tobytes()).hexdigest()
    if huella in vistos:
        continue
    vistos.add(huella)
    gris = np.asarray(imagen.convert('L').resize((32, 32)), dtype=float)
    coeficientes = dctn(gris, type=2)[:8, :8].flatten()
    huellas.append(coeficientes > np.median(coeficientes[1:]))
    subtipo = archivo.parent.name
    filas.append({'archivo':str(archivo.relative_to(raiz)), 'subtipo':subtipo,
                  'clase':'no_defectuosa' if subtipo == 'fresh' else 'defectuosa'})

# Las fotos muy parecidas quedan juntas, incluso si tienen distinto nombre.
vecinos = NearestNeighbors(radius=2/64, metric='hamming', algorithm='brute', n_jobs=2)
vecinos.fit(np.asarray(huellas))
cercanos = vecinos.radius_neighbors(return_distance=False)
padres = list(range(len(filas)))
def grupo(i):
    while padres[i] != i:
        padres[i] = padres[padres[i]]
        i = padres[i]
    return i
for i, otros in enumerate(cercanos):
    for j in otros:
        padres[grupo(int(j))] = grupo(i)
tabla = pd.DataFrame(filas)
tabla['grupo'] = [grupo(i) for i in range(len(filas))]
tabla['parte'] = 'entrenamiento'
particion = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for numero, (_, indices) in enumerate(particion.split(tabla, tabla.subtipo, tabla.grupo)):
    if numero == 0: tabla.loc[indices, 'parte'] = 'prueba'
    if numero == 1: tabla.loc[indices, 'parte'] = 'validacion'
tabla.to_csv(repo / 'notebooks/estado_particiones.csv', index=False)
print(pd.crosstab(tabla.parte, tabla.clase))
print('Grupos:',tabla.grupo.nunique(),'mayor grupo:',tabla.groupby('grupo').size().max())
assert tabla.groupby('grupo').parte.nunique().max() == 1
