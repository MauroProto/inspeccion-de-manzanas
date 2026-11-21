# Inspección de manzanas

Trabajo final de Aprendizaje Automático II, Universidad de San Andrés. YOLO11n detecta las manzanas en video, ByteTrack las sigue y las redes convolucionales estiman la variedad y el daño visible.

## Notebooks en Google Colab

- [Detección y conteo en cinta](https://colab.research.google.com/github/MauroProto/inspeccion-de-manzanas/blob/main/notebooks/deteccion_y_conteo_en_cinta.ipynb) · Evalúa YOLO y compara dos entrenamientos.
- [Clasificación de variedades](https://colab.research.google.com/github/MauroProto/inspeccion-de-manzanas/blob/main/notebooks/clasificacion_de_variedades.ipynb) · Entrena y compara redes con fotos de cuatro variedades.
- [Inspección de manzanas en video](https://colab.research.google.com/github/MauroProto/inspeccion-de-manzanas/blob/main/notebooks/inspeccion_de_manzanas_en_video.ipynb) · Evalúa las dos CNN de daño visible ya entrenadas en el video de Coral. También permite probar otro MP4.
- [Inspección en vivo con la webcam](https://colab.research.google.com/github/MauroProto/inspeccion-de-manzanas/blob/main/notebooks/inspeccion_con_webcam.ipynb) · Corré las dos celdas para ver el recuadro, la variedad estimada y el daño visible en cada manzana.

Los tres primeros notebooks incluyen las tablas y los gráficos de las corridas en Colab. Para repetirlas, elegí una GPU. Las salidas quedan en `/content/resultados`.

[Datos y fuentes](datos/README.md) · [Pesos de los modelos](pesos/) · [Cinco videos originales](videos/)

## Videos de la demo

### Coral · Manzanas de a una · 1,5×

https://github.com/user-attachments/assets/54a59e18-2578-4dea-8ce2-2296e51577e2

### Cinta de cerca

https://github.com/user-attachments/assets/14812c35-5cf8-428d-b961-388e83477b67

<details>
<summary>Ver los otros tres videos</summary>

### Cinta industrial · Plano general

https://github.com/user-attachments/assets/0b1821fb-8c83-433c-920f-0a56e13c0567

### Selección de manzanas

https://github.com/user-attachments/assets/1995f7f0-6597-4fde-abae-e34e4e0f2653

### Manzanas en agua

https://github.com/user-attachments/assets/face3782-05bc-4602-8139-6b92313107fa

</details>

En los videos de planta, las etiquetas de variedad y daño son estimaciones sin verificar por fruta.

[Descargar los MP4 de la demo](https://github.com/MauroProto/inspeccion-de-manzanas/releases/tag/videos-demo)
