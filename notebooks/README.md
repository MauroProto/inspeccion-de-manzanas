# Colab

| Notebook | Para qué sirve |
|---|---|
| [01_variedades.ipynb](01_variedades.ipynb) | Entrenar desde cero la CNN de cuatro variedades. |
| [02_estado.ipynb](02_estado.ipynb) | Entrenar desde cero la CNN de defectuosa / no defectuosa. |
| [03_video_y_camara.ipynb](03_video_y_camara.ipynb) | Procesar video usando los tres modelos guardados. |

Para probar la demo, abrí el notebook 03 en Colab, seleccioná un entorno con GPU y ejecutá las celdas en orden. Los notebooks 01 y 02 permiten repetir los entrenamientos por separado.

Creá una carpeta llamada `Inspeccion de manzanas - Colab` en **Mi unidad** de Google Drive y poné estos archivos directamente dentro:

- [cnn_variedades.pt](../pesos/cnn_variedades.pt), [cnn_estado.pt](../pesos/cnn_estado.pt) y [yolo_manzanas_best.pt](../pruebas/yolo_manzanas_best.pt).
- [manzanas_en_cinta.mp4](../videos/manzanas_en_cinta.mp4), o el [video](../videos/) que quieras usar. Indicá su nombre en `nombre_video`.
- [estado_particiones.csv](estado_particiones.csv), si vas a ejecutar el notebook 02.

Las copias originales de trabajo están en [esta carpeta de Drive](https://drive.google.com/drive/folders/10LdiPiO9YbL18wMG3iVwxnGEkd7VkMtI). Si usás otra cuenta, podés obtener todos los archivos de esa lista desde el repositorio privado y montar tu propio Drive. Para la demo no necesitás cuenta de Roboflow.

`segundos = 0` procesa el video completo. `vista_previa = True` muestra cada décimo cuadro mientras se procesa. El notebook guarda el MP4, un CSV de predicciones y un JSON con los tiempos. Las coordenadas corresponden al video reducido a un ancho máximo de 960 píxeles.

La celda de cámara es opcional. Cambiá `usar_camara` a `True` y concedé permiso en tu navegador para probarla. Todavía falta verificarla con una webcam.

Los dos entrenamientos guardan el orden de clases y la normalización con los pesos. El [CSV de estado](estado_particiones.csv) fija las particiones. [Este script](preparar_particiones_estado.py) permite regenerarlas desde los originales del repositorio.

Para repetir el fine-tuning, el [notebook del grupo](../pruebas/yolo_opencv_manzanas-colab-test.ipynb) descarga Apple Sorting v2 con el secreto `ROBOFLOW_API_KEY` de Colab. Su sección de video es una prueba anterior de detección. La integración completa está en 03.
