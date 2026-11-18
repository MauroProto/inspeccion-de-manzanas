# Datos del proyecto

| Datos en GitHub | Uso |
|---|---|
| [Fruits-360 original-size](originales/variedades/) | 2.501 imágenes de Crimson Snow, Granny Smith, Pink Lady y Red Delicious. La CNN usa 2.245 imágenes repartidas en bloques de rotación, con un margen entre validación y prueba. |
| [Healthy-Defective-Fruits](originales/estado/) | 5.000 fotos reales. Se quitaron 257 duplicados exactos y se separaron grupos de imágenes parecidas con el [CSV](../notebooks/estado_particiones.csv). |
| [Apple Sorting RF100 release-640](originales/apple_sorting_rf100/) | 697 imágenes con cajas. Conserva las etiquetas originales `apple` y `damaged_apple`. |

El YOLO del grupo fue entrenado con la **exportación Apple Sorting v2** de Roboflow. GitHub tiene la edición **RF100 release-640**. Para reproducir el entrenamiento con v2, ver el [notebook de YOLO](../pruebas/yolo_opencv_manzanas-colab-test.ipynb). Ambas etiquetas de las cajas pasan a una sola clase, `manzana`.

Las fuentes y atribuciones están en [FUENTES.md](originales/FUENTES.md). Los videos están en [`videos/`](../videos/), con su [inventario](videos/README.md).
