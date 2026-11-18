# Estado del proyecto · 17 de noviembre de 2026

El sistema procesa videos con los pesos ya entrenados. YOLO11n localiza las manzanas, ByteTrack asigna IDs y dos CNN evalúan cada recorte. El [Colab 03](../notebooks/03_video_y_camara.ipynb) se ejecutó desde una sesión reiniciada y generó un MP4, un CSV y un registro de tiempos. [Instrucciones y archivos](../notebooks/README.md).

| Componente | Evidencia |
|---|---|
| YOLO11n | El [notebook del grupo](../pruebas/yolo_opencv_manzanas-colab-test.ipynb) registró mAP50 **0,960** y mAP50–95 **0,660** en 30 imágenes de Apple Sorting v2 (157 cajas). La prueba en cinta necesita su propia referencia. |
| Variedad | Macro-F1 **0,800** en validación y **1,000** en prueba interna. Fruits-360 contiene vistas relacionadas. La pérdida al elegir los pesos era **0,044** en entrenamiento y **0,483** en validación. Falta comprobar el desempeño con otras frutas físicas. |
| Estado | Macro-F1 **0,917** en prueba. De 449 fotos defectuosas, detectó 448. De 585 fotos sin defecto, rechazó incorrectamente 85. La separación agrupó fotos parecidas, con identidad física desconocida. |
| Video | Dos clips completos. En una T4, **28,1 FPS** en `cinta_industrial` y **7,3–9,2 FPS** en `manzanas_en_cinta`, incluyendo lectura y escritura. Ambos clips de entrada rondan los 30 FPS. La medición del bucle excluye la exportación H.264 y la transferencia a Drive. |

[Curvas, matrices, métricas y archivos de la demo](../resultados/colab_2026-11-17/).

En `manzanas_en_cinta`, **6.197 de 6.244 predicciones por cuadro** fueron «defectuosa». Es una señal de que el modelo se está usando en imágenes distintas de las de entrenamiento. Como no hay etiquetas de referencia en ese clip, esa proporción describe las salidas del sistema. En `cinta_industrial` se observan varias manzanas pequeñas omitidas y algunas cajas sobre partes de la máquina.

## Qué falta para la entrega

1. **Prueba con frutas conocidas.** Filmar o conseguir un lote chico con variedades identificadas y defectos visibles revisados por una persona. Anotar cajas e identidad en los cuadros que se evalúen. Reservar frutas y clips completos para prueba. Medir detección y clasificación de punta a punta, contando las manzanas omitidas.
2. **Comparar el YOLO base y el fine-tuned.** Evaluar YOLO11n de COCO y los pesos del grupo sobre los mismos cuadros y las mismas cajas. Para repetir la corrida publicada hace falta la exportación exacta de Apple Sorting v2. El repositorio conserva la edición RF100 release-640.
3. **Probar la webcam y medir velocidad en el equipo de presentación.** El notebook incluye la celda de cámara, todavía sin prueba física. Si 30 FPS es el objetivo, comparar resolución y frecuencia de clasificación manteniendo fija la prueba de calidad.
4. **Cerrar el informe y la defensa.** Explicar las particiones, el entrenamiento, los resultados medidos y la diferencia entre fotos y video. La comparación con y sin aumentos del plan original sigue pendiente. Pedir a otro integrante que ejecute la demo desde su cuenta siguiendo el README.
