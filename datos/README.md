# Datos

| Conjunto | Uso en el proyecto |
|---|---|
| [Apple Sorting, edición RF100](originales/apple_sorting_rf100/) | Imágenes con recuadros alrededor de las manzanas. El notebook de detección usa [472 para entrenar, 177 para validar y 30 para probar](anotaciones/particiones_rf100_candidato.json). El detector que se usa en los videos se entrenó antes con [Apple Sorting v2](https://universe.roboflow.com/arfiani-nur-sayidah-9lizr/apple-sorting-2bfhk/dataset/2). |
| [Fruits-360](originales/variedades/) | Fotos de Crimson Snow, Granny Smith, Pink Lady y Red Delicious. El notebook separa 1.241 para entrenamiento, 502 para validación y 502 para prueba. Aparta 256 vistas en los límites entre grupos. |
| [Healthy-Defective-Fruits](originales/estado/) | 5.000 fotos de la primera etapa del proyecto. El [archivo de particiones](anotaciones/estado_particiones.csv) identifica las fotos seleccionadas y sus grupos. |
| [Regiones de la superficie](anotaciones/regiones_ssda_sda.csv) y [marcas normales](marcas/) | Ejemplos usados al desarrollar las dos redes de daño visible. Las anotaciones distinguen lesiones de rasgos como el tallo, el cáliz o una etiqueta. |
| Anotaciones de video | [Recuadros en las cintas](anotaciones/cinta_cajas_referencia.csv), [cruces en selección](anotaciones/seleccion_cruces_referencia.json) y [daño por pasada en Coral](anotaciones/coral_estado_visible.csv). |

La primera CNN de daño se entrenó con [SSDA](https://zenodo.org/records/10957715), [SDA](https://github.com/dev-johnlee/SD-Apples-with-KD/tree/4000e787b46ce6da3a074212cf18201372f98b8b) y las manzanas de [Bijoy y colaboradores](https://data.mendeley.com/datasets/xkbjx8959c/1). La segunda partió de esos pesos y sumó ejemplos de marcas normales y piel amarilla. Se eligió la época con fotos Nikon de [Joarder y colaboradores](https://data.mendeley.com/datasets/bdd69gyhv8/1). Los conjuntos completos de estas fuentes se obtienen en los enlaces. El notebook de video carga ambas redes ya entrenadas.

[Fuentes y licencias de las fotos](originales/FUENTES.md) · [Fuentes de los videos](../videos/FUENTES.md)
