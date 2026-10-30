# Fuentes de los datasets

| Carpeta | Contenido | Fuente | Licencia |
|---|---|---|---|
| `variedades/` | 2.501 fotos; cuatro variedades | [Fruits-360](https://github.com/fruits-360/fruits-360-original-size/tree/e09de71fa3e80624be016e1b5c90c1c310d046c5) · Mihai Oltean | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| `estado/` | 5.000 fotos reales de manzanas | [Healthy-Defective-Fruits](https://github.com/luischuquim/Healthy-Defective-Fruits/tree/967ef53c60e195fcda3d64cff90d28cacfc50540) · Chuquimarca, González, Pacheco, Vintimilla y Velastin | [CC BY 4.0](https://data.mendeley.com/datasets/p73b8byghp/1) |

Las imágenes se copiaron sin modificar. Se seleccionaron únicamente los subconjuntos de manzanas que vamos a usar.

- **Variedades:** Granny Smith (640), Red Delicious (600), Pink Lady (625) y Crimson Snow (636). Se conservan las carpetas originales `Training`, `Validation` y `Test`. Esa separación puede contener vistas de la misma fruta; revisar los grupos antes de evaluar generalización.
- **Estado:** `fresh` (2.500), `bruise_defect` (750), `rot_defect` (1.000) y `scab_defect` (750). Para la clasificación binaria, `fresh` será no defectuosa y las otras tres carpetas, defectuosa. Se conserva el subtipo original. No se incluyen imágenes sintéticas.

Los archivos `README_ORIGINAL.md`, `LICENSE` y `CITATION.cff`, cuando están disponibles, pertenecen a las fuentes originales.
