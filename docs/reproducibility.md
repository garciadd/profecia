# RO-Crate por ejecución de entrenamiento

PROFECIA puede generar un RO-Crate independiente al terminar correctamente cada
ejecución de `scripts/workflow_step_by_step_configurable.py`. La función está
desactivada por defecto y un fallo en esta etapa no cambia el resultado del
entrenamiento; el fallo queda en el log y en
`workflow_step_by_step_summary.json`.

## Activación

Añada a `config/train.toml`:

```toml
[reproducibility]
create_rocrate = true
output_dir = "outputs/ro-crates"
upstream_rocrate = "https://repositorio.example/release/ro-crate-metadata.json"
input_catalogue = "config/data.toml"

[reproducibility.model_artifact]
mode = "attached" # attached | external | metadata_only

[reproducibility.croissant]
mode = "descriptive" # descriptive | materialized
include_split_manifest = true
split_manifest_format = "npz"

[reproducibility.code]
mode = "reference" # reference | patch | snapshot
require_clean_repository = true

[reproducibility.licenses]
metadata = "https://spdx.org/licenses/CC0-1.0.html"
code = ""
model = ""
results = ""
derived_dataset = ""

[reproducibility.environment]
lock_file = ""
container_image = ""
container_digest = ""
```

Las rutas relativas se interpretan desde el directorio de trabajo. También se
acepta como `upstream_rocrate` una carpeta local, una ruta directa a
`ro-crate-metadata.json`, un URL o un DOI. Para páginas como ROHub, el resolver
sigue el `Link` HTTP al linkset y desde allí el `describedby` JSON-LD.

El catálogo TOML se inspecciona para obtener los NetCDF, arrays y máscaras
seleccionados. Para la configuración estándar, los nombres de variables y
máscaras también se resuelven mediante los mapas usados por `src.data.io`.

Cuando el nombre local no coincide con el publicado, el catálogo admite un ID
explícito. El ID puede ser absoluto o un sufijo único del `@id` upstream:

```toml
[variables.T2M]
local_file = "t2m_1982_2022_monthly_0.5deg.nc"
upstream_entity_id = "referenced-data/monthly/era5_single_levels_t2m_monthly_0.5deg.nc"
netcdf_variable = "t2m"

[masks.landcover_snow_ice]
local_file = "landcover_snow_ice_0p5deg.npy"
source_local_file = "landcover_mask_0p5_7classes.npy"
derived_from_upstream_entity_id = "https://w3id.org/ro-id/.../landcover_mask_0p5_7classes.npy"
derivation_type = "class_selection"
class_value = 100
class_label = "Snow/Ice"
```

La resolución usa este orden: ID explícito, checksum, nombre/basename/alias. Un
ID explícito inexistente o cualquier variable/máscara requerida sin entidad
provoca `UpstreamResolutionError`; no se genera un crate parcial silencioso.
Los checksums local y upstream y su estado (`verified`, `mismatch` o
`not_available`) se registran en `input_resolution`.

## Contenido y trazabilidad

El módulo captura únicamente información disponible durante la ejecución:

- repositorio, hash completo y URL permanente del commit, rama y URL de rama,
  además de `git_dirty`; el commit es el identificador principal;
- si `git_dirty=true`, incluye y enlaza `code/git-diff.patch`; el commit más el
  parche describen el estado realmente ejecutado;
- comando, inicio y fin UTC, Python, plataforma y distribuciones instaladas;
- copias saneadas de los TOML (claves de contraseña/token/secreto redactadas);
- variables, target, máscaras, semilla, estrategia y metadatos de partición;
- parámetros, modelo/scaler, métricas, muestra de predicciones, figuras, logs y
  artefactos de explicabilidad presentes;
- identificador/experimento/URI de MLflow cuando el entrenamiento obtuvo un
  `run_id`;
- descriptor Croissant del dataset concreto y descriptor FAIR4ML del modelo y
  su evaluación.

Los NetCDF de entrada no se copian. Cada fichero resuelto conserva su `@id`,
`contentUrl`/`dcat:downloadURL`, checksum y `variableMeasured`. Todas las
entradas se validan en `CreateAction.object`, `prov:used`,
`#ml-dataset.prov:wasDerivedFrom` y Croissant.

`landcover_snow_ice` se materializa como una máscara local derivada: la entidad
binaria apunta con `prov:wasDerivedFrom` a la máscara categórica upstream y una
`CreateAction` documenta la selección de la clase `100 (Snow/Ice)` y el TOML
que define la transformación. Antes de escribir el crate se comprueba que la
máscara binaria coincide exactamente con `máscara_categórica == 100`.

### Datos intermedios y Croissant

El entrenamiento transforma primero los NetCDF en arrays NPY por variable y
después genera `X_train.npy`, `y_train.npy`, `X_test.npy`, `y_test.npy`, sus
índices y máscaras. En modo Croissant `descriptive` estos arrays **no se
copian**. El crate conserva únicamente los JSON pequeños `metadata.json`,
`run_config.json`, `dataset_metadata.json`, `split_metadata.json` y
`train_info.json`; las entidades de los NPY registran checksum, tamaño y
procedencia, pero se marcan como intermedios locales no descargables.

El descriptor distingue `<urn-de-ejecución>#ml-dataset`,
`<urn-de-ejecución>#training-dataset` y `<urn-de-ejecución>#test-dataset`, y
reutiliza exactamente esos IDs absolutos en RO-Crate, Croissant y FAIR4ML.
No declara un conjunto de validación fijo: cuando se activa CV, se documentan
la estrategia y los folds internos sobre train. En modo `materialized`, los
NPY se convierten a `data/ml/train.parquet` y `data/ml/test.parquet`; Croissant
añade entonces `RecordSet`, `Field`, `source`, `split` y `sample_id` (cuando
están disponibles los índices espaciales/temporales) sin incrustar filas en el
JSON-LD.

Con `include_split_manifest = true`, `fair/split_manifest.npz` contiene
`train_indices` y `test_indices`. Cada índice sigue el orden C global
`(time_idx * latitude_size + lat_idx) * longitude_size + lon_idx`; el JSON
adjunto registra dimensiones, algoritmo, semilla, tamaños, variables, commit y
SHA-256. El manifiesto identifica exactamente la pertenencia a train/test sin
copiar `X_*`, `y_*` ni los NPY de variables.

El JSON incluye los tamaños y orden de las dimensiones, extremos, paso y orden
de latitud/longitud, valores temporales, conteos total/válido/seleccionado,
fracciones solicitadas y versiones de NumPy y scikit-learn. Puede decodificarse
sin consultar los NPY originales:

```python
import json
import numpy as np
from src.reproducibility.run_metadata import decode_observation_indices

metadata = json.load(open("fair/split_manifest.json"))
with np.load("fair/split_manifest.npz") as split:
    coordinates = decode_observation_indices(split["test_indices"], metadata)
```

Croissant registra directamente ambos archivos como `cr:FileObject`. Los
recursos upstream con URL directa se enumeran en
`croissant_downloadable_resources`; los que solo tienen identidad semántica se
conservan en `croissant_semantic_only_resources` sin inventar una URL.

### Estado del código

- `reference` (predeterminado): solo registra repositorio, rama, commit, URLs y
  estado dirty; no copia el árbol fuente. Si está dirty se emite una advertencia.
- `patch`: si el árbol está modificado, incluye `code/git-diff.patch` y los
  ficheros no versionados relevantes bajo `code/untracked/`.
- `snapshot`: incluye bajo `code/snapshot/` todos los ficheros versionados por
  Git.

Con `require_clean_repository = true`, cualquier cambio versionado o no
versionado impide generar el crate. Para una release definitiva se recomienda
`mode = "reference"` junto con esta comprobación.

Los TOML incluidos en patch o snapshot se saneean igual que las copias de
configuración. El fichero `checksums.sha256` permite verificar el contenido
físico del crate, y cada entidad `File` local registra además tamaño y SHA-256.

### Publicación del modelo

```toml
[reproducibility.model_artifact]
mode = "external"
download_url = "https://archive.example/models/model.joblib"
landing_page = "https://archive.example/records/123"
identifier = "https://doi.org/10.example/model"
```

- `attached` (predeterminado): copia el modelo al crate y registra SHA-256 y
  tamaño.
- `external`: calcula la metadata desde el modelo generado, no lo copia y usa
  `download_url` como `@id`; la URL debe ser HTTP(S) absoluta.
- `metadata_only`: no copia ni publica el artefacto y marca el modelo como no
  reutilizable.

Las licencias se configuran por separado para metadatos, código, modelo,
resultados y dataset derivado. CC0 se aplica por defecto únicamente a los
metadatos. Una licencia de metadatos nunca cambia la licencia de los recursos
upstream. Si no se establece `derived_dataset`, el crate lo declara
explícitamente en `conditionsOfAccess` y no atribuye CC0 al dataset ML.

`environment/python-packages.json` siempre se conserva. Un `lock_file`
configurado se copia bajo `environment/lock/`. Una imagen de contenedor solo se
acepta junto con un digest inmutable; opcionalmente pueden registrarse también
plataforma y runtime.

## Árbol de salida

Cada activación crea un identificador de ejecución con fecha y UUID corto:

```text
outputs/ro-crates/
└── rf_land_monthly/
    └── 20260720T143000Z-a1b2c3d4/
        └── ro-crate/
            ├── ro-crate-metadata.json
            ├── README.md
            ├── checksums.sha256
            ├── code/                 # solo patch o snapshot
            ├── config/
            │   ├── train.toml
            │   └── data.toml
            ├── environment/
            │   └── python-packages.json
            ├── model/
            │   ├── model.joblib
            │   └── train_info.json
            ├── metrics/
            ├── predictions/
            ├── figures/
            ├── logs/
            ├── metadata/
            │   ├── processed_metadata.json
            │   ├── processed_run_config.json
            │   ├── dataset_metadata.json
            │   ├── split_metadata.json
            │   └── train_info.json
            └── fair/
                ├── dataset.croissant.json
                ├── model.fair4ml.json
                ├── split_manifest.json
                └── split_manifest.npz
```

## Perfiles utilizados

- [RO-Crate 1.1](https://www.researchobject.org/ro-crate/1.1/)
- [Process Run Crate 0.5](https://www.researchobject.org/workflow-run-crate/profiles/process_run_crate/),
  el perfil de Workflow Run RO-Crate para la ejecución de un proceso/script
  (el pipeline no se presenta como un workflow formal orquestado por un WMS)
- [Croissant 1.1](https://docs.mlcommons.org/croissant/docs/croissant-spec-1.1.html)
- [FAIR4ML 0.1.0](https://rda-fair4ml.github.io/FAIR4ML-schema/release/0.1.0/)

La validación integrada comprueba la estructura y el parseo RDF de los tres
JSON-LD, IDs compartidos, contenido y rango del split, reconstrucción de
coordenadas, checksums, rutas Croissant, URLs remotas y cobertura del manifiesto
`checksums.sha256`. No sustituye una validación SHACL externa.

## RO-Crate ejecutable

Para que el paquete sea reproducible, y no únicamente descriptivo, active el
modo estricto y describa una release upstream inmutable:

```toml
[reproducibility]
create_rocrate = true
reproducible_run = true

[reproducibility.upstream]
identifier = "https://doi.org/10.xxxx/release"
landing_page = "https://archive.example/records/123"
metadata_url = "https://archive.example/records/123/files/ro-crate-metadata.json"
cache_dir = "./cache/upstream"
download_missing = true
verify_checksums = true
offline = false

[reproducibility.inputs]
require_upstream_checksum = true
allow_derived_local_inputs = true

[reproducibility.environment]
lock_file = "uv.lock"
python_version = "3.12"

[reproducibility.verification]
fail_on_metric_difference = true
metric_absolute_tolerance = 0.0001
```

En este modo toda entrada requerida necesita un `upstream_entity_id` explícito.
Un checksum local distinto solo se admite si el catálogo declara su derivación.
El crate conserva el checksum del descriptor upstream, la resolución efectiva
de cada input, la configuración efectiva saneada y los resultados esperados:

```text
config/effective_config.json
reproducibility/input_resolution.json
reproducibility/expected_results.json
reproduce.sh
```

### Ejecución

Tras descomprimir el crate:

```bash
./reproduce.sh --work-dir ./reproduction

python -m profecia.reproducibility.verify \
  --original-crate . \
  --reproduced-run ./reproduction
```

También están disponibles `--offline`, `--verify-only`, `--container`,
`--local-environment` y `--recompute-split`. El modo offline nunca accede a la
red: exige que el descriptor y todos los recursos estén ya en caché. El modo
`--recompute-split` queda registrado como reproducción no exacta del split.

La prioridad del entorno es una imagen OCI identificada por digest, un lock
incluido en `environment/lock/`, y finalmente el inventario exacto de paquetes.
Un tag OCI sin digest se rechaza. `reproduce.sh` no contiene el código: instala
o valida el commit Git permanente registrado por el crate.

El resultado se escribe sin modificar el original:

```text
reproduction/
├── cache/upstream/
├── inputs/
├── derived-inputs/
├── outputs/
├── input_resolution.json
├── reproduction_report.json
├── reproduction_report.md
└── reproduced-ro-crate/
    ├── ro-crate-metadata.json
    ├── reproduction_report.json
    └── reproduction_report.md
```

El nuevo crate enlaza tanto la ejecución original como la release upstream y
contiene una entidad `#reproduction-assessment` con los controles aprobados,
fallidos, diferencias métricas y nivel de reproducción. Los fallos de descarga,
checksum, entorno o split son fatales; las diferencias de plataforma se
clasifican como condición no reproducible; una recalculación solicitada del
split se registra como advertencia.
