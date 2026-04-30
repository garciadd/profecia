#!/usr/bin/env python3
"""
Script para crear variables con lag temporal (-1, -2, -3 meses).

Para cada variable en /mnt_sentinel_a/ferag/data/raw, crea tres versiones
con desplazamiento temporal, manteniendo las dimensiones originales.
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import xarray as xr
import logging

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
LOGGER = logging.getLogger(__name__)

RAW_DIR = Path("/mnt_sentinel_a/ferag/data/raw")
OUTPUT_SUFFIX = "_lagged"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Crea versiones lagged de archivos NetCDF mensuales. "
            "Si no se indican ficheros, procesa todos los *_1982_2022_monthly_0.5deg.nc "
            f"de {RAW_DIR}."
        )
    )
    parser.add_argument(
        "files",
        nargs="*",
        help=(
            "Lista de ficheros a procesar. Puede ser nombre de archivo dentro de RAW_DIR "
            "o una ruta completa."
        ),
    )
    parser.add_argument(
        "--lags",
        nargs="+",
        type=int,
        default=[1, 2, 3],
        help="Lista de lags en meses a generar. Por defecto: 1 2 3",
    )
    return parser.parse_args()


def resolve_input_files(file_args: list[str]) -> list[Path]:
    if not file_args:
        return sorted(RAW_DIR.glob("*_1982_2022_monthly_0.5deg.nc"))

    resolved_files: list[Path] = []
    missing_files: list[str] = []

    for file_arg in file_args:
        candidate = Path(file_arg)
        if not candidate.is_absolute():
            candidate = RAW_DIR / candidate

        if candidate.exists() and candidate.is_file():
            resolved_files.append(candidate)
        else:
            missing_files.append(file_arg)

    if missing_files:
        missing_str = ", ".join(missing_files)
        raise FileNotFoundError(f"No se encontraron estos ficheros: {missing_str}")

    return resolved_files


def create_lagged_variables(nc_file: Path, lags: list[int] = [1, 2, 3]) -> dict[int, Path]:
    """
    Crea versiones con lag temporal de un archivo NetCDF.
    
    Parameters
    ----------
    nc_file : Path
        Ruta del archivo NetCDF original
    lags : list[int]
        Lista de lags en meses a crear (default: [1, 2, 3])
    
    Returns
    -------
    dict[int, Path]
        Mapeo de lag -> ruta del archivo creado
    """
    LOGGER.info(f"Procesando: {nc_file.name}")
    
    # Cargar dataset
    try:
        ds = xr.open_dataset(nc_file)
    except Exception as e:
        LOGGER.error(f"Error al cargar {nc_file}: {e}")
        return {}
    
    # Identificar dimensión temporal
    time_dim = None
    if 'time' in ds.dims:
        time_dim = 'time'
    else:
        # Buscar otra dimensión temporal
        for dim in ds.dims:
            if 'time' in dim.lower() or dim in ['t', 'date']:
                time_dim = dim
                break
    
    if time_dim is None:
        LOGGER.warning(f"No se encontró dimensión temporal en {nc_file.name}")
        ds.close()
        return {}
    
    n_times = ds.dims[time_dim]
    LOGGER.info(f"  Dimensión temporal: {time_dim} | Timesteps: {n_times}")
    
    results = {}
    
    for lag in lags:
        LOGGER.info(f"  Creando versión con lag={lag} mes(es)...")
        
        # Crear nuevo dataset
        ds_lagged = ds.copy(deep=True)
        
        # Para cada variable en el dataset
        for var_name in ds_lagged.data_vars:
            var_data = ds_lagged[var_name]
            
            # Crear array con datos desplazados
            # Desplazar: los primeros 'lag' timesteps tendrán NaN
            data_arr = var_data.values
            
            if time_dim in var_data.dims:
                # Encontrar la posición de time_dim
                time_axis = var_data.dims.index(time_dim)
                
                # Crear array con NaN al inicio
                lagged_arr = np.full_like(data_arr, np.nan, dtype=np.float32)
                
                # Copiar datos desplazados
                if time_axis == 0:
                    lagged_arr[lag:] = data_arr[:-lag]
                else:
                    # Usar slicing dinámico para otros ejes
                    slices_from = [slice(None)] * data_arr.ndim
                    slices_from[time_axis] = slice(None, -lag)
                    slices_to = [slice(None)] * data_arr.ndim
                    slices_to[time_axis] = slice(lag, None)
                    lagged_arr[tuple(slices_to)] = data_arr[tuple(slices_from)]
                
                ds_lagged[var_name].values = lagged_arr
        
        # Actualizar atributos de time si existen
        if time_dim in ds_lagged.coords:
            original_times = ds_lagged[time_dim].values
            ds_lagged[time_dim].attrs['note'] = f"Original shifted by {lag} months (first {lag} values are NaN)"
        
        # Crear nombre del archivo de salida
        stem = nc_file.stem
        # Extraer nombre base sin fechas
        parts = stem.split('_')
        var_name_parts = []
        for p in parts:
            if p.isdigit() or p == 'monthly' or p == '0.5deg':
                break
            var_name_parts.append(p)
        var_name_base = '_'.join(var_name_parts)
        
        output_file = nc_file.parent / f"{var_name_base}_lag{lag}_months_1982_2022_monthly_0.5deg.nc"
        
        # Guardar
        try:
            ds_lagged.to_netcdf(output_file, engine='netcdf4')
            LOGGER.info(f"    ✓ Guardado: {output_file.name}")
            results[lag] = output_file
        except Exception as e:
            LOGGER.error(f"    Error al guardar {output_file.name}: {e}")
        
        ds_lagged.close()
    
    ds.close()
    return results


def main():
    args = parse_args()

    if any(lag <= 0 for lag in args.lags):
        LOGGER.error("Todos los lags deben ser enteros positivos.")
        return 1

    try:
        nc_files = resolve_input_files(args.files)
    except FileNotFoundError as e:
        LOGGER.error(str(e))
        return 1
    
    if not nc_files:
        LOGGER.error(f"No se encontraron archivos NetCDF en {RAW_DIR}")
        return 1
    
    LOGGER.info(f"Encontrados {len(nc_files)} archivos NetCDF")
    LOGGER.info(f"Lags solicitados: {args.lags}")
    LOGGER.info("=" * 80)
    
    all_results = {}
    failed = []
    
    for nc_file in nc_files:
        try:
            results = create_lagged_variables(nc_file, lags=args.lags)
            if results:
                all_results[nc_file.name] = results
            else:
                failed.append(nc_file.name)
        except Exception as e:
            LOGGER.error(f"Error procesando {nc_file.name}: {e}")
            failed.append(nc_file.name)
    
    # Resumen
    LOGGER.info("=" * 80)
    LOGGER.info(f"RESUMEN:")
    LOGGER.info(f"  Archivos procesados: {len(all_results)}")
    LOGGER.info(f"  Archivos fallidos: {len(failed)}")
    
    if all_results:
        total_files_created = sum(len(v) for v in all_results.values())
        LOGGER.info(f"  Archivos totales creados: {total_files_created}")
        LOGGER.info(f"\nArchivos creados por variable:")
        for var, lags_dict in all_results.items():
            LOGGER.info(f"  {var}:")
            for lag, path in lags_dict.items():
                LOGGER.info(f"    - lag {lag}: {path.name}")
    
    if failed:
        LOGGER.warning(f"\nArchivos no procesados: {failed}")
    
    LOGGER.info("=" * 80)
    LOGGER.info("✓ Script completado")
    
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
