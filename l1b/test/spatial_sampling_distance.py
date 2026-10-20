import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from pyproj import Geod


def calcular_distancia_espacial(latitud, longitud, indice_linea):
    latitud = np.asarray(latitud, dtype=float)
    longitud = np.asarray(longitud, dtype=float)
    if latitud.ndim != 2 or longitud.ndim != 2 or latitud.shape != longitud.shape:
        raise ValueError("Latitude and longitude must be 2D arrays with the same shape")
    if not isinstance(indice_linea, (int, np.integer)) or not 0 <= indice_linea < latitud.shape[0]:
        raise ValueError("The line index must be within the geolocation dimensions")
    if latitud.shape[1] < 2:
        raise ValueError("The line must contain at least two pixels")

    latitud_linia = latitud[indice_linea]
    longitud_linia = longitud[indice_linea]
    if not np.all(np.isfinite(latitud_linia)) or not np.all(np.isfinite(longitud_linia)):
        raise ValueError("The selected line contains non-finite coordinates")
    if np.any(np.abs(latitud_linia) > 90):
        raise ValueError("Latitudes must be between -90 and 90 degrees")

    geodesia = Geod(ellps="WGS84")
    _, _, distancia = geodesia.inv(
        longitud_linia[:-1],
        latitud_linia[:-1],
        longitud_linia[1:],
        latitud_linia[1:],
    )
    return distancia


def generar_grafica_distancia_espacial(ruta_geolocalitzacio, indice_linea, ruta_sortida):
    with Dataset(ruta_geolocalitzacio, "r") as conjunt_dades:
        projeccio = conjunt_dades.groups["projection"]
        latitud = np.ma.asarray(projeccio.variables["latitude"][:]).filled(np.nan)
        longitud = np.ma.asarray(projeccio.variables["longitude"][:]).filled(np.nan)

    distancies = calcular_distancia_espacial(latitud, longitud, indice_linea)
    ruta_sortida = Path(ruta_sortida)
    ruta_sortida.parent.mkdir(parents=True, exist_ok=True)

    figura, eix = plt.subplots(figsize=(10, 6))
    eix.plot(np.arange(distancies.size), distancies)
    eix.set_title(f"L1B Spatial Sampling Distance - Line {indice_linea}")
    eix.set_xlabel("Pixel")
    eix.set_ylabel("Spatial sampling distance [m]")
    eix.ticklabel_format(axis="y", style="plain", useOffset=False)
    eix.grid(True)
    figura.tight_layout()
    figura.savefig(ruta_sortida, dpi=300)
    plt.close(figura)
    return ruta_sortida


def main():
    directori_eodp = Path(__file__).resolve().parents[3]
    directori_dades = directori_eodp / "EODP_TER_2021"
    ruta_geolocalitzacio_defecte = (
        directori_dades
        / "EODP-TS-L1C"
        / "input"
        / "gm_alt100_act_150"
        / "geolocation.nc"
    )
    ruta_sortida_defecte = (
        directori_dades
        / "EODP-TS-L1B"
        / "myoutput"
        / "L1B_Spatial_Sampling_Distance_Line_50.png"
    )

    parser = argparse.ArgumentParser(
        description="Generate the spatial sampling distance plot for a GEO line."
    )
    parser.add_argument(
        "--geolocalitzacio",
        type=Path,
        default=ruta_geolocalitzacio_defecte,
        help="Geolocation NetCDF file.",
    )
    parser.add_argument(
        "--linea",
        type=int,
        default=50,
        help="Zero-based line index (default: 50).",
    )
    parser.add_argument(
        "--sortida",
        type=Path,
        default=ruta_sortida_defecte,
        help="Output PNG path.",
    )
    arguments = parser.parse_args()

    ruta_generada = generar_grafica_distancia_espacial(
        arguments.geolocalitzacio,
        arguments.linea,
        arguments.sortida,
    )
    print(f"Plot saved to: {ruta_generada}")


if __name__ == "__main__":
    main()
