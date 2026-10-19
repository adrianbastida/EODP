import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from pyproj import Geod


def calcular_distancia_espacial(latitud, longitud, indice_linea):
    """Calcula la distància geodèsica entre píxels consecutius d'una línia.

    :param latitud: Matriu de latituds en graus.
    :param longitud: Matriu de longituds en graus.
    :param indice_linea: Índex de la línia que s'ha de calcular, començant per zero.
    :return: Vector de distàncies entre píxels consecutius, en metres.
    """
    latitud = np.asarray(latitud, dtype=float)
    longitud = np.asarray(longitud, dtype=float)
    if latitud.ndim != 2 or longitud.ndim != 2 or latitud.shape != longitud.shape:
        raise ValueError("La latitud i la longitud han de ser matrius 2D de la mateixa mida")
    if not isinstance(indice_linea, (int, np.integer)) or not 0 <= indice_linea < latitud.shape[0]:
        raise ValueError("L'índex de línia ha d'estar dins de les dimensions de la geolocalització")
    if latitud.shape[1] < 2:
        raise ValueError("La línia ha de contenir almenys dos píxels")

    latitud_linia = latitud[indice_linea]
    longitud_linia = longitud[indice_linea]
    if not np.all(np.isfinite(latitud_linia)) or not np.all(np.isfinite(longitud_linia)):
        raise ValueError("La línia seleccionada conté coordenades no finites")
    if np.any(np.abs(latitud_linia) > 90):
        raise ValueError("Les latituds han d'estar entre -90 i 90 graus")

    geodesia = Geod(ellps="WGS84")
    _, _, distancia = geodesia.inv(
        longitud_linia[:-1],
        latitud_linia[:-1],
        longitud_linia[1:],
        latitud_linia[1:],
    )
    return distancia


def generar_grafica_distancia_espacial(ruta_geolocalitzacio, indice_linea, ruta_sortida):
    """Genera i desa la gràfica de distància espacial per a una línia GEO.

    :param ruta_geolocalitzacio: Fitxer NetCDF amb latituds i longituds GEO.
    :param indice_linea: Índex de la línia que s'ha de representar, començant per zero.
    :param ruta_sortida: Ruta del fitxer PNG de sortida.
    :return: Ruta de la figura generada.
    """
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
    """Executa el càlcul i desa la gràfica de distància espacial."""
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
        description="Genera la gràfica de distància espacial d'una línia GEO."
    )
    parser.add_argument(
        "--geolocalitzacio",
        type=Path,
        default=ruta_geolocalitzacio_defecte,
        help="Fitxer NetCDF de geolocalització.",
    )
    parser.add_argument(
        "--linea",
        type=int,
        default=50,
        help="Índex de línia, començant per zero (per defecte: 50).",
    )
    parser.add_argument(
        "--sortida",
        type=Path,
        default=ruta_sortida_defecte,
        help="Ruta del PNG generat.",
    )
    arguments = parser.parse_args()

    ruta_generada = generar_grafica_distancia_espacial(
        arguments.geolocalitzacio,
        arguments.linea,
        arguments.sortida,
    )
    print(f"Gràfica desada a: {ruta_generada}")


if __name__ == "__main__":
    main()
