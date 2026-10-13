
# LEVEL-1C

from l1c.src.initL1c import initL1c
from common.io.writeToa import writeToa, readToa
from common.io.readGeodetic import readGeodetic, getCorners
import mgrs
import numpy as np
from scipy.interpolate import LinearNDInterpolator
from pyproj import CRS, Transformer
from matplotlib.path import Path as PolygonPath
import matplotlib.pyplot as plt
import os
from common.io.l1cProduct import writeL1c
from matplotlib import cm

class l1c(initL1c):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)

    def processModule(self):

        self.logger.info("Start of the L1C Processing Module")

        for band in self.globalConfig.bands:

            self.logger.info("Start of BAND " + band)

            # Read TOA - output of the L1B in Radiances
            # -------------------------------------------------------------------------------
            toa = readToa(self.l1bdir, self.globalConfig.l1b_toa + band + '.nc')
            lat,lon = readGeodetic(self.gmdir, self.globalConfig.gm_geoloc)
            self.checkSize(lat, toa, lon)

            # L1C reprojection onto the MGRS grid
            # -------------------------------------------------------------------------------
            lat_l1c, lon_l1c, toa_l1c = self.l1cProjtoa(lat, lon, toa, band)

            # Write output TOA
            # -------------------------------------------------------------------------------
            writeL1c(self.outdir, self.globalConfig.l1c_toa + band, lat_l1c, lon_l1c, toa_l1c)
            if self.l1cConfig.plotL1cGrid and band == self.globalConfig.bands[0]:
                self.plotGridComparison(lat, lon, lat_l1c, lon_l1c, toa_l1c)

            self.logger.info("End of BAND " + band)

        self.logger.info("End of the L1C Module!")


    def l1cProjtoa(self, lat, lon, toa, band):
        '''
        This function reprojects the L1B radiances into the MGRS grid.

        The MGRS reference system
        https://www.bluemarblegeo.com/knowledgebase/calculator-2020/Military_Grid_Reference_System_(MGRS).htm
        MGRS: '31REQ4367374067'
        31 is the UTM zone, R is the UTM latitude band; EQ are the MGRS column and row band letters
        43673 is the MGRS Easting (5 dig); 74067 is the MGRS Northing (5dig)

        Python mgrs library documentation
        https://pypi.org/project/mgrs/

        :param lat: L1B latitudes [deg]
        :param lon: L1B longitudes [deg]
        :param toa: L1B radiances
        :param band: band
        :return: L1C radiances, L1C latitude and longitude in degrees
        '''
        self.checkSize(lat, toa, lon)
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        toa = np.asarray(toa, dtype=float)
        mascara_valida = np.isfinite(lat) & np.isfinite(lon) & np.isfinite(toa)
        if np.count_nonzero(mascara_valida) < 3:
            raise ValueError("Se necesitan al menos tres píxeles geolocalizados y radiancias válidas para la reproyección")
        m = mgrs.MGRS()
        indices_validos = np.flatnonzero(mascara_valida)
        indice_representativo = indices_validos[len(indices_validos) // 2]
        fila_representativa, columna_representativa = np.unravel_index(indice_representativo, lat.shape)
        codigo_mgrs = m.toMGRS(float(lat[fila_representativa, columna_representativa]), float(lon[fila_representativa, columna_representativa]), MGRSPrecision=0)
        zona_utm = int(codigo_mgrs[:2])
        zonas_entrada = {int(m.toMGRS(float(lat[ii, jj]), float(lon[ii, jj]), MGRSPrecision=0)[:2]) for ii, jj in zip(*np.nonzero(mascara_valida))}
        if len(zonas_entrada) != 1:
            raise ValueError("La escena cruza varias zonas UTM; debe dividirse por zona antes de generar el producto MGRS")
        hemisferio_norte = float(np.nanmedian(lat[mascara_valida])) >= 0
        crs_geografico = CRS.from_epsg(4326)
        codigo_epsg = (32600 if hemisferio_norte else 32700) + zona_utm
        crs_utm = CRS.from_epsg(codigo_epsg)
        a_utm, desde_utm = Transformer.from_crs(crs_geografico, crs_utm, always_xy=True), Transformer.from_crs(crs_utm, crs_geografico, always_xy=True)
        x_entrada, y_entrada = a_utm.transform(lon[mascara_valida], lat[mascara_valida])
        if self.l1cConfig.mgrs_tile_precision not in range(0, 6):
            raise ValueError("La precisión MGRS debe estar entre 0 y 5")
        paso = 100000.0 / (10 ** self.l1cConfig.mgrs_tile_precision)
        x_min, x_max = float(np.min(x_entrada)), float(np.max(x_entrada))
        y_min, y_max = float(np.min(y_entrada)), float(np.max(y_entrada))
        x_centros = np.arange(np.floor(x_min / paso) * paso + paso / 2, np.floor(x_max / paso) * paso + paso / 2 + paso / 2, paso)
        y_centros = np.arange(np.floor(y_min / paso) * paso + paso / 2, np.floor(y_max / paso) * paso + paso / 2 + paso / 2, paso)
        if x_centros.size == 0 or y_centros.size == 0:
            raise ValueError("La extensión geográfica no contiene centros de celdas MGRS")
        x_salida, y_salida = np.meshgrid(x_centros, y_centros, indexing="xy")
        puntos = np.column_stack((x_entrada, y_entrada))
        interpolador = LinearNDInterpolator(puntos, toa[mascara_valida], fill_value=np.nan)
        toa_l1c = np.asarray(interpolador(x_salida, y_salida), dtype=float)
        borde_latitud = np.concatenate((lat[0, :], lat[1:, -1], lat[-1, -2::-1], lat[-2:0:-1, 0]))
        borde_longitud = np.concatenate((lon[0, :], lon[1:, -1], lon[-1, -2::-1], lon[-2:0:-1, 0]))
        borde_valido = np.isfinite(borde_latitud) & np.isfinite(borde_longitud)
        if np.count_nonzero(borde_valido) < 4:
            raise ValueError("No hay suficientes coordenadas válidas en el perímetro L1B para limitar la huella")
        borde_x, borde_y = a_utm.transform(borde_longitud[borde_valido], borde_latitud[borde_valido])
        poligono_huella = PolygonPath(np.column_stack((borde_x, borde_y)), closed=True)
        centros_salida = np.column_stack((x_salida.ravel(), y_salida.ravel()))
        dentro_huella = poligono_huella.contains_points(centros_salida).reshape(toa_l1c.shape)
        toa_l1c[~dentro_huella] = np.nan
        lon_l1c, lat_l1c = desde_utm.transform(x_salida, y_salida)
        return np.asarray(lat_l1c), np.asarray(lon_l1c), toa_l1c

    def plotGridComparison(self, lat_l1b, lon_l1b, lat_l1c, lon_l1c, toa_l1c):
        lat_l1b = np.asarray(lat_l1b, dtype=float)
        lon_l1b = np.asarray(lon_l1b, dtype=float)
        lat_l1c = np.asarray(lat_l1c, dtype=float)
        lon_l1c = np.asarray(lon_l1c, dtype=float)
        mascara_l1b = np.isfinite(lat_l1b) & np.isfinite(lon_l1b)
        mascara_l1c = np.isfinite(lat_l1c) & np.isfinite(lon_l1c) & np.isfinite(toa_l1c)
        if not mascara_l1b.any() or not mascara_l1c.any():
            raise ValueError("No hay coordenadas válidas suficientes para representar las mallas L1B y L1C")
        figura, eje = plt.subplots(figsize=(11, 8), constrained_layout=True)
        borde_l1b = (
            (lon_l1b[0, :], lat_l1b[0, :], mascara_l1b[0, :]),
            (lon_l1b[-1, :], lat_l1b[-1, :], mascara_l1b[-1, :]),
            (lon_l1b[:, 0], lat_l1b[:, 0], mascara_l1b[:, 0]),
            (lon_l1b[:, -1], lat_l1b[:, -1], mascara_l1b[:, -1]),
        )
        for indice, (longitudes, latitudes, validos) in enumerate(borde_l1b):
            eje.plot(np.where(validos, longitudes, np.nan), np.where(validos, latitudes, np.nan), color="#222222", linewidth=1.2, label="L1B" if indice == 0 else None, zorder=2)
        eje.scatter(lon_l1c[mascara_l1c], lat_l1c[mascara_l1c], s=6, color="#C51B29", marker="o", linewidths=0, alpha=0.82, label="L1C MGRS", zorder=3)
        eje.set_xlabel("Longitud [deg]")
        eje.set_ylabel("Latitud [deg]")
        eje.grid(True, linewidth=0.45, alpha=0.45)
        eje.legend(loc="best", framealpha=0.95)
        eje.set_title("Projection on ground")
        ruta_figura = os.path.join(self.outdir, "figure_9-2_l1b_irregular_l1c_regular_grids.png")
        figura.savefig(ruta_figura, dpi=220, bbox_inches="tight")
        plt.close(figura)
        self.logger.info("Figura 9-2 guardada en %s", ruta_figura)

    def checkSize(self, lat, toa, lon):
        '''
        Check the sizes of the input radiances and geodetic coordinates.
        If they don't match, exit.
        :param lat: Latitude 2D matrix
        :param toa: Radiance 2D matrix
        :return: NA
        '''
        formas = (np.shape(lat), np.shape(lon), np.shape(toa))
        if any(len(forma) != 2 for forma in formas):
            raise ValueError("Latitud, longitud y radiancia deben ser matrices 2D")
        if not (formas[0] == formas[1] == formas[2]):
            raise ValueError(f"Las dimensiones de latitud, longitud y radiancia no coinciden: {formas}")
