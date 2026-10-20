
# LEVEL-1C

from l1c.src.initL1c import initL1c
from common.io.writeToa import readToa
from common.io.readGeodetic import readGeodetic
import mgrs
import numpy as np
from scipy.interpolate import bisplev, bisplrep
import matplotlib.pyplot as plt
import os
from common.io.l1cProduct import writeL1c

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
        mascara_geografica = np.isfinite(lat) & np.isfinite(lon)
        mascara_valida = mascara_geografica & np.isfinite(toa)
        if np.count_nonzero(mascara_valida) < 16:
            raise ValueError("Se necesitan al menos 16 píxeles geolocalizados y radiancias válidas para ajustar el spline L1C")
        m = mgrs.MGRS()
        indices_validos = np.flatnonzero(mascara_geografica)
        indice_representativo = indices_validos[len(indices_validos) // 2]
        fila_representativa, columna_representativa = np.unravel_index(indice_representativo, lat.shape)
        codigo_mgrs = m.toMGRS(float(lat[fila_representativa, columna_representativa]), float(lon[fila_representativa, columna_representativa]), MGRSPrecision=0)
        zona_utm = int(codigo_mgrs[:2])
        zonas_entrada = {int(m.toMGRS(float(lat[ii, jj]), float(lon[ii, jj]), MGRSPrecision=0)[:2]) for ii, jj in zip(*np.nonzero(mascara_geografica))}
        if len(zonas_entrada) != 1:
            raise ValueError("La escena cruza varias zonas UTM; debe dividirse por zona antes de generar el producto MGRS")
        if self.l1cConfig.mgrs_tile_precision not in range(0, 6):
            raise ValueError("La precisión MGRS debe estar entre 0 y 5")
        tiles_mgrs = set()
        for fila, columna in zip(*np.nonzero(mascara_geografica)):
            tile = m.toMGRS(
                float(lat[fila, columna]),
                float(lon[fila, columna]),
                MGRSPrecision=self.l1cConfig.mgrs_tile_precision,
            )
            tiles_mgrs.add(tile.decode() if isinstance(tile, bytes) else tile)
        tiles_ordenades = sorted(tiles_mgrs)
        coordenades_l1c = [m.toLatLon(tile) for tile in tiles_ordenades]
        lat_l1c = np.asarray([coordenada[0] for coordenada in coordenades_l1c])
        lon_l1c = np.asarray([coordenada[1] for coordenada in coordenades_l1c])

        mascara_valida_f = mascara_valida.ravel(order="F")
        latitud_entrada = lat.ravel(order="F")[mascara_valida_f]
        longitud_entrada = lon.ravel(order="F")[mascara_valida_f]
        radiancia_entrada = toa.ravel(order="F")[mascara_valida_f]
        try:
            spline, _, estat_fit, missatge_fit = bisplrep(
                latitud_entrada,
                longitud_entrada,
                radiancia_entrada,
                kx=3,
                ky=3,
                s=float(radiancia_entrada.size),
                full_output=True,
                quiet=1,
            )
        except (ValueError, TypeError, RuntimeError) as error:
            raise ValueError(f"Could not fit the L1C bivariate spline for {band}: {error}") from error
        if estat_fit > 0:
            raise ValueError(f"Could not fit the L1C bivariate spline for {band}: {missatge_fit}")

        toa_l1c = np.asarray([
            float(np.asarray(bisplev(float(latitud), float(longitud), spline)).reshape(-1)[0])
            for latitud, longitud in zip(lat_l1c, lon_l1c)
        ])
        return lat_l1c, lon_l1c, toa_l1c

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
        eje.scatter(lon_l1b[mascara_l1b], lat_l1b[mascara_l1b], s=6, color="red", marker="o", linewidths=0, alpha=0.82, label="L1B", zorder=2)
        eje.scatter(lon_l1c[mascara_l1c], lat_l1c[mascara_l1c], s=6, color="blue", marker="o", linewidths=0, alpha=0.82, label="L1C MGRS", zorder=3)
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
