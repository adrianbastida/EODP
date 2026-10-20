import unittest
import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import numpy as np
from pyproj import Transformer
from scipy.interpolate import bisplev, bisplrep

from config.l1cConfig import l1cConfig
from l1c.src.l1c import l1c


class ProvesRejillaL1C(unittest.TestCase):
    def test_comparativa_amb_malles_valides_dibuixa_punts_vermells_i_blaus(self):
        latitud = np.array([[40.0, 40.0], [40.01, 40.01]])
        longitud = np.array([[3.0, 3.01], [3.0, 3.01]])
        processador = l1c.__new__(l1c)
        processador.logger = logging.getLogger(__name__)
        with TemporaryDirectory() as directori:
            processador.outdir = directori
            with patch("l1c.src.l1c.plt.close"):
                try:
                    processador.plotGridComparison(
                        latitud, longitud, latitud.ravel(), longitud.ravel(), np.ones(4)
                    )
                    eix = plt.gcf().axes[0]
                    self.assertEqual(len(eix.lines), 0, "No continuous outlines should be drawn")
                    self.assertEqual(len(eix.collections), 2)
                    for punts, color in zip(eix.collections, ("red", "blue")):
                        np.testing.assert_allclose(
                            punts.get_facecolors()[0], to_rgba(color, alpha=0.82)
                        )
                        np.testing.assert_allclose(
                            punts.get_offsets(), np.column_stack((longitud.ravel(), latitud.ravel()))
                        )
                    self.assertTrue(
                        (Path(directori) / "figure_9-2_l1b_irregular_l1c_regular_grids.png").is_file(),
                        "The comparison PNG should be generated",
                    )
                finally:
                    figura = plt.gcf()
            plt.close(figura)

    def test_l1cProjtoa_amb_malla_utm_alinea_sortida_amb_origen_mgrs(self):
        transformador_utm = Transformer.from_crs(32631, 4326, always_xy=True)
        x_entrada, y_entrada = np.meshgrid(
            500025.0 + np.arange(8) * 80.0,
            4600025.0 + np.arange(8) * 80.0,
        )
        longitud, latitud = transformador_utm.transform(x_entrada, y_entrada)
        radiancia = x_entrada + 2 * y_entrada
        processador = l1c.__new__(l1c)
        processador.l1cConfig = l1cConfig()

        latitud_sortida, longitud_sortida, radiancia_sortida = processador.l1cProjtoa(
            latitud,
            longitud,
            radiancia,
            "VNIR-0",
        )

        self.assertEqual(latitud_sortida.ndim, 1)
        self.assertEqual(longitud_sortida.shape, latitud_sortida.shape)
        self.assertEqual(radiancia_sortida.shape, latitud_sortida.shape)

        x_sortida, y_sortida = Transformer.from_crs(
            4326, 32631, always_xy=True
        ).transform(longitud_sortida, latitud_sortida)
        pas = 100000 / 10**processador.l1cConfig.mgrs_tile_precision
        np.testing.assert_allclose(
            x_sortida / pas,
            np.rint(x_sortida / pas),
            atol=2e-4,
        )
        np.testing.assert_allclose(
            y_sortida / pas,
            np.rint(y_sortida / pas),
            atol=2e-4,
        )
        spline = bisplrep(
            latitud.ravel(order="F"),
            longitud.ravel(order="F"),
            radiancia.ravel(order="F"),
            kx=3,
            ky=3,
            s=float(radiancia.size),
        )
        radiancia_esperada = np.asarray([
            float(np.asarray(bisplev(float(lat), float(lon), spline)).reshape(-1)[0])
            for lat, lon in zip(latitud_sortida, longitud_sortida)
        ])
        np.testing.assert_allclose(
            radiancia_sortida,
            radiancia_esperada,
            rtol=1e-12,
            atol=1e-12,
        )
