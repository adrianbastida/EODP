import logging
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from config.globalConfig import globalConfig
from config.l1bConfig import l1bConfig
from l1b.src.l1b import l1b


class ProvesFiguresEqualitzacio(unittest.TestCase):
    def test_comparativa_amb_quatre_bandes_desa_figures_i_utilitza_alt_50(self):
        senyal_dn = np.arange(60 * 4, dtype=float).reshape(60, 4)
        referencia = senyal_dn * 0.5
        processador = l1b.__new__(l1b)
        processador.logger = logging.getLogger(__name__)
        processador.l1bConfig = l1bConfig()
        processador.globalConfig = globalConfig()
        processador.auxdir = "auxiliary"
        processador.indir = "input"
        with TemporaryDirectory() as directori:
            processador.outdir = directori
            for indice_banda in range(4):
                banda = f"VNIR-{indice_banda}"
                with self.subTest(banda=banda), patch(
                    "l1b.src.l1b.os.path.isfile", return_value=True
                ), patch(
                    "l1b.src.l1b.readFactor", side_effect=[np.full(4, 2.0), np.ones(4)]
                ), patch(
                    "l1b.src.l1b.readToa", return_value=referencia
                ), patch("l1b.src.l1b.plt.close"):
                    processador.plotEqualizationModel(senyal_dn, banda)
                    figura = plt.gcf()
                    linies = figura.axes[0].lines
                    self.assertEqual([linia.get_color() for linia in linies], ["black", "red", "blue"])
                    guany = processador.l1bConfig.gain[indice_banda]
                    np.testing.assert_allclose(linies[0].get_ydata(), (senyal_dn[50] - 1) / 2 * guany)
                    np.testing.assert_allclose(linies[1].get_ydata(), senyal_dn[50] * guany)
                    np.testing.assert_allclose(linies[2].get_ydata(), referencia[50])
                plt.close(figura)
                ruta = Path(directori) / f"figure_{indice_banda + 1}_equalization_{banda}.png"
                self.assertTrue(ruta.is_file(), "Each figure should be saved without overwriting the others")

    def test_processament_amb_quatre_bandes_genera_totes_les_comparatives(self):
        processador = l1b.__new__(l1b)
        processador.logger = logging.getLogger(__name__)
        processador.l1bConfig = l1bConfig()
        processador.globalConfig = globalConfig()
        processador.auxdir = "auxiliary"
        processador.indir = "input"
        processador.outdir = "output"
        with patch("l1b.src.l1b.readToa", return_value=np.ones((60, 4))), patch(
            "l1b.src.l1b.readQualityMask", return_value=np.zeros((60, 4), dtype=np.uint8)
        ), patch("l1b.src.l1b.readFactor", return_value=np.ones(4)), patch(
            "l1b.src.l1b.writeToa"
        ), patch.object(processador, "plotL1bToa"), patch.object(
            processador, "plotEqualizationModel"
        ) as comparativa:
            processador.processModule()
        self.assertEqual(
            [crida.args[1] for crida in comparativa.call_args_list],
            list(processador.globalConfig.bands),
            "The comparison should be generated for every band",
        )

    def test_comparativa_sense_linia_50_no_genera_figura(self):
        processador = l1b.__new__(l1b)
        processador.logger = logging.getLogger(__name__)
        with self.assertLogs(processador.logger, level="WARNING"), patch(
            "l1b.src.l1b.plt.subplots"
        ) as grafica:
            processador.plotEqualizationModel(np.ones((50, 4)), "VNIR-1")
        grafica.assert_not_called()