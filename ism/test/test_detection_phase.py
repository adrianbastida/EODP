import tempfile
import unittest
from pathlib import Path

import numpy as np

from ism.src.detectionPhase import detectionPhase


class ProvesFaseDeteccio(unittest.TestCase):
    """Proves de la reproduïbilitat de la fase de detecció ISM."""

    def test_compute_amb_llavor_de_referencia_reprodueix_prnu(self):
        directori_auxiliar = Path(__file__).resolve().parents[2] / "auxiliary"
        irradiancia = np.ones((2, 150), dtype=float)

        with tempfile.TemporaryDirectory() as directori_temporal:
            entrada = Path(directori_temporal) / "entrada"
            sortida = Path(directori_temporal) / "sortida"
            entrada.mkdir()
            sortida.mkdir()
            fase = detectionPhase(str(directori_auxiliar), str(entrada), str(sortida))
            fase.ismConfig.apply_dark_signal = False
            fase.ismConfig.apply_bad_dead = False
            fase.ismConfig.save_after_ph2e = False
            fase.ismConfig.save_after_prnu = False
            fase.ismConfig.save_detection_stage = False

            area_pixel = fase.ismConfig.pix_size**2
            fotons = fase.irrad2Phot(
                irradiancia,
                area_pixel,
                fase.ismConfig.t_int,
                fase.ismConfig.wv[0],
            )
            electrons = fase.phot2Electr(fotons, fase.ismConfig.QE)
            prnu = np.random.RandomState(fase.ismConfig.seed).standard_normal(150)
            esperat = electrons * (1 + prnu[np.newaxis, :] * fase.ismConfig.kprnu)

            obtingut = fase.compute(irradiancia, "VNIR-0")

        np.testing.assert_allclose(obtingut, esperat, rtol=1e-12, atol=1e-12)
