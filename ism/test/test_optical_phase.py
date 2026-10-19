import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

import numpy as np

from ism.src.opticalPhase import opticalPhase


class ProvesFaseOptica(unittest.TestCase):
    """Proves de la integració espectral de la fase òptica."""

    def test_spectralIntegration_amb_mostreig_no_uniform_aplica_rectangles_esquerres(self):
        longituds_ona = np.array([0.0, 0.5, 1.0, 2.0])
        cub_tao = np.array([[[1.0, 3.0, 5.0, 7.0]]])
        isrf = np.array([1.0, 2.0, 1.0])
        longituds_ona_isrf_um = np.array([0.0, 0.001, 0.002])
        directori_auxiliar = Path(__file__).resolve().parents[2] / "auxiliary"

        with tempfile.TemporaryDirectory() as directori_temporal:
            fase = opticalPhase(
                str(directori_auxiliar),
                directori_temporal,
                directori_temporal,
            )
            with patch(
                "ism.src.opticalPhase.readIsrf",
                return_value=(isrf, longituds_ona_isrf_um),
            ):
                resultat = fase.spectralIntegration(
                    cub_tao,
                    longituds_ona,
                    "VNIR-0",
                )

        np.testing.assert_allclose(resultat, [[4.25]], rtol=0, atol=1e-12)
