import unittest

import numpy as np

from l1b.test.spatial_sampling_distance import calcular_distancia_espacial


class ProvesDistanciaEspacial(unittest.TestCase):
    def test_calcular_distancia_espacial_amb_coordenades_valides_retorna_metres(self):
        latitud = np.array([[0.0, 0.0, 0.0]])
        longitud = np.array([[0.0, 0.001, 0.002]])

        distancies = calcular_distancia_espacial(latitud, longitud, 0)

        self.assertEqual(distancies.shape, (2,))
        np.testing.assert_allclose(distancies, [111.31949, 111.31949], rtol=1e-5)

    def test_calcular_distancia_espacial_amb_dimensions_incompatibles_llanca_error(self):
        with self.assertRaisesRegex(ValueError, "same shape"):
            calcular_distancia_espacial(np.zeros((2, 3)), np.zeros((2, 2)), 0)

    def test_calcular_distancia_espacial_amb_index_foradades_llanca_error(self):
        with self.assertRaisesRegex(ValueError, "line index"):
            calcular_distancia_espacial(np.zeros((2, 3)), np.zeros((2, 3)), 2)
