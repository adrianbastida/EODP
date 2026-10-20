
# LEVEL-1B MODULE

from l1b.src.initL1b import initL1b
from common.io.writeToa import writeToa, readToa, readQualityMask
from common.src.auxFunc import getIndexBand
from common.io.readFactor import readFactor, EQ_MULT, EQ_ADD, NC_EXT
import numpy as np
import os
import matplotlib.pyplot as plt
from common.plot.plotMat2D import plotMat2D
from scipy.interpolate import interp1d, griddata
from scipy.signal import fftconvolve
import pywt
from config.ismConfig import ismConfig

class l1b(initL1b):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)
        self.ismConfig = ismConfig()

    def processModule(self):

        self.logger.info("Start of the L1B Processing Module")

        for band in self.globalConfig.bands:

            self.logger.info("Start of BAND " + band)

            # Read TOA - output of the ISM in Digital Numbers
            # -------------------------------------------------------------------------------
            toa = readToa(self.indir, self.globalConfig.ism_toa + band + '.nc')
            toa_dn_original = np.array(toa, dtype=float, copy=True)
            quality = readQualityMask(self.indir, self.globalConfig.ism_toa + band + '.nc')
            if quality is None:
                quality = self.buildQualityMask(toa, band)
            if self.l1bConfig.do_cosmetic_filling:
                toa = self.cosmeticFilling(toa, quality)

            # Equalization (radiometric correction)
            # -------------------------------------------------------------------------------
            if self.l1bConfig.do_equalization:
                self.logger.info("EODP-ALG-L1B-1010: Radiometric Correction (equalization)")

                # Read the multiplicative and additive factors from auxiliary/equalization/
                eq_mult = readFactor(os.path.join(self.auxdir,self.l1bConfig.eq_mult+band+NC_EXT),EQ_MULT)
                eq_add = readFactor(os.path.join(self.auxdir,self.l1bConfig.eq_add+band+NC_EXT),EQ_ADD)

                # Do the equalization and save to file
                toa = self.equalization(toa, eq_add, eq_mult)
                writeToa(self.outdir, self.globalConfig.l1b_toa_eq + band, toa)

            # Restitution (absolute radiometric gain)
            # -------------------------------------------------------------------------------
            self.logger.info("EODP-ALG-L1B-1020: Absolute radiometric gain application (restoration)")
            toa = self.restoration(toa, self.l1bConfig.gain[getIndexBand(band)])
            if self.l1bConfig.do_straylight_correction:
                toa = self.kernelCorrection(toa, self.l1bConfig.straylight_kernel, "straylight")
            if self.l1bConfig.do_psf_deconvolution:
                toa = self.kernelCorrection(toa, self.l1bConfig.psf_kernel, "PSF")
            if self.l1bConfig.denoise_wavelet:
                toa = self.waveletDenoise(toa)

            # Write output TOA
            # -------------------------------------------------------------------------------
            writeToa(self.outdir, self.globalConfig.l1b_toa + band, toa)
            self.plotL1bToa(toa, self.outdir, band)
            self.plotEqualizationModel(toa_dn_original, band)

            self.logger.info("End of BAND " + band)

        self.logger.info("End of the L1B Module!")


    def equalization(self, toa, eq_add, eq_mult):
        """
        Equlization. Apply an offset and a gain.
        :param toa: TOA in DN
        :param eq_add: Offset in DN
        :param eq_mult: Gain factor, adimensional
        :return: TOA in DN, equalized
        """
        imagen = np.asarray(toa, dtype=float)
        factor_aditivo = np.asarray(eq_add, dtype=float)
        factor_multiplicativo = np.asarray(eq_mult, dtype=float)
        if imagen.ndim != 2 or factor_aditivo.shape not in ((), (imagen.shape[1],)) or factor_multiplicativo.shape not in ((), (imagen.shape[1],)):
            raise ValueError("La imagen debe ser 2D y los factores de ecualización deben ser escalares o vectores del tamaño ACT")
        if not np.all(np.isfinite(factor_aditivo)) or not np.all(np.isfinite(factor_multiplicativo)) or np.any(factor_multiplicativo == 0):
            raise ValueError("Los factores de ecualización deben ser finitos y el multiplicativo distinto de cero")
        return (imagen - factor_aditivo) / factor_multiplicativo

    def restoration(self,toa,gain):
        """
        Absolute Radiometric Gain - restore back to radiances
        :param toa: TOA in DN
        :param gain: gain in [rad/DN]
        :return: TOA in radiances [mW/sr/m2]
        """
        #TODO
        toa=toa*gain
        if toa.size:
            self.logger.debug('Sanity check. TOA in radiances after gain application ' + str(toa[-1,-1]) + ' [mW/m2/sr]')

        return toa

    def buildQualityMask(self, toa, band):
        forma = np.shape(toa)
        mascara = np.zeros(forma, dtype=np.uint8)
        total = forma[1]
        num_malas = int(total * self.ismConfig.bad_pix / 100.0)
        num_muertas = int(total * self.ismConfig.dead_pix / 100.0)
        paso_malas = max(1, int(total / num_malas)) if num_malas else total
        paso_muertas = max(1, int(total / num_muertas)) if num_muertas else total
        indices_malas = np.arange(5, total, paso_malas)[:num_malas] if num_malas else np.array([], dtype=int)
        indices_muertas = np.arange(0, total, paso_muertas)[:num_muertas] if num_muertas else np.array([], dtype=int)
        indices_malas = np.setdiff1d(indices_malas, indices_muertas, assume_unique=True)
        mascara[:, indices_malas] |= self.globalConfig.quality_bad_pixel
        mascara[:, indices_muertas] |= self.globalConfig.quality_dead_pixel
        mascara[np.asarray(toa) >= (2 ** self.ismConfig.bit_depth - 1)] |= self.globalConfig.quality_saturated_pixel
        return mascara

    def cosmeticFilling(self, toa, quality):
        imagen = np.asarray(toa, dtype=float)
        mascara = np.asarray(quality, dtype=np.uint8)
        if imagen.ndim != 2 or mascara.shape != imagen.shape:
            raise ValueError("La máscara de calidad debe ser 2D y tener la misma forma que TOA")
        bits_permanentes = self.globalConfig.quality_bad_pixel | self.globalConfig.quality_dead_pixel
        columnas_defectuosas = np.any((mascara & bits_permanentes) != 0, axis=0)
        columnas_validas = np.flatnonzero(~columnas_defectuosas)
        if columnas_defectuosas.any() and columnas_validas.size < 2:
            raise ValueError("No hay suficientes columnas sanas para rellenar los píxeles defectuosos")
        alt_indices = np.arange(imagen.shape[0])
        for columna in np.flatnonzero(columnas_defectuosas):
            for linea in alt_indices:
                saturados_fila = (mascara[linea] & self.globalConfig.quality_saturated_pixel) != 0
                validas_fila = np.flatnonzero(~columnas_defectuosas & ~saturados_fila)
                if validas_fila.size < 2:
                    raise ValueError(f"No hay suficientes píxeles sanos en la línea {linea} para rellenar defectos")
                orden = 3 if validas_fila.size >= 4 else 1
                interpolador = interp1d(validas_fila, imagen[linea, validas_fila], kind=orden, bounds_error=False, fill_value="extrapolate")
                imagen[linea, columna] = interpolador(columna)
        saturados = (mascara & self.globalConfig.quality_saturated_pixel) != 0
        buenos = ~saturados & ~np.broadcast_to(columnas_defectuosas, imagen.shape)
        if saturados.any():
            puntos = np.column_stack(np.nonzero(buenos))
            if puntos.shape[0] < 3:
                raise ValueError("No hay suficientes píxeles válidos para rellenar las saturaciones")
            ubicaciones = np.column_stack(np.nonzero(saturados))
            relleno = griddata(puntos, imagen[buenos], ubicaciones, method="cubic")
            faltantes = ~np.isfinite(relleno)
            if faltantes.any():
                relleno[faltantes] = griddata(puntos, imagen[buenos], ubicaciones[faltantes], method="nearest")
            imagen[tuple(ubicaciones.T)] = relleno
        return imagen

    def kernelCorrection(self, toa, kernel, nombre):
        imagen = np.asarray(toa, dtype=float)
        filtro = np.asarray(kernel, dtype=float) if kernel is not None else np.array([])
        if filtro.ndim != 2 or not np.all(np.isfinite(filtro)):
            raise ValueError(f"Debe configurarse un kernel 2D finito calibrado para la corrección {nombre}")
        if nombre == "straylight":
            margen = tuple(dimension // 2 for dimension in filtro.shape)
            imagen_ext = np.pad(imagen, tuple((valor, valor) for valor in margen), mode="reflect") if any(margen) else imagen
            estimacion_ext = fftconvolve(imagen_ext, filtro, mode="same")
            estimacion = estimacion_ext[margen[0]:margen[0] + imagen.shape[0], margen[1]:margen[1] + imagen.shape[1]] if any(margen) else estimacion_ext
            return imagen - estimacion
        if nombre != "PSF":
            raise ValueError("La corrección de deconvolución sólo admite un kernel PSF")
        if filtro.shape[0] > imagen.shape[0] or filtro.shape[1] > imagen.shape[1]:
            raise ValueError("La PSF debe ser menor que la imagen que se va a deconvolucionar")
        suma_psf = np.sum(filtro)
        if not np.isfinite(suma_psf) or suma_psf <= 0:
            raise ValueError("La suma de la PSF debe ser positiva y finita")
        psf = filtro / suma_psf
        regularizacion = float(self.l1bConfig.psf_wiener_regularization)
        if not np.isfinite(regularizacion) or regularizacion <= 0:
            raise ValueError("La regularización de Wiener debe ser finita y positiva")
        padding = max(psf.shape) // 2
        imagen_ext = np.pad(imagen, ((padding, padding), (padding, padding)), mode="reflect") if padding else imagen
        psf_extendida = np.zeros(imagen_ext.shape, dtype=float)
        inicio_ext = tuple((dimension_imagen - dimension_psf) // 2 for dimension_imagen, dimension_psf in zip(imagen_ext.shape, psf.shape))
        psf_extendida[inicio_ext[0]:inicio_ext[0] + psf.shape[0], inicio_ext[1]:inicio_ext[1] + psf.shape[1]] = psf
        otf_extendida = np.fft.fft2(np.fft.ifftshift(psf_extendida))
        transferencia_wiener = np.conj(otf_extendida) / (np.abs(otf_extendida) ** 2 + regularizacion)
        restaurada = np.real(np.fft.ifft2(np.fft.fft2(imagen_ext) * transferencia_wiener))
        return restaurada[padding:padding + imagen.shape[0], padding:padding + imagen.shape[1]] if padding else restaurada

    def waveletDenoise(self, toa):
        imagen = np.asarray(toa, dtype=float)
        coeficientes = pywt.wavedec2(imagen, "db2", mode="symmetric")
        detalle = coeficientes[-1][-1]
        sigma = np.median(np.abs(detalle)) / 0.6744897501960817
        umbral = sigma * np.sqrt(2.0 * np.log(max(imagen.size, 2)))
        coeficientes_filtrados = [coeficientes[0]] + [
            tuple(pywt.threshold(banda, umbral, mode="soft") for banda in nivel)
            for nivel in coeficientes[1:]
        ]
        return pywt.waverec2(coeficientes_filtrados, "db2", mode="symmetric")[:imagen.shape[0], :imagen.shape[1]]

    def plotL1bToa(self, toa_l1b, outputdir, band):
        saveas_str = self.globalConfig.l1b_toa + band
        title_str = 'TOA Level-1B [mW/m2/sr]'
        plotMat2D(toa_l1b, title_str, 'ACT', 'ALT', outputdir, saveas_str)

    def plotEqualizationModel(self, toa_dn_original, band):
        indice_linia = 50
        if np.ndim(toa_dn_original) != 2 or np.shape(toa_dn_original)[0] <= indice_linia:
            self.logger.warning("Cannot generate the comparison for %s: ALT line 50 is required", band)
            return
        ruta_eq_mult = os.path.join(self.auxdir, self.l1bConfig.eq_mult + band + NC_EXT)
        ruta_eq_add = os.path.join(self.auxdir, self.l1bConfig.eq_add + band + NC_EXT)
        if not os.path.isfile(ruta_eq_mult) or not os.path.isfile(ruta_eq_add):
            self.logger.warning("Cannot generate the comparison: missing equalization factors for %s", band)
            return
        eq_mult = readFactor(ruta_eq_mult, EQ_MULT)
        eq_add = readFactor(ruta_eq_add, EQ_ADD)
        toa_eq_dn = self.equalization(toa_dn_original, eq_add, eq_mult)
        ganancia = self.l1bConfig.gain[getIndexBand(band)]
        toa_l1b_eq = toa_eq_dn * ganancia
        toa_l1b_sin_eq = toa_dn_original * ganancia
        archivo_referencia = os.path.join(self.indir, self.globalConfig.ism_toa_isrf + band + NC_EXT)
        if not os.path.isfile(archivo_referencia):
            self.logger.warning("Cannot generate the comparison: missing ISRF reference %s", archivo_referencia)
            return
        toa_isrf = readToa(self.indir, self.globalConfig.ism_toa_isrf + band + NC_EXT)
        if toa_isrf.shape != toa_dn_original.shape:
            self.logger.warning("Cannot generate the comparison: ISRF shape %s and DN shape %s do not match", toa_isrf.shape, toa_dn_original.shape)
            return
        indices_act = np.arange(toa_dn_original.shape[1])
        figura, eje = plt.subplots(figsize=(12, 6))
        eje.plot(indices_act, toa_l1b_eq[indice_linia], color='black', label='TOA L1B with equalization')
        eje.plot(indices_act, toa_l1b_sin_eq[indice_linia], color='red', label='TOA L1B without equalization')
        eje.plot(indices_act, toa_isrf[indice_linia], color='blue', label='TOA after the ISRF')
        eje.set_title(f'Effect of equalization for {band} - ALT {indice_linia}')
        eje.set_xlabel('ACT pixel [-]')
        eje.set_ylabel('TOA [mW/m2/sr]')
        eje.grid(True, linestyle='--', alpha=0.55)
        eje.legend(loc='upper left')
        figura.tight_layout()
        numero_figura = getIndexBand(band) + 1
        ruta_figura = os.path.join(self.outdir, f'figure_{numero_figura}_equalization_{band}.png')
        figura.savefig(ruta_figura, dpi=300)
        if band == 'VNIR-0':
            figura.savefig(os.path.join(self.outdir, 'figura3.png'), dpi=300)
        plt.close(figura)
        self.logger.info("Equalization figure %s saved to %s", numero_figura, ruta_figura)
