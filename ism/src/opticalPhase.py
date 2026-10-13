
from ism.src.initIsm import initIsm
from math import pi
from ism.src.mtf import mtf
from numpy.fft import fftshift, ifftshift, ifft2, fft2
import numpy as np
from common.io.writeToa import writeToa
from common.io.readIsrf import readIsrf
from scipy.interpolate import interp1d
from scipy.integrate import trapezoid
from common.plot.plotMat2D import plotMat2D
from common.plot.plotF import plotF
from scipy.signal import convolve2d
from common.src.auxFunc import getIndexBand

class opticalPhase(initIsm):

    def __init__(self, auxdir, indir, outdir):
        super().__init__(auxdir, indir, outdir)

    def compute(self, sgm_toa, sgm_wv, band):
        """
        The optical phase is in charge of simulating the radiance
        to irradiance conversion, the spatial filter (PSF)
        and the spectral filter (ISRF).
        :return: TOA image in irradiances [mW/m2/nm],
                    with spatial and spectral filter
        """
        self.logger.info("EODP-ALG-ISM-1000: Optical stage")

        # Calculation and application of the ISRF
        # -------------------------------------------------------------------------------
        self.logger.info("EODP-ALG-ISM-1010: Spectral modelling. ISRF")
        toa = self.spectralIntegration(sgm_toa, sgm_wv, band)

        self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [e-]")

        if self.ismConfig.save_after_isrf:
            saveas_str = self.globalConfig.ism_toa_isrf + band
            writeToa(self.outdir, saveas_str, toa)

        # Radiance to Irradiance conversion
        # -------------------------------------------------------------------------------
        self.logger.info("EODP-ALG-ISM-1020: Radiances to Irradiances")
        toa = self.rad2Irrad(toa,
                             self.ismConfig.D,
                             self.ismConfig.f,
                             self.ismConfig.Tr)

        self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [e-]")

        # Spatial filter
        # -------------------------------------------------------------------------------
        # Calculation and application of the system MTF
        self.logger.info("EODP-ALG-ISM-1030: Spatial modelling. PSF/MTF")
        myMtf = mtf(self.logger, self.outdir)
        Hsys = myMtf.system_mtf(toa.shape[0], toa.shape[1],
                                self.ismConfig.D, self.ismConfig.wv[getIndexBand(band)], self.ismConfig.f, self.ismConfig.pix_size,
                                self.ismConfig.kLF, self.ismConfig.wLF, self.ismConfig.kHF, self.ismConfig.wHF,
                                self.ismConfig.defocus, self.ismConfig.ksmear, self.ismConfig.kmotion,
                                self.outdir, band)

        # Apply system MTF
        toa = self.applySysMtf(toa, Hsys, band, self.ismConfig.mtf_padding_pixels)
        self.logger.debug("TOA [0,0] " +str(toa[0,0]) + " [e-]")



        # Write output TOA & plots
        # -------------------------------------------------------------------------------
        if self.ismConfig.save_optical_stage:
            saveas_str = self.globalConfig.ism_toa_optical + band

            writeToa(self.outdir, saveas_str, toa)

            title_str = 'TOA after the optical phase [mW/sr/m2]'
            xlabel_str='ACT'
            ylabel_str='ALT'
            plotMat2D(toa, title_str, xlabel_str, ylabel_str, self.outdir, saveas_str)

            idalt = int(toa.shape[0]/2)
            saveas_str = saveas_str + '_alt' + str(idalt)
            plotF([], toa[idalt,:], title_str, xlabel_str, ylabel_str, self.outdir, saveas_str)

        return toa

    def rad2Irrad(self, toa, D, f, Tr):
        """
        Radiance to Irradiance conversion
        :param toa: Input TOA image in radiances [mW/sr/m2]
        :param D: Pupil diameter [m]
        :param f: Focal length [m]
        :param Tr: Optical transmittance [-]
        :return: TOA image in irradiances [mW/m2]
        """
        # TODO
        irrad = Tr * toa * (np.pi / 4.0) * ((D / f) ** 2)
        return irrad


    def applySysMtf(self, toa, Hsys, band, padding_pixels=16):
        """Aplica la MTF del sistema con extensión reflectante para limitar artefactos de borde."""
        imagen = np.asarray(toa, dtype=float)
        if imagen.ndim != 2 or Hsys.shape != imagen.shape:
            raise ValueError("La imagen TOA y la MTF deben ser matrices 2D de la misma forma")
        if not isinstance(padding_pixels, (int, np.integer)) or padding_pixels < 0:
            raise ValueError("El margen de padding debe ser un entero no negativo")
        padding = int(padding_pixels)
        if padding == 0:
            return np.real(ifft2(fft2(imagen) * ifftshift(Hsys)))
        imagen_ext = np.pad(imagen, ((padding, padding), (padding, padding)), mode="reflect")
        mtf_extendida = mtf(self.logger, self.outdir).system_mtf(
            imagen_ext.shape[0], imagen_ext.shape[1], self.ismConfig.D, self.ismConfig.wv[getIndexBand(band)], self.ismConfig.f, self.ismConfig.pix_size,
            self.ismConfig.kLF, self.ismConfig.wLF, self.ismConfig.kHF, self.ismConfig.wHF, self.ismConfig.defocus, self.ismConfig.ksmear, self.ismConfig.kmotion, self.outdir, band, plot=False
        )
        filtrada = np.real(ifft2(fft2(imagen_ext) * ifftshift(mtf_extendida)))
        return filtrada[padding:padding + imagen.shape[0], padding:padding + imagen.shape[1]]

    def spectralIntegration(self, sgm_toa, sgm_wv, band):
        """
        Integration with the ISRF to retrieve one band
        :param sgm_toa: Spectrally oversampled TOA cube 3D in irradiances [mW/m2]
        :param sgm_wv: wavelengths of the input TOA cube
        :param band: band
        :return: TOA image 2D in radiances [mW/m2]
        """
        if sgm_toa.ndim != 3 or np.asarray(sgm_wv).ndim != 1 or sgm_toa.shape[2] != len(sgm_wv):
            raise ValueError("El cubo TOA debe ser 3D y su eje espectral debe coincidir con sgm_wv")
        sgm_wv = np.asarray(sgm_wv, dtype=float)
        if not np.all(np.isfinite(sgm_wv)) or np.any(np.diff(sgm_wv) <= 0):
            raise ValueError("Las longitudes de onda de la escena deben ser finitas y estrictamente crecientes")
        isrf, wv_isrf = readIsrf(self.auxdir + '/' + self.ismConfig.isrffile, band)
        isrf = np.asarray(isrf, dtype=float)
        wv_isrf = np.asarray(wv_isrf, dtype=float)
        if isrf.ndim != 1 or wv_isrf.ndim != 1 or isrf.size != wv_isrf.size:
            raise ValueError("La ISRF y su eje de longitudes de onda deben ser vectores de igual tamaño")
        wv_isrf_nm = wv_isrf * 1000.0
        if np.any(np.diff(wv_isrf_nm) <= 0):
            raise ValueError("Las longitudes de onda ISRF deben ser estrictamente crecientes")
        integral_isrf = trapezoid(isrf, x=wv_isrf_nm)
        if not np.isfinite(integral_isrf) or integral_isrf <= 0:
            raise ValueError("La integral de la ISRF debe ser positiva y finita")
        isrf_normalizada = isrf / integral_isrf
        isrf_en_escena = interp1d(
            wv_isrf_nm, isrf_normalizada, kind="linear", bounds_error=False, fill_value=0.0
        )(sgm_wv)
        toa = trapezoid(
            np.asarray(sgm_toa, dtype=float) * isrf_en_escena[np.newaxis, np.newaxis, :],
            x=sgm_wv, axis=2
        )
        return toa


