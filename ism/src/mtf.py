from math import pi
from config.ismConfig import ismConfig
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import j1
from numpy.matlib import repmat
from common.io.readMat import writeMat
from common.plot.plotMat2D import plotMat2D
from scipy.interpolate import interp2d
from numpy.fft import fftshift, ifft2
import os

class mtf:
    """
    Class MTF. Collects the analytical modelling of the different contributions
    for the system MTF
    """
    def __init__(self, logger, outdir):
        self.ismConfig = ismConfig()
        self.logger = logger
        self.outdir = outdir

    def system_mtf(self, nlines, ncolumns, D, lambd, focal, pix_size,
                   kLF, wLF, kHF, wHF, defocus, ksmear, kmotion, directory, band, plot=True):
        """
        System MTF
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param pix_size: pixel size in meters [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :param directory: output directory
        :return: mtf
        """

        self.logger.info("Calculation of the System MTF")

        # Calculate the 2D relative frequencies
        self.logger.debug("Calculation of 2D relative frequencies")
        fn2D, fr2D, fnAct, fnAlt = self.freq2d(nlines, ncolumns, D, lambd, focal, pix_size)

        # Diffraction MTF
        self.logger.debug("Calculation of the diffraction MTF")
        Hdiff = self.mtfDiffract(fr2D)

        # Defocus
        Hdefoc = self.mtfDefocus(fr2D, defocus, focal, D)

        # WFE Aberrations
        Hwfe = self.mtfWfeAberrations(fr2D, lambd, kLF, wLF, kHF, wHF)

        # Detector
        Hdet  = self. mtfDetector(fn2D)

        # Smearing MTF
        Hsmear = self.mtfSmearing(fnAlt, ncolumns, ksmear)

        # Motion blur MTF
        Hmotion = self.mtfMotion(fn2D, kmotion)

        # Calculate the System MTF
        self.logger.debug("Calculation of the Sysmtem MTF by multiplying the different contributors")
        Hsys = Hdiff * Hdefoc * Hwfe * Hdet * Hsmear * Hmotion

        # Plot cuts ACT/ALT of the MTF
        if plot:
            self.plotMtf(Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band)


        return Hsys

    def freq2d(self,nlines, ncolumns, D, lambd, focal, w):
        """
        Calculate the relative frequencies 2D (for the diffraction MTF)
        :param nlines: Lines of the TOA
        :param ncolumns: Columns of the TOA
        :param D: Telescope diameter [m]
        :param lambd: central wavelength of the band [m]
        :param focal: focal length [m]
        :param w: pixel size in meters [m]
        :return fn2D: normalised frequencies 2D (f/(1/w))
        :return fr2D: relative frequencies 2D (f/(1/fc))
        :return fnAct: 1D normalised frequencies 2D ACT (f/(1/w))
        :return fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        """
        if nlines <= 0 or ncolumns <= 0 or w <= 0 or lambd <= 0 or D <= 0 or focal <= 0:
            raise ValueError("Las dimensiones y los parámetros ópticos deben ser positivos")
        fAlt = np.fft.fftshift(np.fft.fftfreq(nlines, d=w))
        fAct = np.fft.fftshift(np.fft.fftfreq(ncolumns, d=w))
        [fAltxx, fActxx] = np.meshgrid(fAlt, fAct, indexing='ij')
        f2D = np.sqrt(fAltxx * fAltxx + fActxx * fActxx)

        fn2D = f2D/ (1/w)
        fc = D/(lambd * focal)
        fr2D =  f2D/fc
        fnAct = fAct/(1/w)
        fnAlt = fAlt/(1/w)

        return fn2D, fr2D, fnAct, fnAlt

    def mtfDiffract(self,fr2D):
        """
        Optics Diffraction MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :return: diffraction MTF
        """
        #TODO
        fr_limitada = np.clip(fr2D, 0.0, 1.0)
        Hdiff = (2 / np.pi) * (np.arccos(fr_limitada) - fr_limitada * np.sqrt(1 - fr_limitada * fr_limitada))
        Hdiff[fr2D >= 1.0] = 0.0

        return Hdiff


    def mtfDefocus(self, fr2D, defocus, focal, D):
        """
        Defocus MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param defocus: Defocus coefficient (defocus/(f/N)). 0-2 low defocusing
        :param focal: focal length [m]
        :param D: Telescope diameter [m]
        :return: Defocus MTF
        """
        #TODO

        fr_limitada = np.clip(fr2D, 0.0, 1.0)
        x = np.pi * defocus * fr_limitada * (1 - fr_limitada)
        Hdefoc = np.ones_like(x, dtype=float)
        np.divide(2 * j1(x), x, out=Hdefoc, where=np.abs(x) > 1e-8)


        return Hdefoc

    def mtfWfeAberrations(self, fr2D, lambd, kLF, wLF, kHF, wHF):
        """
        Wavefront Error Aberrations MTF
        :param fr2D: 2D relative frequencies (f/fc), where fc is the optics cut-off frequency
        :param lambd: central wavelength of the band [m]
        :param kLF: Empirical coefficient for the aberrations MTF for low-frequency wavefront errors [-]
        :param wLF: RMS of low-frequency wavefront errors [m]
        :param kHF: Empirical coefficient for the aberrations MTF for high-frequency wavefront errors [-]
        :param wHF: RMS of high-frequency wavefront errors [m]
        :return: WFE Aberrations MTF
        """
        #TODO

        fr_limitada = np.clip(fr2D, 0.0, 1.0)
        Hwfe = np.exp(-fr_limitada * (1 - fr_limitada) * (kLF * (wLF / lambd) ** 2
                    + kHF * (wHF / lambd) ** 2
            )
        )
        return Hwfe

    def mtfDetector(self,fn2D):
        """
        Detector MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :return: detector MTF
        """
        #TODO
        Hdet = np.abs(np.sinc(fn2D))
        return Hdet

    def mtfSmearing(self, fnAlt, ncolumns, ksmear):
        """
        Smearing MTF
        :param ncolumns: Size of the image ACT
        :param fnAlt: 1D normalised frequencies 2D ALT (f/(1/w))
        :param ksmear: Amplitude of low-frequency component for the motion smear MTF in ALT [pixels]
        :return: Smearing MTF
        """

        #TODO

        HsmearAlt = np.sinc(ksmear * fnAlt)

        Hsmear = repmat(HsmearAlt, ncolumns, 1).T

        return Hsmear

    def mtfMotion(self, fn2D, kmotion):
        """
        Motion blur MTF
        :param fnD: 2D normalised frequencies (f/(1/w))), where w is the pixel width
        :param kmotion: Amplitude of high-frequency component for the motion smear MTF in ALT and ACT
        :return: detector MTF
        """
        #TODO
        Hmotion = np.sinc(kmotion * fn2D)

        return Hmotion

    def plotMtf(self,Hdiff, Hdefoc, Hwfe, Hdet, Hsmear, Hmotion, Hsys, nlines, ncolumns, fnAct, fnAlt, directory, band):
        """
        Plotting the system MTF and all of its contributors
        :param Hdiff: Diffraction MTF
        :param Hdefoc: Defocusing MTF
        :param Hwfe: Wavefront electronics MTF
        :param Hdet: Detector MTF
        :param Hsmear: Smearing MTF
        :param Hmotion: Motion blur MTF
        :param Hsys: System MTF
        :param nlines: Number of lines in the TOA
        :param ncolumns: Number of columns in the TOA
        :param fnAct: normalised frequencies in the ACT direction (f/(1/w))
        :param fnAlt: normalised frequencies in the ALT direction (f/(1/w))
        :param directory: output directory
        :param band: band
        :return: N/A
        """
        #Codigo:---------------------------------------------------------------------
        # Central indices
        iAlt = nlines // 2
        iAct = ncolumns // 2

        # Make sure output directory exists
        os.makedirs(directory, exist_ok=True)

        # ---------------------------------------------------------
        # ACT CUT
        # ---------------------------------------------------------
        mascara_act = (fnAct >= 0.0) & (fnAct <= 0.5)
        figura, eje = plt.subplots(figsize=(10, 6))

        eje.plot(fnAct[mascara_act], Hdiff[iAlt, mascara_act], label='Diffraction')
        eje.plot(fnAct[mascara_act], Hdefoc[iAlt, mascara_act], label='Defocus')
        eje.plot(fnAct[mascara_act], Hwfe[iAlt, mascara_act], label='WFE')
        eje.plot(fnAct[mascara_act], Hdet[iAlt, mascara_act], label='Detector')
        eje.plot(fnAct[mascara_act], Hsmear[iAlt, mascara_act], label='Smearing')
        eje.plot(fnAct[mascara_act], Hmotion[iAlt, mascara_act], label='Motion blur')
        eje.plot(fnAct[mascara_act], Hsys[iAlt, mascara_act], label='System MTF',
                 linewidth=2)

        eje.set_xlabel('Normalised spatial frequency ACT')
        eje.set_ylabel('MTF')
        eje.set_title('MTF ACT - Band ' + str(band))
        eje.set_xticks(np.arange(0.0, 0.51, 0.1))
        eje.set_yticks(np.arange(0.0, 1.01, 0.2))
        eje.grid(True, alpha=0.35)
        eje.legend(loc='lower left')
        eje.set_ylim(0.0, 1.05)
        eje.set_xlim(0.0, 0.5)

        figura.tight_layout()

        figura.savefig(
            os.path.join(directory, 'MTF_ACT_band_' + str(band) + '.png'),
            dpi=300
        )

        plt.close(figura)

        # ---------------------------------------------------------
        # ALT CUT
        # ---------------------------------------------------------
        plt.figure(figsize=(10, 6))

        plt.plot(fnAlt, Hdiff[:, iAct], label='Diffraction')
        plt.plot(fnAlt, Hdefoc[:, iAct], label='Defocus')
        plt.plot(fnAlt, Hwfe[:, iAct], label='WFE')
        plt.plot(fnAlt, Hdet[:, iAct], label='Detector')
        plt.plot(fnAlt, Hsmear[:, iAct], label='Smearing')
        plt.plot(fnAlt, Hmotion[:, iAct], label='Motion blur')
        plt.plot(fnAlt, Hsys[:, iAct], label='System MTF',
                 linewidth=2)

        plt.xlabel('Normalised spatial frequency ALT')
        plt.ylabel('MTF')
        plt.title('MTF ALT - Band ' + str(band))
        plt.grid()
        plt.legend()
        plt.ylim([0, 1.05])
        plt.xlim([0, 0.5])

        plt.tight_layout()

        plt.savefig(
            os.path.join(directory, 'MTF_ALT_band_' + str(band) + '.png'),
            dpi=300
        )

        plt.close()

        if band == 'VNIR-0':
            mascara_positiva = (fnAlt >= 0.0) & (fnAlt <= 0.5)
            figura, eje = plt.subplots(figsize=(10, 6))
            eje.plot(fnAlt[mascara_positiva], Hdiff[mascara_positiva, iAct], label='Diffraction MTF')
            eje.plot(fnAlt[mascara_positiva], Hdefoc[mascara_positiva, iAct], label='Defocus MTF')
            eje.plot(fnAlt[mascara_positiva], Hwfe[mascara_positiva, iAct], label='WFE Aberations MTF')
            eje.plot(fnAlt[mascara_positiva], Hdet[mascara_positiva, iAct], label='Detector MTF')
            eje.plot(fnAlt[mascara_positiva], Hsmear[mascara_positiva, iAct], label='Smearing MTF')
            eje.plot(fnAlt[mascara_positiva], Hmotion[mascara_positiva, iAct], label='Motion blur MTF')
            eje.plot(fnAlt[mascara_positiva], Hsys[mascara_positiva, iAct], color='black', linewidth=2, label='System MTF')
            eje.axvline(0.5, color='black', linestyle='--', linewidth=1.2, label='f Nyquist')
            eje.set_xlabel('Spatial frequencies f/(1/w) [-]')
            eje.set_ylabel('MTF')
            eje.set_title('System MTF - slice ALT')
            eje.set_xlim(0.0, 0.5)
            eje.set_ylim(0.0, 1.05)
            eje.grid(True, alpha=0.35)
            eje.legend(loc='lower left', fontsize=8)
            figura.tight_layout()
            figura.savefig(os.path.join(directory, 'figura1.png'), dpi=300)
            plt.close(figura)

            figura, eje = plt.subplots(figsize=(10, 6))
            mapa = eje.imshow(Hsys, origin='lower', aspect='auto', cmap='jet', vmin=0.0, vmax=1.0)
            eje.set_xlabel('ACT')
            eje.set_ylabel('ALT')
            eje.set_title('System MTF for VNIR-0')
            figura.colorbar(mapa, ax=eje, label='MTF')
            figura.tight_layout()
            figura.savefig(os.path.join(directory, 'figura2.png'), dpi=300)
            plt.close(figura)




