import astropy.io.fits as fits
import matplotlib.pylab as plt
import matplotlib
import numpy as np

cmap = matplotlib.colormaps.get_cmap('rainbow')

DATA_PATH = './data/IAG/'


class open_iag():
    """
    Open data and store as class attributes

    inputs:
    -------
    group : int
            group identifier (0-10) to indicate which telluric spectra to load

    attributes:
    ----------
    v           - array [npoint] wavenumber array (cm-1)
    solar_atlas - array [npoint] IAG telluric corrected solar atlas
    err_atlas   - array [npoint] 1 sigma error values for solar_atlas
    flag_atlas  - array [npoint] flag values for solar_atlas
    nspec       - int  number of spectra in group
    CD_H2O      - float  log_10 of H2O column density (median across all fits)
    CD_O2       - float  log_10 of O2 column density (median across all fits)
    stel_mod    - array [npoint] solar model used to make telluric spectrum shifted to rest frame
    res_med     - array [npoint] median of residuals
    iodine      - array [npoint] iodine template
    jd          - array [nspec] julian date of observation
    tau         - array [nspec] best fit H2O optical depth for spectrum
    airmass     - array [nspec] airmass of observation
    tel_spectra - array [nspec, npoint] telluric spectra
    """

    def __init__(self, group, data_path=DATA_PATH):
        self.group = group

        f = fits.open(data_path + 'telluric_spectra_%s.fits' % int(group))
        fsol = fits.open(data_path + 'iag_telfree_solaratlas.fits')

        self.nspec = f[0].header['NSPEC']
        self.CD_H2O = f[0].header['CD_H2O']
        self.CD_O2 = f[0].header['CD_O2']

        self.v = f[1].data['v']
        self.stel_mod = f[1].data['stel_mod']
        self.res_med = f[1].data['res_med']
        self.iodine = f[1].data['iodine']

        specnums = []
        for col in f[1].columns:
            if col.name.startswith('telluric'):
                specnums.append(col.name[9:])
        self.specnums = np.array(specnums, dtype='str')

        self.tau = np.zeros(self.nspec)
        self.jd = np.zeros(self.nspec)
        self.airmass = np.zeros(self.nspec)
        self.tel_spectra = np.zeros((self.nspec, len(self.v)))
        for i, specnum in enumerate(specnums):
            self.tau[i] = f[0].header['TAU_' + specnum]
            self.jd[i] = f[0].header['JD_' + specnum]
            self.airmass[i] = f[0].header['AMS_' + specnum]
            self.tel_spectra[i] = f[1].data['telluric_' + specnum]
        f.close()

        self.solar_atlas = fsol[1].data['s']
        self.err_atlas = fsol[1].data['err']
        self.flag_atlas = fsol[1].data['flags']
        fsol.close()

    def plot_tel(self, scale=None):
        """
        Plot telluric spectra.

        scale : None | 'O2' | 'H2O'
            None  — no scaling
            'O2'  — scale by airmass
            'H2O' — scale by airmass × tau
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))

        if scale is None:
            for i in range(self.nspec):
                i_color = np.where(i == np.argsort(self.airmass * self.tau))[0][0]
                ax.plot(self.v, self.tel_spectra[i], c=cmap(1.0 * i_color / self.nspec))
        elif scale == 'O2':
            for i in range(self.nspec):
                i_color = np.where(i == np.argsort(self.airmass))[0][0]
                ax.plot(self.v, self.tel_spectra[i] ** (1 / self.airmass[i]),
                        c=cmap(1.0 * i_color / self.nspec))
        elif scale == 'H2O':
            for i in range(self.nspec):
                i_color = np.where(i == np.argsort(self.airmass * self.tau))[0][0]
                ax.plot(self.v, self.tel_spectra[i] ** (1 / (self.airmass[i] * self.tau[i])),
                        c=cmap(1.0 * i_color / self.nspec))

        ax.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax.set_ylabel('Transmittance')
        ax.set_title('Group %s' % self.group)
        plt.subplots_adjust(bottom=0.2, left=0.18)
        ax.set_ylim(0, 1.2)
        return ax

    def plot_stel(self, flag_level=None, plot_tel=False):
        """
        Plot telluric-corrected solar atlas with 2σ error band.

        flag_level : None | float (0–4)
            Mark points with flag > flag_level.
        plot_tel : bool
            Overlay mean telluric spectrum.
        """
        fig, ax = plt.subplots(1, 1, figsize=(6, 3))
        ax.fill_between(self.v,
                        self.solar_atlas - 2 * self.err_atlas,
                        self.solar_atlas + 2 * self.err_atlas,
                        facecolor='gray', label=r'2$\sigma$')
        ax.plot(self.v, self.solar_atlas, 'k', lw=0.5, label='IAG Solar Atlas')
        ax.set_xlabel('Wavenumber (cm$^{-1}$)')
        ax.set_ylabel('Transmittance')
        plt.subplots_adjust(bottom=0.2, left=0.18)
        ax.set_ylim(0, 1.2)

        if flag_level is not None:
            imark = np.where(self.flag_atlas > flag_level)[0]
            if len(imark) > 0:
                ax.plot(self.v[imark], self.solar_atlas[imark], 'm.', label='Flagged')

        if plot_tel:
            ax.plot(self.v, np.mean(self.tel_spectra, axis=0), c='steelblue', label='Mean Telluric')

        ax.legend(loc='best', fontsize=12)
        return ax

    def plot_visible_spectrum(self, index=0, threshold=0.95):
        """
        Affiche le spectre tellurique comme un spectre d'absorption coloré.

        index     : int   — index du spectre tellurique à afficher
        threshold : float — en dessous de cette valeur, une raie sombre est tracée
        """
        wl = 1e7 / self.v
        trans = np.clip(self.tel_spectra[index], 0, 1)

        sort_idx = np.argsort(wl)
        wl = wl[sort_idx]
        trans = trans[sort_idx]

        def get_spectrum_rgb(wls):
            r = np.zeros_like(wls)
            g = np.zeros_like(wls)
            b = np.zeros_like(wls)

            m = (wls >= 380) & (wls < 440)
            r[m] = -(wls[m] - 440) / (440 - 380); b[m] = 1.0

            m = (wls >= 440) & (wls < 490)
            g[m] = (wls[m] - 440) / (490 - 440); b[m] = 1.0

            m = (wls >= 490) & (wls < 510)
            g[m] = 1.0; b[m] = -(wls[m] - 510) / (510 - 490)

            m = (wls >= 510) & (wls < 580)
            r[m] = (wls[m] - 510) / (580 - 510); g[m] = 1.0

            m = (wls >= 580) & (wls < 645)
            r[m] = 1.0; g[m] = -(wls[m] - 645) / (645 - 580)

            m = (wls >= 645) & (wls <= 750)
            r[m] = 1.0

            factor = np.zeros_like(wls)
            factor = np.where((wls >= 380) & (wls <= 750), 1.0, factor)
            factor = np.where((wls >= 380) & (wls < 420),
                              0.3 + 0.7 * (wls - 380) / (420 - 380), factor)
            factor = np.where((wls > 700) & (wls <= 750),
                              0.3 + 0.7 * (750 - wls) / (750 - 700), factor)
            return np.stack([r * factor, g * factor, b * factor], axis=-1)

        rgb_base = get_spectrum_rgb(wl)
        absorption = np.where(trans < threshold, trans, 1.0)
        rgb_final = rgb_base * absorption[:, np.newaxis]

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.imshow(rgb_final.reshape(1, -1, 3),
                  extent=[wl.min(), wl.max(), 0, 1],
                  aspect='auto', origin='lower')
        ax.axvline(750, color='white', lw=1, linestyle='--', alpha=0.4, label='Limite visible (750 nm)')
        ax.legend(loc='lower right', fontsize=9, framealpha=0.3)
        ax.set_xlabel("Longueur d'onde (nm)")
        ax.set_yticks([])
        ax.set_title(f'Spectre Tellurique - Groupe {self.group} (Spec {index})')
        plt.tight_layout()
        return ax
