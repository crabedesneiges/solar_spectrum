import numpy as np
from astropy.constants import c as c_light


def convert_wavenumber_to_freq(v_cm, spectrum):
    """
    Convertit un axe wavenumber (cm⁻¹) en fréquence (Hz) et trie par ordre croissant.

    Parameters
    ----------
    v_cm     : array   Wavenumbers (cm⁻¹)
    spectrum : array   Spectre associé

    Returns
    -------
    f_hz_sorted      : array   Fréquences triées (Hz)
    spectrum_sorted  : array   Spectre trié
    """
    c_cms = c_light.cgs.value
    f_hz = c_cms * v_cm
    sort_idx = np.argsort(f_hz)
    return f_hz[sort_idx], spectrum[sort_idx]


def loss_l2(f, u_obs, S_sim):
    """
    Norme L² intégrée : E = ∫ (u_obs − S_sim)² df  (règle des trapèzes).
    """
    return np.trapz((u_obs - S_sim)**2, x=f)


def loss_l2_normalized(f, u_obs, S_sim):
    """
    Norme L² normalisée par la largeur spectrale.
    Évite que les zones plates dominent sur les raies.
    """
    return np.trapz((u_obs - S_sim)**2, x=f) / (f[-1] - f[0])
