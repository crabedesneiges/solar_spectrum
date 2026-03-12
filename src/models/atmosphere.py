import numpy as np
import scipy.constants as const


def planck_function(f, T):
    """
    Radiance spectrale (loi de Planck) en fonction de la fréquence.

    Parameters
    ----------
    f : array   Fréquences (Hz)
    T : float   Température de surface (K)
    """
    h = const.h
    c = const.c
    k = const.k

    with np.errstate(over='ignore'):
        exponent = np.clip((h * f) / (k * T), None, 700)
        B_f = (2 * h * f**3 / c**2) / (np.exp(exponent) - 1)
    return B_f


def doppler_profile(f, f0, T_atm, mass_amu):
    """
    Profil Doppler gaussien normalisé.

    Parameters
    ----------
    f        : array   Fréquences (Hz)
    f0       : float   Fréquence centrale (Hz)
    T_atm    : float   Température atmosphère (K)
    mass_amu : float   Masse atomique (u)
    """
    k = const.k
    c_ = const.c
    m_kg = mass_amu * 1.66054e-27

    sigma = f0 * np.sqrt(np.abs(k * T_atm / (m_kg * c_**2)))
    if sigma == 0:
        return np.zeros_like(f, dtype=float)
    phi = (1.0 / (sigma * np.sqrt(2 * np.pi))) * \
          np.exp(-0.5 * ((f - f0) / sigma)**2)
    return phi


def forward_model_transmittance(f, T_atm, lines_params):
    """
    Modèle de transmittance Beer-Lambert avec profils Doppler.

    Parameters
    ----------
    f            : array   Fréquences (Hz)
    T_atm        : float   Température de l'atmosphère (K)
    lines_params : list of dict
        Chaque dict contient : {'f0': Hz, 'mass': amu, 'c': colonne optique}

    Returns
    -------
    transmittance : array (0–1)
    """
    tau_total = np.zeros_like(f, dtype=float)
    for line in lines_params:
        phi = doppler_profile(f, line['f0'], T_atm, line['mass'])
        tau_total += line['c'] * phi
    return np.exp(-tau_total)


def full_forward_model(f, K, T_surface, T_atm, concentrations, lines_info):
    """
    Modèle physique complet : corps noir × absorption atmosphérique.

    S(f) = K · B(f, T_surface) · exp(−Σ τ_i(f, T_atm))

    Parameters
    ----------
    f              : array          Fréquences (Hz)
    K              : float          Facteur de normalisation
    T_surface      : float          Température de surface du Soleil (K) — contrôle l'enveloppe
    T_atm          : float          Température de l'atmosphère solaire (K) — contrôle la largeur des raies
    concentrations : array (N,)     Colonnes optiques pour chaque raie
    lines_info     : list of dict   Chaque dict contient 'f0_hz' et 'mass'

    Returns
    -------
    spectrum : array   Spectre modélisé (mêmes unités que K · B)
    """
    continuum = K * planck_function(f, T_surface)
    transmittance = forward_model_N(f, T_atm, concentrations, lines_info)
    return continuum * transmittance


def forward_model_N(f, T_atm, concentrations, lines_info):
    """
    Modèle de transmittance pour N raies avec concentrations séparées.

    Parameters
    ----------
    f              : array          Fréquences (Hz)
    T_atm          : float          Température de l'atmosphère (K)
    concentrations : array (N,)     Colonnes optiques pour chaque raie
    lines_info     : list of dict   Chaque dict contient 'f0_hz' et 'mass'

    Returns
    -------
    transmittance : array (0–1)
    """
    tau_total = np.zeros_like(f, dtype=float)
    for c, line in zip(concentrations, lines_info):
        phi = doppler_profile(f, line['f0_hz'], T_atm, line['mass'])
        tau_total += c * phi
    return np.exp(-tau_total)
