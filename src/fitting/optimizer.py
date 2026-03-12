import numpy as np
from scipy.optimize import minimize

from src.models.atmosphere import (
    planck_function,
    forward_model_transmittance,
    forward_model_N,
    full_forward_model,
    doppler_profile,
)
from src.utils.spectral import loss_l2, loss_l2_normalized


def model_na(f, T_atm, c1, c2, f1, f2):
    """
    Modèle à 2 raies pour le Sodium (D1 et D2).

    Parameters
    ----------
    f     : array   Fréquences (Hz)
    T_atm : float   Température atmosphère (K)
    c1    : float   Colonne optique raie D2
    c2    : float   Colonne optique raie D1
    f1    : float   Fréquence raie D2 (Hz)
    f2    : float   Fréquence raie D1 (Hz)
    """
    mass_na = 22.9897
    lines = [
        {'f0': f1, 'mass': mass_na, 'c': c1},
        {'f0': f2, 'mass': mass_na, 'c': c2},
    ]
    return forward_model_transmittance(f, T_atm, lines)


def objective_na(params, f_obs, trans_obs):
    """
    Fonction objectif L² pour le modèle Na à 5 paramètres.
    params = [T_atm/1000, c1/1e10, c2/1e10, f1/1e14, f2/1e14]
    """
    T_atm = params[0] * 1000.0
    c1 = params[1] * 1e10
    c2 = params[2] * 1e10
    f1 = params[3] * 1e14
    f2 = params[4] * 1e14
    trans_sim = model_na(f_obs, T_atm, c1, c2, f1, f2)
    return loss_l2(f_obs, trans_obs, trans_sim)


def fit_na_lines(f_obs, trans_obs, f_D2, f_D1):
    """
    Ajuste les raies D du Sodium sur le spectre observé.

    Parameters
    ----------
    f_obs     : array   Fréquences (Hz)
    trans_obs : array   Transmittance observée
    f_D2      : float   Fréquence Na D2 NIST (Hz)
    f_D1      : float   Fréquence Na D1 NIST (Hz)

    Returns
    -------
    result : OptimizeResult (scipy)
    """
    x0 = [5.0, 1.0, 1.0, f_D2 / 1e14, f_D1 / 1e14]
    bounds = [
        (3.0, 20.0),
        (0.01, 500.0),
        (0.01, 500.0),
        (f_D2 / 1e14 - 0.002, f_D2 / 1e14 + 0.002),
        (f_D1 / 1e14 - 0.002, f_D1 / 1e14 + 0.002),
    ]
    result = minimize(
        objective_na, x0,
        args=(f_obs, trans_obs),
        method='Powell',
        bounds=bounds,
        options={'xtol': 1e-8, 'ftol': 1e-12, 'maxiter': 10000},
    )
    return result


def objective_N(params, f_obs, trans_obs, lines_info):
    """
    Fonction objectif L² normalisée pour N raies.
    params[0]  = T_atm / 1000
    params[1:] = concentrations / 1e10
    """
    T_atm = params[0] * 1000.0
    concentrations = params[1:] * 1e10
    trans_sim = forward_model_N(f_obs, T_atm, concentrations, lines_info)
    return loss_l2_normalized(f_obs, trans_obs, trans_sim)


def upper_envelope(f_obs, spectrum_obs, n_bins=200, percentile=99):
    """
    Estime l'enveloppe supérieure (continuum) par fenêtres fréquentielles uniformes.

    Divise le spectre en n_bins intervalles de fréquence et prend le percentile
    supérieur dans chaque bin. Garantit une couverture uniforme sur toute la gamme
    spectrale, même dans les régions très absorbées où aucun point ne dépasse un
    seuil fixe.

    Parameters
    ----------
    f_obs      : array   Fréquences (Hz), triées
    spectrum_obs : array Spectre observé
    n_bins     : int     Nombre de fenêtres (défaut 200)
    percentile : float   Percentile à prendre dans chaque bin (défaut 99)

    Returns
    -------
    f_env  : array (n_bins,)   Fréquences centrales des bins
    s_env  : array (n_bins,)   Valeurs de l'enveloppe
    """
    edges = np.linspace(f_obs.min(), f_obs.max(), n_bins + 1)
    f_env, s_env = [], []
    for i in range(n_bins):
        mask = (f_obs >= edges[i]) & (f_obs < edges[i + 1])
        if mask.sum() == 0:
            continue
        f_env.append(0.5 * (edges[i] + edges[i + 1]))
        s_env.append(np.percentile(spectrum_obs[mask], percentile))
    return np.array(f_env), np.array(s_env)


def fit_planck_continuum(f_obs, spectrum_obs, n_bins=200, percentile=99):
    """
    Ajuste K et T_surface sur l'enveloppe supérieure du spectre.

    Utilise une estimation par fenêtres glissantes (upper_envelope) pour
    exploiter la pente de Planck sur toute la gamme disponible, y compris
    les régions fortement absorbées.

    Parameters
    ----------
    f_obs        : array   Fréquences (Hz)
    spectrum_obs : array   Spectre observé
    n_bins       : int     Nombre de bins pour l'enveloppe (défaut 200)
    percentile   : float   Percentile par bin (défaut 99)

    Returns
    -------
    K         : float   Facteur de normalisation optimal
    T_surface : float   Température de surface optimale (K)
    f_env     : array   Fréquences de l'enveloppe utilisée
    s_env     : array   Valeurs de l'enveloppe utilisée
    """
    f_env, s_env = upper_envelope(f_obs, spectrum_obs, n_bins=n_bins, percentile=percentile)

    def obj_planck(params):
        K = params[0]
        T = params[1] * 1000.0
        B = planck_function(f_env, T)
        # Loss L² avec poids uniformes (un point par bin → couverture uniforme)
        return np.mean((s_env - K * B) ** 2)

    # Initialisation : K tel que K·B(f_median) ≈ median(s_env)
    f_med = np.median(f_env)
    B_med = planck_function(f_med, 5778.0)
    K0 = np.median(s_env) / B_med

    x0 = [K0, 5.778]
    bounds = [(K0 * 1e-3, K0 * 1e3), (3.0, 8.0)]

    result = minimize(obj_planck, x0, method='Powell', bounds=bounds,
                      options={'xtol': 1e-12, 'ftol': 1e-16, 'maxiter': 50000})

    K_opt = result.x[0]
    T_opt = result.x[1] * 1000.0
    return K_opt, T_opt, f_env, s_env


def optimize_full_3stage(f_obs, spectrum_obs, selected_lines,
                         n_bins=200, percentile=99,
                         K_init=None, T_surface_init=None, T_atm_init=5000):
    """
    Optimisation physique complète en 3 passes :

      Passe 1 — Continuum  : ajuste K et T_surface (sautée si K_init et T_surface_init fournis)
      Passe 2 — Absorption : fixe K et T_surface, ajuste T_atm + concentrations
      Passe 3 — Global     : affine tous les paramètres simultanément

    Le modèle complet est :
        S(f) = K · B(f, T_surface) · exp(−Σ τ_i(f, T_atm))

    Parameters
    ----------
    f_obs          : array   Fréquences (Hz)
    spectrum_obs   : array   Spectre observé
    selected_lines : list    Sortie de select_deepest_lines
    n_bins         : int     Bins pour l'enveloppe (passe 1 seulement, défaut 200)
    percentile     : float   Percentile par bin (passe 1 seulement, défaut 99)
    K_init         : float   (optionnel) Facteur de normalisation initial — saute passe 1 si fourni
    T_surface_init : float   (optionnel) Température de surface initiale (K) — saute passe 1 si fourni
    T_atm_init     : float   Température atmosphère initiale pour la passe 2 (K)

    Returns
    -------
    result : dict avec les clés :
        'K'         — facteur de normalisation
        'T_surface' — température de surface (K)
        'T_atm'     — température atmosphère (K)
        'c_opt'     — concentrations optimales (N,)
        'spectrum'  — spectre modélisé final
        'residuals' — résidus
        'success'   — bool
        'loss'      — valeur finale de la loss
    """
    N = len(selected_lines)

    # ── Passe 1 : corps noir sur l'enveloppe supérieure (optionnelle) ────────
    if K_init is not None and T_surface_init is not None:
        print("Passe 1 : continuum fourni externalement, passe ignorée.")
        print(f"  K         = {K_init:.4e}")
        print(f"  T_surface = {T_surface_init:.0f} K")
    else:
        print("Passe 1 : ajustement du continuum (corps noir, enveloppe supérieure)...")
        K_init, T_surface_init, _, _ = fit_planck_continuum(
            f_obs, spectrum_obs, n_bins=n_bins, percentile=percentile)
        print(f"  K         = {K_init:.4e}")
        print(f"  T_surface = {T_surface_init:.0f} K  (attendu ~5778 K)")

    # ── Passe 2 : absorption sur le spectre normalisé ─────────────────────
    # L'atlas IAG est déjà normalisé (~0–1). On ajuste directement exp(-Σ τ_i)
    # contre spectrum_obs sans diviser par K·B (ce qui introduirait la pente
    # Planck et rendrait le modèle incapable de suivre un spectre plat).
    print("\nPasse 2 : ajustement des absorptions (T_atm + concentrations)...")

    # Borne T_atm : permet jusqu'à 20 000 K, clippe l'init si hors bornes
    T_ATM_MAX = 20.0   # en milliers de K
    T_atm_init_scaled = np.clip(T_atm_init / 1000.0, 3.0, T_ATM_MAX - 0.001)

    # Optimisations locales raie par raie
    c_init = np.ones(N)
    delta_local = 1.5e12
    for i, line in enumerate(selected_lines):
        f0 = line['f0_hz']
        mask = (f_obs > f0 - delta_local) & (f_obs < f0 + delta_local)
        if mask.sum() < 5:
            continue

        def obj_local(c_scaled, f0=f0, mask=mask, line=line):
            tau = c_scaled[0] * 1e10 * doppler_profile(f_obs[mask], f0, T_atm_init, line['mass'])
            sim = np.exp(-tau)
            return np.trapz((spectrum_obs[mask] - sim) ** 2, x=f_obs[mask])

        res = minimize(obj_local, [c_init[i]], method='Powell',
                       bounds=[(0.001, 1000)],
                       options={'xtol': 1e-8, 'ftol': 1e-12})
        c_init[i] = res.x[0]

    # Optimisation globale absorption (T_atm + c_i)
    def obj_abs(params):
        T_atm = params[0] * 1000.0
        if T_atm <= 0:
            return 1e10
        conc = params[1:] * 1e10
        sim = forward_model_N(f_obs, T_atm, conc, selected_lines)
        val = loss_l2_normalized(f_obs, spectrum_obs, sim)
        return val if np.isfinite(val) else 1e10

    x0_abs = np.concatenate([[T_atm_init_scaled], c_init])
    bounds_abs = [(3.0, T_ATM_MAX)] + [(0.001, 1000.0)] * N
    res_abs = minimize(obj_abs, x0_abs, method='Powell', bounds=bounds_abs,
                       options={'xtol': 1e-8, 'ftol': 1e-14, 'maxiter': 100000})

    T_atm_opt = res_abs.x[0] * 1000.0
    c_opt     = res_abs.x[1:] * 1e10

    # Spectre modélisé : transmittance pure (cohérent avec l'atlas normalisé)
    spectrum_final = forward_model_N(f_obs, T_atm_opt, c_opt, selected_lines)
    spectrum_final = np.nan_to_num(spectrum_final, nan=1.0, posinf=1.0, neginf=0.0)
    residuals = spectrum_obs - spectrum_final

    print(f"\n{'─'*50}")
    print(f"  Succès      : {res_abs.success}")
    print(f"  Loss finale : {res_abs.fun:.4e}")
    print(f"  K           = {K_init:.4e}  (fixé, §2)")
    print(f"  T_surface   = {T_surface_init:.0f} K  (fixé, §2)")
    print(f"  T_atm       = {T_atm_opt:.0f} K")
    print(f"  RMS résidus = {np.sqrt(np.mean(residuals**2)):.4f}")

    return {
        'K':         K_init,
        'T_surface': T_surface_init,
        'T_atm':     T_atm_opt,
        'c_opt':     c_opt,
        'spectrum':  spectrum_final,   # transmittance modélisée (0–1)
        'residuals': residuals,
        'success':   res_abs.success,
        'loss':      res_abs.fun,
    }


def optimize_local_then_global(f_obs, trans_obs, selected_lines, T_init=5000):
    """
    Optimisation en 2 passes :
      1. Optimisation locale : ajuste chaque concentration sur une fenêtre ±1.5e12 Hz
      2. Optimisation globale : affine tous les paramètres ensemble

    Parameters
    ----------
    f_obs          : array   Fréquences (Hz)
    trans_obs      : array   Transmittance observée
    selected_lines : list    Sortie de select_deepest_lines
    T_init         : float   Température initiale (K)

    Returns
    -------
    result : OptimizeResult (scipy)
    """
    N = len(selected_lines)
    c_init = np.ones(N)
    delta_local = 1.5e12

    print("Passe 1 : optimisations locales...")
    for i, line in enumerate(selected_lines):
        f0 = line['f0_hz']
        mask = (f_obs > f0 - delta_local) & (f_obs < f0 + delta_local)
        if mask.sum() < 5:
            continue

        def obj_local(c_scaled, f0=f0, mask=mask, line=line):
            tau = c_scaled[0] * 1e10 * doppler_profile(f_obs[mask], f0, T_init, line['mass'])
            sim = np.exp(-tau)
            return np.trapz((trans_obs[mask] - sim)**2, x=f_obs[mask])

        res = minimize(obj_local, [c_init[i]], method='Powell',
                       bounds=[(0.001, 1000)],
                       options={'xtol': 1e-8, 'ftol': 1e-12})
        c_init[i] = res.x[0]
        depth_fit = 1 - np.exp(-c_init[i] * 1e10 * doppler_profile(f0, f0, T_init, line['mass']))
        print(f"  {i+1:2d}. {line['species']:6s} λ={line['wl_nm']:.2f} nm  "
              f"c={c_init[i]*1e10:.3e}  depth_fit={depth_fit:.3f}")

    print("\nPasse 2 : optimisation globale...")
    x0 = np.concatenate([[T_init / 1000.0], c_init])
    bounds = [(3.0, 8.0)] + [(0.001, 1000.0)] * N

    result = minimize(
        objective_N, x0,
        args=(f_obs, trans_obs, selected_lines),
        method='Powell',
        bounds=bounds,
        options={'xtol': 1e-8, 'ftol': 1e-14, 'maxiter': 100000},
    )
    return result
