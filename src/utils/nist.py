import numpy as np
import astropy.units as u
import scipy.constants as const
from astroquery.nist import Nist


class NISTQuery:
    """Interface simplifiée pour interroger la base de données NIST."""

    def get_lines(self, species, wavelength_range_nm):
        """
        Interroge NIST pour une espèce sur une plage de longueurs d'onde.

        Parameters
        ----------
        species              : str    Ex: "Na I", "Fe I"
        wavelength_range_nm  : tuple  (min_nm, max_nm)

        Returns
        -------
        lines : astropy Table
        """
        min_wav, max_wav = wavelength_range_nm
        lines = Nist.query(min_wav * u.nm, max_wav * u.nm, linename=species)
        return lines

    def list_wavelengths(self, lines):
        return lines["Observed"]


def get_all_nist_lines(species_list, wl_min_nm, wl_max_nm):
    """
    Récupère les raies NIST pour une liste d'espèces atomiques.

    Parameters
    ----------
    species_list : list of (str, float)   Ex: [("Na I", 22.99), ("Fe I", 55.85)]
    wl_min_nm    : float   Longueur d'onde minimale (nm)
    wl_max_nm    : float   Longueur d'onde maximale (nm)

    Returns
    -------
    all_lines : list of dict
        Chaque dict contient : 'species', 'wl_nm', 'f0_hz', 'mass'
    """
    all_lines = []
    for species, mass_amu in species_list:
        try:
            lines = Nist.query(wl_min_nm * u.nm, wl_max_nm * u.nm, linename=species)
            wls = np.array(lines["Observed"].data, dtype=float)
            wls = wls[~np.isnan(wls) & (wls > 0)]
            for wl in wls:
                all_lines.append({
                    'species': species,
                    'wl_nm': wl,
                    'f0_hz': const.c / (wl * 1e-9),
                    'mass': mass_amu,
                })
            print(f"  {species}: {len(wls)} raies trouvées")
        except Exception as e:
            print(f"  {species}: erreur ({e})")

    all_lines.sort(key=lambda x: x['f0_hz'])
    return all_lines


def select_deepest_lines(f_obs, trans_obs, all_lines, N, f_tolerance=1e11):
    """
    Sélectionne les N raies les plus profondes dans le spectre observé.

    Parameters
    ----------
    f_obs       : array   Fréquences observées (Hz)
    trans_obs   : array   Transmittance observée
    all_lines   : list    Sortie de get_all_nist_lines
    N           : int     Nombre de raies à sélectionner
    f_tolerance : float   Fenêtre en Hz autour de chaque raie (défaut : 1e11 Hz)

    Returns
    -------
    selected : list of dict (les N raies triées par profondeur décroissante)
    """
    scored = []
    used_f_centers = []

    for line in all_lines:
        f0 = line['f0_hz']

        if any(abs(f0 - fc) < 3e11 for fc in used_f_centers):
            continue

        mask = (f_obs > f0 - f_tolerance) & (f_obs < f0 + f_tolerance)
        if mask.sum() < 3:
            continue

        depth = 1.0 - np.min(trans_obs[mask])
        if depth < 0.02:
            continue

        scored.append({**line, 'depth': depth})
        used_f_centers.append(f0)

    scored.sort(key=lambda x: x['depth'], reverse=True)
    selected = scored[:N]

    print(f"\nTop {N} raies sélectionnées :")
    for i, s in enumerate(selected):
        print(f"  {i+1:2d}. {s['species']:6s}  "
              f"λ={s['wl_nm']:.3f} nm  profondeur={s['depth']:.3f}")
    return selected
