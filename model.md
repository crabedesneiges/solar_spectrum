# Forward Model of the Solar Spectrum

This document describes the forward model used to simulate the solar spectrum in this project.  
The goal of the model is to link a small number of physical parameters to an observed spectrum.

---

## 1. Modeling Philosophy

The model is intentionally simple and based on basic physical assumptions:

- The solar surface is modeled as a blackbody.
- Absorption lines are produced by atoms in the solar atmosphere.
- The width of spectral lines is caused by thermal Doppler broadening.
- All calculations are performed in frequency (or wavenumber) space.

---

## 2. Solar Surface Emission

### 2.1 Blackbody Approximation

The solar surface emission is modeled as blackbody radiation characterized by an effective temperature $T_{\text{surf}}$.

The spectral radiance is given by Planck’s law in frequency space:

$$
B_\nu(T) = \frac{2 h \nu^3}{c^2} \frac{1}{\exp\left(\frac{h\nu}{kT}\right) - 1}
$$

This provides the smooth spectral envelope of the solar spectrum, without any absorption features.

### 2.2 Choice of Surface Temperature

The surface temperature is set to:

$$
T_{\text{surf}} = 5770\ \text{K}
$$

This value corresponds to the effective temperature of the Sun, defined as the temperature of a blackbody emitting the same total radiative power as the Sun.  
It is a standard astrophysical reference value.

---

## 3. Atmospheric Absorption

### 3.1 Absorption Model

Absorption by the solar atmosphere is modeled using a multiplicative attenuation term applied to the surface emission:

$$
I(\nu) = B_\nu(T_{\text{surf}})\,\exp[-\tau(\nu)]
$$

where $ \tau(\nu) $ is the optical depth.

For a given atomic species, the optical depth is modeled as:

$$
\tau(\nu) = c \, \phi(\nu)
$$

where:
- $c$ is an effective absorption strength parameter,
- $\phi(\nu)$ is the normalized spectral line profile.

---

## 4. Doppler Broadening of Spectral Lines

### 4.1 Physical Origin

Atoms in the solar atmosphere have a thermal velocity distribution.  
Their motion induces Doppler shifts of the absorbed frequencies, leading to a broadening of spectral lines.

This effect produces a **Gaussian line profile in frequency space**.

### 4.2 Doppler Line Profile

The Doppler-broadened line profile is given by:

$
\phi(\nu) =
\frac{1}{\Delta\nu_D \sqrt{\pi}}
\exp\left[-\left(\frac{\nu - \nu_0}{\Delta\nu_D}\right)^2\right]
$

where:
- $\nu_0$ is the central frequency of the transition,
- $\Delta\nu_D$ is the Doppler width.

### 4.3 Doppler Width

The Doppler width is expressed as:

$
\Delta\nu_D =
\frac{\nu_0}{c}
\sqrt{\frac{2 k T_{\text{atm}}}{m}}
$

where:
- $T_{\text{atm}}$ is the atmospheric temperature,
- $m$ is the atomic mass of the absorbing species.

The Doppler width increases with temperature and decreases with atomic mass.

---

## 5. Choice of Atmospheric Temperature

The solar atmosphere is stratified and does not have a single temperature.  
Visible absorption lines (such as sodium D lines) are primarily formed in the photosphere, where temperatures range approximately from 4500 K to 5800 K.

A characteristic atmospheric temperature is therefore chosen as:

$$
T_{\text{atm}} \approx 5000\ \text{K}
$$

This value represents an effective temperature of the line-forming region and serves as a tunable parameter in the model.

---

## 6. Complete Forward Model

Combining surface emission and atmospheric absorption, the final forward model is:

$$
S(\nu) =
B_\nu(T_{\text{surf}})
\prod_i \exp\left[-c_i \, \phi_i(\nu)\right]
$$

where the product runs over the considered atomic species.

