from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
from aep8 import flux
from astropy import units as u
from astropy.constants import c, m_e, m_p
from astropy.table import Table
from scipy.interpolate import CubicSpline

from ._electron_loss import get_electron_energy_loss
from ._refraction_index import get_refraction_index


@dataclass
class RadiationBelt:
    energy: Optional[list[u.Quantity]] = (0.05 * u.MeV, 8.5 * u.MeV)
    nbins: int = 20
    particle: Literal["e", "p"] = "e"
    solar: Literal["max", "min"] = "max"

    def flux_table(self, observer_location, obstime) -> Table:
        emin, emax = self.energy
        energy_bins = np.geomspace(emin, emax, num=self.nbins)
        flux_integral = [
            flux(
                observer_location,
                obstime,
                e,
                kind="integral",
                solar=self.solar,
                particle=self.particle,
            )
            for e in energy_bins
        ]
        return Table([energy_bins, u.Quantity(flux_integral)], names=["energy", "flux"])


@dataclass
class CerenkovEmission:
    material: str = "si02_suprasil_2a"
    particle: Literal["e", "p"] = "e"
    solar: Literal["max", "min"] = "max"
    energy: Optional[list[u.Quantity]] = (0.05 * u.MeV, 8.5 * u.MeV)

    def emission(self, observer_location, obstime) -> tuple[u.Quantity, u.Quantity]:
        # --- Material optical parameters: index of refraction and density
        material_properties = {
            "silica": (1.5, 2.2),
            "sio2": (1.5, 2.2),
            "si02_suprasil_2a": (1.5, 2.2),
            "sapphire": (1.75, 4.0),
        }
        if self.material not in material_properties:
            raise ValueError(f"Unknown material option: '{self.material}'")

        n_val, rho = material_properties[self.material]

        # Retrieve electron flux data (AE8 model)
        rb = RadiationBelt(energy=self.energy, particle=self.particle, solar=self.solar)
        flux_data = rb.flux_table(observer_location, obstime)

        # Energy grid for interpolation (MeV)
        ee = np.logspace(np.log10(0.04), np.log10(8.0), 1000) * u.MeV
        cs_flux = CubicSpline(
            flux_data["energy"].value,
            flux_data["flux"].value,
            bc_type="natural",
            extrapolate=True,
        )
        Fe = cs_flux(ee.value)

        # Compute midpoint energies between bins
        em = 0.5 * (ee[:-1] + ee[1:])

        # Get rest mass of particle and convert to MeV
        mass = m_e if self.particle == "e" else m_p
        mass_mev = (mass * c**2).to("MeV").value

        # Compute Lorentz gamma and beta (v/c)
        gamma = 1 + em.value / mass_mev
        beta = np.sqrt(1 - 1.0 / gamma**2)

        # gamma_full = 1 + ee.value / mass_mev
        # beta_full  = np.sqrt(1 - 1.0 / gamma_full**2)

        # Interpolate flux at midpoints
        cs_fm = CubicSpline(ee.value, Fe, bc_type="natural", extrapolate=True)
        Fm = cs_fm(em.value)

        # Cerenkov emission condition: n * beta > 1
        fC = np.maximum(0, 1 - 1.0 / n_val**2 / beta**2)

        # Get inverse energy loss (1/dE/dX) from stopping power
        Ek, dEdX = get_electron_energy_loss(self.material)
        cs_dEdX = CubicSpline(Ek, 1.0 / dEdX, bc_type="natural", extrapolate=True)
        gEE = cs_dEdX(em.value)

        # Cerenkov emission integrand: flux × path length × emission factor
        intg = gEE * Fm * fC
        cs_intg = CubicSpline(em.value, intg, bc_type="natural", extrapolate=True)

        # # Compute Cerenkov angle: theta = arccos(1 / n / beta)
        # the_arg = np.clip(1 / n_val / beta_full, -1.0, 1.0)
        # theta = np.arccos(the_arg)
        # # FlagReal = np.imag(theta) == 0

        # # Normalize cumulative Cerenkov emission
        # cumulative_integral = np.concatenate(([0], np.cumsum(intg * np.diff(ee.value))))
        # Cint = cumulative_integral / cumulative_integral[-1]

        # # Angular distribution in theta
        # theta_max = np.arccos(1.0 / n_val)
        # th = theta_max * 10 ** np.arange(-2, 0.01, 0.01)
        # thm = 0.5 * (th[:-1] + th[1:])

        # # Map theta bins to energies
        # beta_theta = 1.0 / n_val / np.cos(thm)
        # gamma_theta = 1.0 / np.sqrt(1 - beta_theta**2)
        # eth = mass_mev * (gamma_theta - 1)

        # # Rescale stopping power and flux
        # gEE_eth = cs_dEdX(eth)
        # cs_gEE = CubicSpline(eth, gEE_eth, bc_type="natural", extrapolate=True)

        # # Angular Cerenkov intensity distribution
        # int_qC = (
        #     cs_fm(eth) / cs_fm(1)
        #     * (gamma_theta * beta_theta) ** 3
        #     * np.sin(thm) ** 3
        #     * gEE_eth / cs_gEE(1)
        # )

        # # Cumulative angular intensity
        # qCth = int_qC * np.diff(th)
        # qCth_padded = np.concatenate(([0], qCth))
        # qCth_padded_safe = np.where(qCth_padded == 0, np.nan, qCth_padded)
        # ICth = np.concatenate(([0], np.cumsum(qCth))) / qCth_padded_safe

        # Wavelength-dependent refractive index and emission
        Lam, n, _ = get_refraction_index(self.material)
        Nn = len(n)
        IC1mu = np.full(Nn, np.nan)

        for i in range(Nn):
            gamma_i = 1 + em.value / mass_mev
            beta_i = np.sqrt(1 - 1.0 / gamma_i**2)
            fCi = np.maximum(0, 1 - 1.0 / n[i] ** 2 / beta_i**2)
            intg_i = gEE * Fm * fCi
            Lnorm = 2 * np.pi / 137 / rho / (1e-8 * Lam[i]) ** 2 * cs_intg(1)
            Lnorm = Lnorm * 1e-4  # count/cm^2/s/micron
            int_val = np.sum(intg_i * np.diff(ee.value)) / cs_intg(1)
            L1mu = int_val * Lnorm
            IC1mu[i] = L1mu / (2 * np.pi * n[i] ** 2)

        # Convert intensity to arcseconds squared
        intensity_micron = IC1mu * u.count / u.cm**2 / u.s / u.sr / u.micron
        wavelength = Lam * u.Angstrom

        # freq =   (c / (wavelength)).to(u.Hz)

        # Photon energy and frequency
        # photon_energy = (h * c / (wavelength)).to(u.erg)

        # 1  mJy (milliJansky ) = 1e-26 * erg / s / cm**2 / Hz
        # flux_mJy = (
        #     intensity_arcsec2
        #     * h.to(u.erg * u.s)
        #     * freq
        #     * (c.to(u.Angstrom / u.s) / freq**2)
        #     / u.mJy.to(u.erg / u.s / u.cm**2 / u.Hz)
        # )
        # intensity_flux_arcsec2 = flux_mJy / ((c.to(u.Angstrom / u.s) / (freq**2)) / u.mJy.to(u.erg / u.s / u.cm**2 / u.Hz) )  # Flux per energy
        return wavelength, intensity_micron


@dataclass
class CerenkovBackground:
    """
    Model for Cerenkov particle-induced background using the AEP8 flux model.

    This model estimates the background induced by Cerenkov radiation from charged particles (electrons or protons)
    interacting with telescope optics. It uses integral flux data from the NASA AE8/AP8 radiation belt model
    to calculate the intensity of radiation emitted by typical optical materials such as silica or suprasil,
    used in the ULTRASAT telescope lenses.

    Parameters
    ----------
    particle : {'e', 'p'}, optional
        Particle type ('e' for electrons, 'p' for protons), by default 'e'.
    solar : {'max', 'min'}, optional
        Solar activity condition, by default 'max'.
    material : str, optional
        Material type for optics, by default 'si02_suprasil_2a'.

    Notes
    -----
    The underlying radiation belt flux data is obtained using the :doc:`NASA AE8/AP8 model from IRBEM <irbem:api/radiation_models>`.
    constraining the flux of charged particles in Earth's radiation belts.

    References
    ----------
    This module is a Python adaptation of the MATLAB `Cerenkov` function from the MAATv2 package:
    https://www.mathworks.com/matlabcentral/fileexchange/128984-astropack-maatv2

    Examples
    --------
    >>> from astropy.coordinates import EarthLocation
    >>> from astropy.time import Time
    >>> from astropy import units as u
    >>> from m4opt.synphot.background import CerenkovBackground
    >>> observer_location = EarthLocation.from_geodetic(35 * u.deg, -20 * u.deg, 300 * u.km)
    >>> obstime = Time("2024-08-17T00:41:04Z")
    >>> cerenkov_model = CerenkovBackground(particle='e', solar='max', material='si02_suprasil_2a')
    >>> wavelength, intensity = cerenkov_model.cerenkov_emission(observer_location, obstime)
    >>> print(wavelength, intensity)
    """

    particle: Literal["e", "p"] = "e"
    solar: Literal["max", "min"] = "max"
    material: str = "si02_suprasil_2a"

    def radiation_belt(self, observer_location, obstime) -> Table:
        """
        Returns a table of integral flux values from Earth's radiation belts.

        Parameters
        ----------
        observer_location : astropy.coordinates.EarthLocation
            Observer's geographic location.
        obstime : astropy.time.Time
            Time of observation.

        Returns
        -------
        astropy.table.Table
            Table of energy bins and corresponding integral flux values.
        """
        rb = RadiationBelt(particle=self.particle, solar=self.solar)
        return rb.flux_table(observer_location, obstime)

    def cerenkov_emission(
        self, observer_location, obstime
    ) -> tuple[u.Quantity, u.Quantity]:
        """
        Computes the wavelength-dependent Cerenkov emission intensity.

        Parameters
        ----------
        observer_location : astropy.coordinates.EarthLocation
            Observer's geographic location.
        obstime : astropy.time.Time
            Time of observation.

        Returns
        -------
        wavelength : astropy.units.Quantity
            Wavelength array of the emitted radiation.
        intensity : astropy.units.Quantity
            Corresponding intensity per wavelength.
        """
        emission_model = CerenkovEmission(
            material=self.material, particle=self.particle, solar=self.solar
        )
        return emission_model.emission(observer_location, obstime)


# from astropy.coordinates import EarthLocation
# from astropy.time import Time
# from astropy import units as u
# observer_location = EarthLocation.from_geodetic(lon=15 * u.deg,     lat=0 * u.deg,     height=35786 * u.km )

# obstime = Time("2024-01-01T00:00:00")


# # RadiationBelt
# rb = RadiationBelt(
#     energy=[0.05 * u.MeV, 8.5 * u.MeV],
#     nbins=20,
#     particle="e",
#     solar="max"
# )

# # Get the flux table
# flux_table = rb.flux_table(observer_location, obstime)
# print(flux_table)


# # Create the background model
# cb = CerenkovBackground(
#     particle="e",
#     solar="max",
#     material="si02_suprasil_2a"
# )

# # Call the method to get the emission
# wavelength, intensity = cb.cerenkov_emission(observer_location, obstime)

# # Print results
# print(wavelength)
# print(intensity)
