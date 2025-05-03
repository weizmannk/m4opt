CerenkovBackground Module
=========================

License
-------
GNU General Public License Version 3

Author (original MATLAB version): Eran O. Ofek (Oct 2019)
Python port: this version is an adaptation of the original MATLAB code
Original URL: http://weizmann.ac.il/home/eofek/matlab/

Description
-----------
Python module to calculate the Cherenkov background induced by geostationary electrons
in optical materials like silica or sapphire.

This module is a Python port of the original MATLAB code `Cerenkov.m` from the AstroPack package.

It computes the Cherenkov photon yield, intensity spectrum, and angular distribution
due to the interaction of energetic electrons with optical materials, based on physical
models and AE9 electron flux data.

Dependencies
------------
- ``geostat_electrons_spec_flux``: Electron flux model (AE9)
- ``get_electron_energy_loss``: Electron stopping power
- ``get_refraction_index``: Wavelength-dependent refractive index

Reference (MATLAB version)
--------------------------
https://github.com/EranOfek/AstroPack/blob/main/matlab/astro/%2Bultrasat/Cerenkov.m
