import numpy as np
import coldatoms_lib
import random

class RadiationPressure(object):
    """
    The force experienced by level atoms undergoing resonance fluorescence.

    This class computes the radiation pressure force, both determinstic and
    fluctuating recoil components, on an atom driven by a monochromatic laser
    field. The class handles spatial variations in laser intensity and of the
    atomic transition frequency as in a Zeeman slower. The wavevector of the
    laser field is assumed constant. RadiationPressure does not deal with
    attenuation of the laser field and is therefore limited to optically thin
    samples.
    An important application of RadiationPressure is to the simulation of
    Doppler cooling of atoms by combining two red detuned lasers with opposite
    propagation directions. In the context of laser cooling we are limited to
    situations of low total saturation and we cannot handle sub-Doppler
    cooling.

    Parameters
    ----------
    gamma : float
        The atomic decay rate (2\pi / excited state lifetime).
    hbar_k : float
        The recoil momentum of a single photon.
    intensity : object
        An object with a method 'intensities' that takes an array of atomic
        positions and returns an array of intensities at those positions.
    detuning : object
        An object with a method 'detunings' that takes an array of atomic
        positions and an array of atomic velocities and returns an array of
        detunings at those positions and velocities. Red detuning is
        negative and blue detuning is positive.
    seed : int
        A seed for the random number generator used to generate the
        stochastic recoil force. If None, a random seed is generated.

    Attributes
    ----------
    gamma : float
        The atomic decay rate (2\pi / excited state lifetime).
    hbar_k : float
        The recoil momentum of a single photon.
    intensity : object
        An object with a method 'intensities' that takes an array of atomic
        positions and returns an array of intensities at those positions.
    detuning : object
        An object with a method 'detunings' that takes an array of atomic
        positions and an array of atomic velocities and returns an array of
        detunings at those positions and velocities. Red detuning is
        negative and blue detuning is positive.
    rng_context : object
        A context object for the random number generator used to generate the
        stochastic recoil force.

    Methods
    -------
    force(dt, ensemble, f)  
        Compute the radiation pressure force on an ensemble of atoms.
    """

    def __init__(self, gamma, hbar_k, intensity, detuning,seed=None):
        self.gamma = gamma
        self.hbar_k = np.copy(hbar_k)
        self.intensity = intensity
        self.detuning = detuning
        if seed is None:
            seed = random.randint(10,10**6+10)
        rng = coldatoms_lib.rng
        rng.seed(seed)
        self.rng_context = rng.context()

    def force(self, dt, ensemble, f):
        s_of_r = self.intensity.intensities(ensemble.x)
        deltas = self.detuning.detunings(ensemble.x, ensemble.v)
        nbars = np.zeros_like(deltas)
        coldatoms_lib.compute_nbars(dt, self.gamma, s_of_r, deltas, nbars)
        coldatoms_lib.add_radiation_pressure(self.rng_context, self.hbar_k, nbars, f)

