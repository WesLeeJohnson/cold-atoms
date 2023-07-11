import numpy as np
import coldatoms_lib


def _harmonic_trap_forces_ref(positions, q, kx, ky, kz, phi, dt, f):
    cphi = np.cos(phi)
    sphi = np.sin(phi)

    x = positions[:, 0]
    y = positions[:, 1]
    z = positions[:, 2]

    f[:, 0] += dt * q * (
        (-kx * cphi * cphi - ky * sphi * sphi) * x + cphi * sphi * (ky - kx) * y)
    f[:, 1] += dt * q * (
        cphi * sphi * (ky - kx) * x + (-kx * sphi * sphi - ky * cphi * cphi) * y)
    f[:, 2] += -dt * q * kz * z


class HarmonicTrapPotential(object):
    def __init__(self, kx, ky, kz):
        self.kx = kx
        self.ky = ky
        self.kz = kz
        self.phi = 0.0
        self._harmonic_trap_forces = coldatoms_lib.harmonic_trap_forces
        self._harmonic_trap_forces_per_particle_charges = coldatoms_lib.harmonic_trap_forces_per_particle_charge

    def use_reference_implementations(self):
        self._harmonic_trap_forces = _harmonic_trap_forces_ref
        self._harmonic_trap_forces_per_particle_charges = _harmonic_trap_forces_ref

    def force(self, dt, ensemble, f):
        positions = ensemble.x
        if 'charge' in ensemble.ensemble_properties:
            q = ensemble.ensemble_properties['charge']
            self._harmonic_trap_forces(positions, q, self.kx, self.ky, self.kz,
                self.phi, dt, f)
        elif 'charge' in ensemble.particle_properties:
            q = ensemble.particle_properties['charge']
            self._harmonic_trap_forces_per_particle_charges(
                positions, q, self.kx, self.ky, self.kz, self.phi, dt, f)
        else:
            raise RuntimeError('Must provide a charge to compute coulomb force')

class RotatingWallTrapPotential(object):
    """
    This class defines the trap potential. It is used to calculate the
    forces on the ions due to the trap potential. This includes the
    rotating wall potential. 

    Parameters
    ----------
    kz : float
        The axial stiffness.
    delta : float
        The rotating wall strength.
    omega : float
        The rotating wall frequency.
    phi_0 : float
        The initial phase of the rotating wall.

    Methods
    -------
    reset_phase()
        Resets the phase of the rotating wall.
    force(dt, ensemble, f)
        Calculates the force on the ions due to the trap potential.

    Examples
    --------
    >>> trap_potential = TrapPotential(m_Be/q_Be*wz**2, ma.delta, ma.wrot, np.pi / 2.0)
    """
    def __init__(self, kz, delta, omega, phi_0):
        self.kz = kz
        self.kx = -(0.5 + delta) * kz
        self.ky = -(0.5 - delta) * kz
        self.phi_0 = phi_0
        self.phi = phi_0
        self.omega = omega

    def reset_phase(self):
        """
        Resets the phase of the rotating wall.
        """
        self.phi = self.phi_0

    def force(self, dt, ensemble, f):
        """
        Calculates the force on the ions due to the trap potential.

        Parameters
        ----------
        dt : float
            The time step.
        ensemble : Ensemble 
            The ensemble of ions.
        f : ndarray
            The force on the ions.
        """
        self.phi += self.omega * 0.5 * dt

        q = ensemble.ensemble_properties['charge']
        if q is None:
            q = ensemble.particle_properties['charge']
            if q is None:
                raise RuntimeError('Must provide ensemble or per particle charge')

        cphi = np.cos(self.phi)
        sphi = np.sin(self.phi)
        kx = self.kx
        ky = self.ky

        x = ensemble.x[:, 0]
        y = ensemble.x[:, 1]
        z = ensemble.x[:, 2]

        f[:, 0] += dt * q * (
            (-kx * cphi * cphi - ky * sphi * sphi) * x + cphi * sphi * (ky - kx) * y)
        f[:, 1] += dt * q * (
            cphi * sphi * (ky - kx) * x + (-kx * sphi * sphi - ky * cphi * cphi) * y)
        f[:, 2] += -dt * q *self.kz * z

        self.phi += self.omega * 0.5 * dt