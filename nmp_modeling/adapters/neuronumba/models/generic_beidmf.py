# ==========================================================================
# Generic two-state heterogeneous BEI-DMF model for neuronumba
#
# Default behavior is equivalent to the Deco2014 / Deco2018 two-population
# balanced excitation-inhibition dynamic mean-field model.
#
# Heterogeneity can be introduced through:
#   1. Firing-rate gain modulation: M_e, M_i, gain_map_e, gain_map_i
#   2. Local synaptic weights: w_ee, w_ei, w_ie, w_ii
#   3. Static gating parameters: taon, taog, gamma_e, gamma_i
# ==========================================================================
import numpy as np
import numba as nb
from overrides import overrides

from neuronumba.basic.attr import Attr
from neuronumba.fitting.fic.fic import FICHerzog2022
from neuronumba.numba_tools.types import NDA_f8_2d
from neuronumba.numba_tools.config import NUMBA_CACHE, NUMBA_FASTMATH, NUMBA_NOGIL
from neuronumba.simulator.models.model import Model, LinearCouplingModel

ONE = 1.0
EPS = 1e-12

class GenericBEIDMF(LinearCouplingModel):
    _state_var_names = ["S_e", "S_i"]
    _coupling_var_names = ["S_e"]
    _observable_var_names = ["Ie", "re"]
    _state_var_bounds = {"S_e": (0.0, 1.0), "S_i": (0.0, 1.0)}

    auto_fic = Attr(default=False)

    # ----------------------------------------------------------------------
    # Static gating dynamics
    # ----------------------------------------------------------------------
    taon = Attr(default=100.0, attributes=Model.Tag.REGIONAL)      # NMDA characteristic time (ms), E population time constant
    taog = Attr(default=10.0, attributes=Model.Tag.REGIONAL)       # GABA characteristic time (ms), I population time constant
    gamma_e = Attr(default=0.641, attributes=Model.Tag.REGIONAL)   # NMDA kinetic factor, excitatory synaptic efficacy
    gamma_i = Attr(default=1.0, attributes=Model.Tag.REGIONAL)     # GABA kinetic factor, inhibitory synaptic efficacy

    # ----------------------------------------------------------------------
    # External input and baseline current parameters
    # ----------------------------------------------------------------------
    I0 = Attr(default=0.382, attributes=Model.Tag.REGIONAL)        # Overall effective external input (nA)
    Jext_e = Attr(default=1.0, attributes=Model.Tag.REGIONAL)      # External input scaling for E population
    Jext_i = Attr(default=0.7, attributes=Model.Tag.REGIONAL)      # External input scaling for I population
    I_external = Attr(default=0.0, attributes=Model.Tag.REGIONAL)  # Additional external current to E population

    # ----------------------------------------------------------------------
    # Local synaptic weights
    # ----------------------------------------------------------------------
    J_NMDA = Attr(default=0.15, attributes=Model.Tag.REGIONAL)     # NMDA current scale (nA)
    J = Attr(default=1.0, attributes=Model.Tag.REGIONAL)           # FIC inhibitory feedback strength

    w_ee = Attr(default=1.4, attributes=Model.Tag.REGIONAL)        # E-to-E recurrent excitation
    w_ei = Attr(default=1.0, attributes=Model.Tag.REGIONAL)        # E-to-I recruitment
    w_ie = Attr(default=1.0, attributes=Model.Tag.REGIONAL)        # I-to-E feedback inhibition scale
    w_ii = Attr(default=1.0, attributes=Model.Tag.REGIONAL)        # I self-inhibition scale

    # ----------------------------------------------------------------------
    # Firing-rate transfer function parameters
    # ----------------------------------------------------------------------
    ae = Attr(default=310.0, attributes=Model.Tag.REGIONAL)
    be = Attr(default=125.0, attributes=Model.Tag.REGIONAL)
    de = Attr(default=0.16 , attributes=Model.Tag.REGIONAL)

    ai = Attr(default=615.0, attributes=Model.Tag.REGIONAL)
    bi = Attr(default=177.0, attributes=Model.Tag.REGIONAL)
    di = Attr(default=0.087, attributes=Model.Tag.REGIONAL)

    # ----------------------------------------------------------------------
    # Generic gain modulation
    # Effective gains are:
    #   M_e_eff = M_e * (1 + gain_e * gain_map_e)
    #   M_i_eff = M_i * (1 + gain_i * gain_map_i)
    # ----------------------------------------------------------------------
    M_e = Attr(default=1.0, attributes=Model.Tag.REGIONAL)
    M_i = Attr(default=1.0, attributes=Model.Tag.REGIONAL)

    gain_map_e = Attr(default=0.0, attributes=Model.Tag.REGIONAL)
    gain_map_i = Attr(default=0.0, attributes=Model.Tag.REGIONAL)
    gain_e = Attr(default=0.0, attributes=Model.Tag.REGIONAL)
    gain_i = Attr(default=0.0, attributes=Model.Tag.REGIONAL)

    @overrides
    def _init_dependant(self):
        super()._init_dependant()
        if self.auto_fic and not self._attr_defined("J"):
            self.J = FICHerzog2022().compute_J(self.weights, self.g)

    def initial_state(self, n_rois):
        state = np.empty((GenericBEIDMF.n_state_vars, n_rois))
        state[0] = 0.001
        state[1] = 0.001
        return state

    def get_numba_dfun(self):
        m = self.m.copy()
        P = self.P

        @nb.njit(
            nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]),
            cache=NUMBA_CACHE,
            fastmath=NUMBA_FASTMATH,
            nogil=NUMBA_NOGIL,
        )
        def GenericBEIDMF_dfun(state: NDA_f8_2d, coupling: NDA_f8_2d):
            Se = state[0, :]
            Si = state[1, :]

            J_NMDA = m[np.intp(P.J_NMDA)]

            Ie = (
                m[np.intp(P.Jext_e)] * m[np.intp(P.I0)]
                + m[np.intp(P.w_ee)] * J_NMDA * Se
                - m[np.intp(P.w_ie)] * m[np.intp(P.J)] * Si
                + J_NMDA * coupling[0, :]
                + m[np.intp(P.I_external)]
            )

            Ii = (
                m[np.intp(P.Jext_i)] * m[np.intp(P.I0)]
                + m[np.intp(P.w_ei)] * J_NMDA * Se
                - m[np.intp(P.w_ii)] * Si
            )

            M_e_eff = m[np.intp(P.M_e)] * (
                ONE + m[np.intp(P.gain_e)] * m[np.intp(P.gain_map_e)]
            )
            M_i_eff = m[np.intp(P.M_i)] * (
                ONE + m[np.intp(P.gain_i)] * m[np.intp(P.gain_map_i)]
            )

            y_e = M_e_eff * (m[np.intp(P.ae)] * Ie - m[np.intp(P.be)])
            den_e = ONE - np.exp(-m[np.intp(P.de)] * y_e)
            re = np.where(np.abs(den_e) < EPS, ONE / m[np.intp(P.de)], y_e / den_e)

            y_i = M_i_eff * (m[np.intp(P.ai)] * Ii - m[np.intp(P.bi)])
            den_i = ONE - np.exp(-m[np.intp(P.di)] * y_i)
            ri = np.where(np.abs(den_i) < EPS, ONE / m[np.intp(P.di)], y_i / den_i)

            dSe = (
                -Se / m[np.intp(P.taon)]
                + m[np.intp(P.gamma_e)] * (ONE - Se) * re / 1000.0
            )

            dSi = (
                -Si / m[np.intp(P.taog)]
                + m[np.intp(P.gamma_i)] * ri / 1000.0
            )

            return np.stack((dSe, dSi)), np.stack((Ie, re))

        return GenericBEIDMF_dfun
