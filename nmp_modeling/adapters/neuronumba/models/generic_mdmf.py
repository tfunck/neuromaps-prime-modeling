# ==========================================================================
# Generic heterogeneous multiscale dynamic mean-field model for neuronumba
#
# Default behavior follows the Naskar et al. 2021 MDMF model:
#   - Deco2014-style excitatory and inhibitory population currents
#   - neurotransmitter-dependent synaptic gating kinetics
#   - local inhibitory plasticity as a dynamic J state
#
# Heterogeneity can be introduced through:
#   1. Neurotransmitter kinetic parameters: t_glu, t_gaba, alpha_e/i, beta_e/i
#   2. Firing-rate gain modulation: M_e, M_i, gain_map_e, gain_map_i
#   3. Local synaptic weights: w_ee, w_ei, w_ie, w_ii
#   4. Plasticity parameters: plasticity_gamma, rho, J_init
# ==========================================================================
import numpy as np
import numba as nb

from neuronumba.basic.attr import Attr
from neuronumba.numba_tools.types import NDA_f8_2d
from neuronumba.numba_tools.config import NUMBA_CACHE, NUMBA_FASTMATH, NUMBA_NOGIL
from neuronumba.simulator.models.model import Model, LinearCouplingModel

ONE = 1.0
EPS = 1e-12
HZ_TO_PER_MS = 1000.0

class GenericMDMF(LinearCouplingModel):
    _state_var_names = ["S_e", "S_i", "J"]
    _coupling_var_names = ["S_e"]
    _observable_var_names = ["Ie", "re"]
    _state_var_bounds = {
        "S_e": (0.0, 1.0),
        "S_i": (0.0, 1.0),
        "J": (0.0, np.inf),
    }

    # ----------------------------------------------------------------------
    # Neurotransmitter-dependent synaptic gating dynamics
    # ----------------------------------------------------------------------
    t_glu = Attr(default=7.46, attributes=Model.Tag.REGIONAL)      # Glutamate concentration
    t_gaba = Attr(default=1.82, attributes=Model.Tag.REGIONAL)     # GABA concentration
    alpha_e = Attr(default=0.072, attributes=Model.Tag.REGIONAL)   # NMDA forward binding rate
    alpha_i = Attr(default=0.53, attributes=Model.Tag.REGIONAL)    # GABA forward binding rate
    beta_e = Attr(default=0.0066, attributes=Model.Tag.REGIONAL)   # NMDA backward/unbinding rate
    beta_i = Attr(default=0.18, attributes=Model.Tag.REGIONAL)     # GABA backward/unbinding rate

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

    w_ee = Attr(default=1.4, attributes=Model.Tag.REGIONAL)        # E-to-E recurrent excitation
    w_ei = Attr(default=1.0, attributes=Model.Tag.REGIONAL)        # E-to-I recruitment
    w_ie = Attr(default=1.0, attributes=Model.Tag.REGIONAL)        # I-to-E feedback inhibition scale
    w_ii = Attr(default=1.0, attributes=Model.Tag.REGIONAL)        # I self-inhibition scale

    # ----------------------------------------------------------------------
    # Dynamic inhibitory plasticity
    # ----------------------------------------------------------------------
    J_init = Attr(default=1.0, attributes=Model.Tag.REGIONAL)      # Initial inhibitory feedback strength
    plasticity_gamma = Attr(default=1.0, attributes=Model.Tag.REGIONAL)
    rho = Attr(default=3.0, attributes=Model.Tag.REGIONAL)         # Target E firing rate

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

    def initial_state(self, n_rois):
        state = np.empty((GenericMDMF.n_state_vars, n_rois))
        state[0] = 0.001
        state[1] = 0.001
        state[2] = self.J_init
        return state

    def get_noise_template(self):
        """Return the default state-variable noise template."""
        return np.r_[1.0, 1.0, 0.0]

    def get_numba_dfun(self):
        m = self.m.copy()
        P = self.P

        @nb.njit(
            nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]),
            cache=NUMBA_CACHE,
            fastmath=NUMBA_FASTMATH,
            nogil=NUMBA_NOGIL,
        )
        def GenericMDMF_dfun(state: NDA_f8_2d, coupling: NDA_f8_2d):
            Se = state[0, :]
            Si = state[1, :]
            J = state[2, :]

            J_NMDA = m[np.intp(P.J_NMDA)]

            Ie = (
                m[np.intp(P.Jext_e)] * m[np.intp(P.I0)]
                + m[np.intp(P.w_ee)] * J_NMDA * Se
                - m[np.intp(P.w_ie)] * J * Si
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
                -m[np.intp(P.beta_e)] * Se
                + m[np.intp(P.alpha_e)] * m[np.intp(P.t_glu)] * (ONE - Se) * re / HZ_TO_PER_MS
            )

            dSi = (
                -m[np.intp(P.beta_i)] * Si
                + m[np.intp(P.alpha_i)] * m[np.intp(P.t_gaba)] * (ONE - Si) * ri / HZ_TO_PER_MS
            )

            dJ = (
                m[np.intp(P.plasticity_gamma)] * (ri / HZ_TO_PER_MS) * ((re - m[np.intp(P.rho)]) / HZ_TO_PER_MS)
            )

            return np.stack((dSe, dSi, dJ)), np.stack((Ie, re))

        return GenericMDMF_dfun
