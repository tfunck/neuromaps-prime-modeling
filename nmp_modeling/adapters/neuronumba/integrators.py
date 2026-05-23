import numpy as np
import numba as nb

from neuronumba.basic.attr import Attr
from neuronumba.simulator.integrators.base_integrator import Integrator
from neuronumba.numba_tools.config import NUMBA_CACHE, NUMBA_FASTMATH, NUMBA_NOGIL


class EulerStochastic(Integrator):
    """Euler-Maruyama integrator with state-wise or ROI-wise noise."""

    sigmas = Attr(default=None, required=True)
    _sqrt_dt = Attr(dependant=True)

    def _init_dependant(self):
        self._sqrt_dt = np.sqrt(self.dt)

    def get_numba_scheme(self, dfun):
        dt = self.dt
        stimulus = np.zeros((1, 1))
        sigmas = np.asarray(self.sigmas, dtype=np.float64)
        sqrt_dt = self._sqrt_dt

        if sigmas.ndim == 1:
            return self._get_numba_scheme_1d(dfun, dt, stimulus, sigmas, sqrt_dt)

        if sigmas.ndim == 2:
            return self._get_numba_scheme_2d(dfun, dt, stimulus, sigmas, sqrt_dt)

        raise ValueError("sigmas must be a 1D or 2D array.")

    @staticmethod
    def _get_numba_scheme_1d(dfun, dt, stimulus, sigmas, sqrt_dt):
        """Return a scheme for state-variable-specific noise."""

        @nb.njit(
            nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]),
            cache=NUMBA_CACHE,
            fastmath=NUMBA_FASTMATH,
            nogil=NUMBA_NOGIL,
        )
        def scheme(state, coupling):
            d_state, observed = dfun(state, coupling)

            if stimulus.shape[1] == state.shape[1]:
                d_state = d_state + stimulus

            noise = np.zeros(state.shape)
            n_rois = state.shape[1]

            for i in range(sigmas.shape[0]):
                if sigmas[i] > 0.0:
                    noise[i] = np.random.normal(
                        loc=0.0,
                        scale=sigmas[i],
                        size=n_rois,
                    )

            next_state = state + dt * d_state + sqrt_dt * noise
            return next_state, observed

        return scheme

    @staticmethod
    def _get_numba_scheme_2d(dfun, dt, stimulus, sigmas, sqrt_dt):
        """Return a scheme for state-variable-by-ROI-specific noise."""

        @nb.njit(
            nb.types.UniTuple(nb.f8[:, :], 2)(nb.f8[:, :], nb.f8[:, :]),
            cache=NUMBA_CACHE,
            fastmath=NUMBA_FASTMATH,
            nogil=NUMBA_NOGIL,
        )
        def scheme(state, coupling):
            d_state, observed = dfun(state, coupling)

            if stimulus.shape[1] == state.shape[1]:
                d_state = d_state + stimulus

            noise = np.zeros(state.shape)

            for i in range(sigmas.shape[0]):
                for j in range(sigmas.shape[1]):
                    if sigmas[i, j] > 0.0:
                        noise[i, j] = np.random.normal(
                            loc=0.0,
                            scale=sigmas[i, j],
                        )

            next_state = state + dt * d_state + sqrt_dt * noise
            return next_state, observed

        return scheme
