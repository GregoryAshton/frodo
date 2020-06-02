from scipy.special import hermite
import numpy as np


class BaseFluxModel(object):
    @property
    def hermititian_polynomial(self):
        """ Return a Hermite polynomial of degree i """
        try:
            return self._cached_polynomials
        except AttributeError:
            self._cached_polynomials = [hermite(j) for j in range(self.n_shapelets)]
            return self._cached_polynomials

    @property
    def normalisation(self):
        try:
            return self._normalisation_list
        except AttributeError:
            self._normalisation_list = np.array(
                [
                    np.sqrt(2 ** j * np.sqrt(np.pi) * np.math.factorial(j))
                    for j in range(self.n_shapelets)
                ]
            )
            return self._normalisation_list


class SinglePulseFluxModel(BaseFluxModel):
    """ Model-object for the flux

    Parameters
    ----------
    n_shapelets: int,
        The number of shapelets
    frequency_model: function,
        A function which takes as the first argument, the time, and any
        additional model arguments. This models the time-evolution of the
        frequency.

    """

    def __init__(self, n_shapelets):
        self.n_shapelets = n_shapelets
        self._set_up_parameters()

    def _set_up_parameters(self):
        """ Initiates the parameters """
        self.amp_keys = ["C{}".format(i) for i in range(self.n_shapelets)]
        self.parameters = dict(beta=None, toa=None, base_flux=None)
        for i in range(self.n_shapelets):
            self.parameters[self.amp_keys[i]] = None

    def __call__(self, time, **kwargs):
        x = (time - kwargs["toa"]) / kwargs["beta"]
        pre = np.exp(-x ** 2 / 2) #/ np.sqrt(kwargs["beta"])
        coefs = [
            kwargs[self.amp_keys[i]] for i in range(self.n_shapelets)
        ] / self.normalisation
        return kwargs["base_flux"] + pre * np.polynomial.hermite.Hermite(coefs)(x)
