import bilby
import numpy as np


class NullLikelihood(bilby.core.likelihood.Likelihood):
    def __init__(self, data):
        """
        A Gaussian likelihood for fitting a null to the flux data

        Parameters
        ----------
        data: pyglit.main.TimeDomainData
            Object containing time and flux-density data to analyse
        """

        self.parameters = dict(sigma=None)
        bilby.core.likelihood.Likelihood.__init__(self, dict.fromkeys(self.parameters))

        self.x = data.time
        self.y = data.flux

    def log_likelihood(self):
        sigma = self.parameters["sigma"]
        base_flux = self.parameters["base_flux"]
        log_l = np.sum(
            -((self.y - base_flux) / sigma) ** 2 / 2
            - np.log(2 * np.pi * sigma ** 2) / 2
        )
        return log_l


class PulsarLikelihood(bilby.core.likelihood.Likelihood):
    def __init__(self, data, model):
        """
        A Gaussian likelihood for fitting pulsar flux data

        Parameters
        ----------
        data: pyglit.main.TimeDomainData
            Object containing time and flux-density data to analyse
        model: pyglit.main.PulsarFluxModel
            The model to fit to the data
        """

        self.parameters = model.parameters
        self.parameters["sigma"] = None
        self.sigma = None
        bilby.core.likelihood.Likelihood.__init__(self, dict.fromkeys(self.parameters))

        self.x = data.time
        self.y = data.flux
        self.func = model
        print(
            "Found {} x-data points and {} y-data points".format(
                len(self.x), len(self.y)
            )
        )
        if np.ptp(np.diff(self.x)) < 1e-10:
            print("Data evenly sampled")
        else:
            print("Data not evenly sampled")

    def log_likelihood(self):
        log_l = np.sum(
            -(self.residual / self.sigma) ** 2 / 2
            - np.log(2 * np.pi * self.sigma ** 2) / 2
        )
        return np.nan_to_num(log_l)

    @property
    def residual(self):
        """ Residual of the function against the data. """
        return self.y - self.func(self.x, **self.parameters)

    @property
    def sigma(self):
        return self.parameters["sigma"]
