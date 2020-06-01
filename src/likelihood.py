import inspect

import bilby
import numpy as np
from scipy.special import logsumexp


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
        bilby.core.likelihood.Likelihood.__init__(self, dict.fromkeys(self.parameters))

        self.x = data.time
        self.y = data.flux
        self.func = model
        #if np.ptp(np.diff(self.x)) < 1e-10:
        #    print("Data evenly sampled")
        #else:
        #    print("Data not evenly sampled")

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


class PulsarHyperLikelihood(bilby.core.likelihood.Likelihood):
    def __init__(self, df, toa_model):
        self.log_evidence = df.log_evidence.values
        self.log_null_evidence = df.log_null_evidence.values
        self.log_prior_width = np.log(df.toa_prior_width.values)
        self.pulse_number = df.pulse_number.values.astype(float)
        self.toa = df.toa.values
        self.toa_std = df.toa_std.values
        self.toa_model = toa_model
        self.function_keys = inspect.getargspec(toa_model).args
        self.function_keys.pop(0)
        self.parameters = dict.fromkeys(self.function_keys)
        self.parameters['sigma_t'] = None
        self.parameters['xi'] = None

    def log_likelihood(self):
        xi = self.parameters['xi']
        sigma_t = self.parameters['sigma_t']
        sigma2 = sigma_t**2 + self.toa_std**2
        fpars = {k: self.parameters[k] for k in self.function_keys}
        residual = self.toa - self.toa_model(self.pulse_number, **fpars)
        log_l = - residual ** 2 / sigma2 / 2 - np.log(2 * np.pi * sigma2) / 2
        P_d_S_Lambda = self.log_evidence + log_l + self.log_prior_width
        d = [np.log(xi) + P_d_S_Lambda,
             np.log(1 - xi) + self.log_null_evidence]
        return np.nan_to_num(np.sum(logsumexp(d, axis=0))) - np.sum(self.log_evidence)



