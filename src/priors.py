from bilby.core.prior import Prior, Uniform, PriorDict, LogUniform
import numpy as np


def get_priors(args, data):
    priors = PriorDict()

    if args.base_flux:
        priors['base_flux'] = Uniform(
            0, data.max_flux, 'base_flux', latex_label='base flux')
    else:
        priors["base_flux"] = 0

    dt = 0.5 * args.fractional_time_prior_width * data.duration
    priors['toa'] = Uniform(
        data.max_time - dt, data.max_time + dt, "toa")
    priors['beta'] = Uniform(0, args.beta_max, 'beta', latex_label=r'$\beta$')
    for i in range(args.n_shapelets):
        key = 'C{}'.format(i)
        priors[key] = SpikeAndSlab(
            slab=Uniform(0, args.c_max_multiplier * np.abs(data.max_flux)),
            name=key, mix=args.c_mix)

    priors['sigma'] = Uniform(0, args.sigma_multiplier * data.max_flux, 'sigma')
    return priors


class SpikeAndSlab(Prior):
    def __init__(self, slab=None, mix=0.5, name=None, latex_label=None, unit=None):
        """Spike and slab with spike at the slab minimum

        Parameters
        ----------

        """
        if isinstance(slab, Uniform) is False:
            raise NotImplementedError()
        minimum = slab.minimum
        maximum = slab.maximum
        super(SpikeAndSlab, self).__init__(
            name=name, latex_label=latex_label, unit=unit, minimum=minimum,
            maximum=maximum)
        self.mix = mix
        self.spike = minimum
        self.slab = slab

    def rescale(self, val):
        """
        'Rescale' a sample from the unit line element to the appropriate SpikeAndSlab prior.

        Parameters
        ----------
        val: Union[float, int, array_like]

        This maps to the inverse CDF. This has been analytically solved for this case,

        """
        self.test_valid_for_rescaling(val)

        if isinstance(val, (float, int)):
            p = (val - self.mix) / (1 - self.mix)
            if p < 0:
                icdf = self.minimum
            else:
                icdf = self.minimum + p * (self.maximum - self.minimum)
        else:
            p = (val - self.mix) / (1 - self.mix)
            icdf = self.minimum + p * (self.maximum - self.minimum)
            icdf[p < 0] = self.minimum

        return icdf

    def prob(self, val):
        """Return the prior probability of val.

        Parameters
        ----------
        val: Union[float, int, array_like]

        Returns
        -------
        float: Prior probability of val
        """

        if isinstance(val, (float, int)):
            if val == self.spike:
                return self.mix
            else:
                return (1 - self.mix) * self.slab.prob(val)
        else:
            probs = self.slab.prob(val) * (1 - self.mix)
            probs[val == self.spike] = self.mix
            return probs
