import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import normaltest


class TimeDomainData:
    """ Object to store time-domain flux data """

    def __init__(self):
        pass

    @property
    def start(self):
        return self.time[0]

    @property
    def end(self):
        return self.time[-1]

    @property
    def duration(self):
        return self.end - self.start

    @property
    def N(self):
        return len(self.time)

    @property
    def RMS_flux(self):
        return np.sqrt(np.mean(self.flux ** 2))

    @property
    def max_flux(self):
        return np.max(self.flux)

    @property
    def max_time(self):
        return self.time[np.argmax(self.flux)]

    @classmethod
    def from_txt(cls, filename, dtstart=None, duration=None, pulse_number=None):
        """ Read in the time and flux from a txt file

        Parameters
        ----------
        filename: str
            The path to the file to read.
        dtstart: float, optional
            time-delta from the start from which to truncate the data from. If
            None, then the start of the data is used.
        duration: float, optional
            The duration of data to truncate.
        pulse_number: int:
            The pulse number to truncate.

        """
        df = pd.read_csv(filename)
        return cls._sort_and_filter_dataframe(df, dtstart, duration, pulse_number)

    def from_csv(cls, filename, dtstart=None, duration=None, pulse_number=None):
        """ Read in the time and flux from a csv

        Parameters
        ----------
        filename: str
            The path to the file to read
        dtstart: float, optional
            time-delta from the start from which to truncate the data from. If
            None, then the start of the data is used.
        duration: float, optional
            The duration of data to truncate.
        pulse_number: int:
            The pulse number to truncate.

        """
        df = pd.read_csv(filename)
        return cls._sort_and_filter_dataframe(df, dtstart, duration, pulse_number)

    @staticmethod
    def _sort_and_filter_dataframe(df, dtstart, duration, pulse_number):
        df = df.sort_values("time")
        if dtstart is None:
            tstart = df.time.values[0]
        else:
            tstart = df.time.values[0] + dtstart
        if duration is None:
            duration = df.time.values[-1] - df.time.values[0]

        df = df[(tstart <= df.time) & (df.time < tstart + duration)]

        df = df[df.pulse_number == pulse_number]

        # if pulse_number is not None:
        #     if pulse_number in df.pulse_number.values:
        #         # This shifts the window of pulses to be centered on the start
        #         # which is where the pulse is for zero-phase data
        #         match_idx = np.arange(len(df))[df.pulse_number == pulse_number]
        #         shift_idx = match_idx - int(len(match_idx) / 2.0)
        #         df = df.iloc[shift_idx]
        #     else:
        #         raise ValueError("No data for pulse_number={}".format(pulse_number))

        time_domain_data = TimeDomainData()
        time_domain_data.time = df.time.values
        time_domain_data.flux = df.flux.values
        return time_domain_data

    def plot_max_likelihood(self, result=None, model=None, xlims=None):
        """ Plot the data and max-likelihood

        Parameters
        ----------
        result: bilby.core.result.Result
            The result object to show alongside the data
        model: function
            Function fitted to the data

        """
        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)

        ax1.plot(self.time, self.flux, label="data", lw=2)

        # Plot the maximum likelihood
        s = result.posterior.iloc[result.posterior.log_likelihood.idxmax()]
        ax1.plot(self.time, model(self.time, **s), "--", label="max-l model")
        ax1.legend()
        ax1.set_ylabel("Flux")

        ax2.plot(self.time, self.flux - model(self.time, **s))
        ax2.set_xlabel("time")
        ax2.set_ylabel("Flux residual")

        if xlims is not None:
            ax1.set_xlim(*xlims)

        fig.tight_layout()
        fig.savefig(
            "{}/{}_maxl_with_data.png".format(result.outdir, result.label), dpi=500
        )

    def plot_fit(self, result=None, model=None):
        """ Plot the data and the fit and a residual

        Parameters
        ----------
        result: bilby.core.result.Result
            The result object to show alongside the data
        model: function
            Function fitted to the data

        """
        fig, (ax1, ax2) = plt.subplots(
            nrows=2, sharex=True, figsize=(5, 4), gridspec_kw=dict(height_ratios=[2, 1]))

        ax1.plot(self.time, self.flux, label="data", lw=1, color="C0")

        # Plot the 90%
        npreds = 100
        preds = np.zeros((npreds, len(self.time)))
        for ii in range(npreds):
            s = result.posterior.iloc[np.random.randint(len(result.posterior))]
            preds[ii] = model(self.time, **s)
        ax1.fill_between(
            self.time,
            np.quantile(preds, q=0.05, axis=0),
            np.quantile(preds, q=0.95, axis=0),
            color="C1", alpha=0.8)

        # Plot the maximum likelihood
        s = result.posterior.iloc[result.posterior.log_likelihood.idxmax()]
        ax1.plot(self.time, model(self.time, **s), lw=0.5, color="C2")

        median_sigma = np.median(result.posterior["sigma"])
        ax2.axhspan(-median_sigma, median_sigma, color='k', alpha=0.2)

        res_preds = self.flux - preds
        ax2.fill_between(
            self.time,
            np.quantile(res_preds, 0.05, axis=0),
            np.quantile(res_preds, 0.95, axis=0),
            color='C1', alpha=0.5)
        ax2.plot(self.time, self.flux - model(self.time, **s), "C0", lw=0.5)

        ax1.set_ylabel("Flux [Arb. Units]")
        ax2.set_xlabel("Time [s]")
        ax2.set_ylabel("Flux residual")

        fig.tight_layout()
        fig.savefig(
            "{}/{}_maxl_with_data.png".format(result.outdir, result.label),
            dpi=600
        )

    @property
    def normal_pvalue(self):
        return normaltest(self.flux).pvalue
