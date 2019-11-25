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
        df = pd.read_csv(filename, sep=" ")
        return cls._sort_and_filter_dataframe(df, dtstart, duration, pulse_number)


    @classmethod
    def from_pickle(cls, filename, dtstart=None, duration=None, pulse_number=None):
        """ Read in the time and flux from a python pickle file

        Parameters
        ----------
        filename: str
            The path to the file to read. It is assumed this is a pandas
            HDFStore file with a key 'df'
        dtstart: float, optional
            time-delta from the start from which to truncate the data from. If
            None, then the start of the data is used.
        duration: float, optional
            The duration of data to truncate.
        pulse_number: int:
            The pulse number to truncate.

        """
        df = pd.read_pickle(filename)
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

        if pulse_number is not None:
            if pulse_number in df.pulse_number.values:
                # This shifts the window of pulses to be centered on the start
                # which is where the pulse is for zero-phase data
                match_idx = np.arange(len(df))[df.pulse_number == pulse_number]
                shift_idx = match_idx - int(len(match_idx) / 2.0)
                df = df.iloc[shift_idx]
            else:
                raise ValueError("No data for pulse_number={}".format(pulse_number))

        time_domain_data = TimeDomainData()
        time_domain_data.time = df.time.values
        time_domain_data.flux = df.flux.values
        return time_domain_data

    def plot(self, result=None, model=None, xlims=None):
        """ Plot the data and make likelihood

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

    @property
    def normal_pvalue(self):
        return normaltest(self.flux).pvalue
