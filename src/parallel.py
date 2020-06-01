import argparse
import datetime
import sys
import os
import json
import pickle

import matplotlib.pyplot as plt
import dynesty
from dynesty.plotting import traceplot
from dynesty.utils import resample_equal, unitcheck
from numpy import inf
import numpy as np
from schwimmbad import MPIPool
import bilby
from pandas import DataFrame

from .flux_model import SinglePulseFluxModel
from .data import TimeDomainData
from .likelihood import PulsarLikelihood

from bilby.core.utils import reflect
from numpy import linalg

logger = bilby.core.utils.logger


def main():
    """ Do nothing function to play nicely with MPI """
    pass


class SpikeAndSlab(bilby.core.prior.Prior):
    def __init__(self, spike=0, slab=None, mix=0.5, name=None, latex_label=None, unit=None):
        """Dirac delta function prior, this always returns peak.

        Parameters
        ----------
        peak: float
            Peak value of the delta function
        name: str
            See superclass
        latex_label: str
            See superclass
        unit: str
            See superclass

        """
        #if isinstance(slab, bilby.core.prior.Uniform) is False:
            #raise NotImplementedError()
        minimum = np.min([spike, slab.minimum])
        maximum = np.max([spike, slab.maximum])
        super(SpikeAndSlab, self).__init__(
            name=name, latex_label=latex_label, unit=unit, minimum=minimum,
            maximum=maximum)
        self.mix = mix
        self.spike = spike
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

        CDF_at_spike = self.slab.cdf(self.spike) * (1 - self.mix)
        if isinstance(val, (float, int)):
            if val <= CDF_at_spike:
                return self.slab.rescale(val) / (1 - self.mix)
            elif (CDF_at_spike < val) * (val <= CDF_at_spike + self.mix):
                return self.spike
            elif val > CDF_at_spike + self.mix:
                return (self.slab.rescale(val - self.mix)) / (1 - self.mix)
        else:
            rescales = np.ones_like(val)
            low_idxs = val <= CDF_at_spike
            mid_idxs = (CDF_at_spike < val) * (val <= CDF_at_spike + self.mix)
            up_idxs = val > CDF_at_spike + self.mix
            rescales[low_idxs] = self.slab.rescale(val[low_idxs]) / (1 - self.mix)
            rescales[mid_idxs] = self.spike
            rescales[up_idxs] = (self.slab.rescale(val[up_idxs] - self.mix)) / (1 - self.mix)
            return rescales

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


def plot(data, model, result):
    xlims = (result.priors['toa'].minimum, result.priors['toa'].maximum)
    xlims = None
    result.plot_corner(priors=True)
    data.plot(result, model, xlims=xlims)


def create_parser():
    parser = argparse.ArgumentParser("frodo", add_help=True)

    parser.add_argument('-p', '--pulse-number', type=int, default=None,
                        help='The pulse number to analyse')
    parser.add_argument('-s', '--n-shapelets', type=int, required=True,
                        help='The number of shapelets to use')
    parser.add_argument('-d', '--data-file', type=str, help='The data file',
                        required=True)
    parser.add_argument('--plot', action='store_true', help='Create plots')

    parser.add_argument("--sampling-seed", default=1234)
    parser.add_argument("--clean", action="store_true")

    data_group = parser.add_argument_group(title="Data")
    data_group.add_argument('--data')

    dynesty_group = parser.add_argument_group(title="Dynesty Settings")
    dynesty_group.add_argument(
        "-n", "--nlive", default=1000, type=int, help="Number of live points"
    )
    dynesty_group.add_argument(
        "--dlogz",
        default=0.1,
        type=float,
        help="Stopping criteria: remaining evidence, (default=0.1)",
    )
    dynesty_group.add_argument(
        "--n-effective",
        default=inf,
        type=float,
        help="Stopping criteria: effective number of samples, (default=inf)",
    )
    dynesty_group.add_argument(
        "--dynesty-sample",
        default="rwalk",
        type=str,
        help="Dynesty sampling method (default=rwalk). Note, the dynesty rwalk"
        " method is overwritten by parallel bilby for an optimised version ",
    )
    dynesty_group.add_argument(
        "--dynesty-bound",
        default="multi",
        type=str,
        help="Dynesty bounding method (default=multi)",
    )
    dynesty_group.add_argument(
        "--walks",
        default=100,
        type=int,
        help="Minimum number of walks, defaults to 100",
    )
    dynesty_group.add_argument(
        "--maxmcmc",
        default=5000,
        type=int,
        help="Maximum number of walks, defaults to 5000",
    )
    dynesty_group.add_argument(
        "--nact",
        default=5,
        type=int,
        help="Number of autocorrelation times to take, defaults to 5",
    )
    dynesty_group.add_argument(
        "--min-eff",
        default=10,
        type=float,
        help="The minimum efficiency to switch from uniform sampling.",
    )
    dynesty_group.add_argument(
        "--facc", default=0.5, type=float, help="See dynesty.NestedSampler"
    )
    dynesty_group.add_argument(
        "--vol-dec", default=0.5, type=float, help="See dynesty.NestedSampler"
    )
    dynesty_group.add_argument(
        "--vol-check", default=8, type=float, help="See dynesty.NestedSampler"
    )
    dynesty_group.add_argument(
        "--enlarge", default=1.5, type=float, help="See dynesty.NestedSampler"
    )
    dynesty_group.add_argument(
        "--n-check-point",
        default=100000,
        type=int,
        help="Steps to take before checkpoint",
    )
    return parser


def reorder_loglikelihoods(unsorted_loglikelihoods, unsorted_samples, sorted_samples):
    """ Reorders the stored log-likelihood after they have been reweighted

    This creates a sorting index by matching the reweights `result.samples`
    against the raw samples, then uses this index to sort the
    loglikelihoods

    Parameters
    ----------
    sorted_samples, unsorted_samples: array-like
        Sorted and unsorted values of the samples. These should be of the
        same shape and contain the same sample values, but in different
        orders
    unsorted_loglikelihoods: array-like
        The loglikelihoods corresponding to the unsorted_samples

    Returns
    -------
    sorted_loglikelihoods: array-like
        The loglikelihoods reordered to match that of the sorted_samples


    """

    idxs = []
    for ii in range(len(unsorted_loglikelihoods)):
        idx = np.where(np.all(sorted_samples[ii] == unsorted_samples, axis=1))[0]
        if len(idx) > 1:
            print(
                "Multiple likelihood matches found between sorted and "
                "unsorted samples. Taking the first match."
            )
        idxs.append(idx[0])
    return unsorted_loglikelihoods[idxs]


def sample_rwalk_parallel_with_act(args):
    """ A dynesty sampling method optimised for parallel_bilby

    """

    # Unzipping.
    (u, loglstar, axes, scale, prior_transform, loglikelihood, kwargs) = args
    rstate = np.random
    # Bounds
    nonbounded = kwargs.get("nonbounded", None)
    periodic = kwargs.get("periodic", None)
    reflective = kwargs.get("reflective", None)

    # Setup.
    n = len(u)
    walks = kwargs.get("walks", 50)  # minimum number of steps
    maxmcmc = kwargs.get("maxmcmc", 10000)  # maximum number of steps
    nact = kwargs.get("nact", 10)  # number of act

    accept = 0
    reject = 0
    nfail = 0
    act = np.inf
    u_list = []
    v_list = []
    logl_list = []

    drhat, dr, du, u_prop, logl_prop = np.nan, np.nan, np.nan, np.nan, np.nan
    while len(u_list) < nact * act:
        # Propose a direction on the unit n-sphere.
        drhat = rstate.randn(n)
        drhat /= linalg.norm(drhat)

        # Scale based on dimensionality.
        dr = drhat * rstate.rand() ** (1.0 / n)

        # Transform to proposal distribution.
        du = np.dot(axes, dr)
        u_prop = u + scale * du

        # Wrap periodic parameters
        if periodic is not None:
            u_prop[periodic] = np.mod(u_prop[periodic], 1)
        # Reflect
        if reflective is not None:
            u_prop[reflective] = reflect(u_prop[reflective])

        # Check unit cube constraints.
        if unitcheck(u_prop, nonbounded):
            pass
        else:
            nfail += 1
            if accept > 0:
                u_list.append(u_list[-1])
                v_list.append(v_list[-1])
                logl_list.append(logl_list[-1])
            continue

        # Check proposed point.
        v_prop = prior_transform(u_prop)
        logl_prop = loglikelihood(v_prop)
        if logl_prop >= loglstar:
            u = u_prop
            v = v_prop
            logl = logl_prop
            accept += 1
            u_list.append(u)
            v_list.append(v)
            logl_list.append(logl)
        else:
            reject += 1
            if accept > 0:
                u_list.append(u_list[-1])
                v_list.append(v_list[-1])
                logl_list.append(logl_list[-1])

        # If we've taken the minimum number of steps, calculate the ACT
        if accept + reject > walks:
            act = bilby.core.sampler.dynesty.estimate_nmcmc(
                accept_ratio=accept / (accept + reject + nfail),
                old_act=walks,
                maxmcmc=maxmcmc,
                safety=5,
            )

        # If we've taken too many likelihood evaluations then break
        if accept + reject > maxmcmc:
            logger.warning(
                "Hit maximum number of walks {} with accept={}, reject={}, "
                "nfail={}, and act={}. Try increasing maxmcmc".format(
                    maxmcmc, accept, reject, nfail, act
                )
            )
            break

    # If the act is finite, pick randomly from within the chain
    factor = 0.1
    if len(u_list) == 0:
        logger.warning("No accepted points: returning -inf")
        u = u
        v = prior_transform(u)
        logl = -np.inf
    elif np.isfinite(act) and int(factor * nact * act) < len(u_list):
        idx = np.random.randint(int(factor * nact * act), len(u_list))
        u = u_list[idx]
        v = v_list[idx]
        logl = logl_list[idx]
    else:
        logger.warning(
            "len(u_list)={}<{}: returning the last point in the chain".format(
                len(u_list), int(factor * nact * act)
            )
        )
        u = u_list[-1]
        v = v_list[-1]
        logl = logl_list[-1]

    blob = {"accept": accept, "reject": reject, "fail": nfail, "scale": scale}

    ncall = accept + reject
    return u, v, logl, ncall, blob


def write_checkpoint(
    sampler, resume_file, sampling_time, search_parameter_keys, no_plot=False
):
    """ Writes a checkpoint file

    Parameters
    ----------
    sampler: dynesty.NestedSampler
        The sampler object itself
    resume_file: str
        The name of the resume/checkpoint file to use
    sampling_time: float
        The total sampling time in seconds
    search_parameter_keys: list
        A list of the search parameter keys used in sampling (used for
        constructing checkpoint plots and pre-results)
    no_plot: bool
        If true, don't create a check point plot

    """
    print("")
    logger.info("Writing checkpoint file {}".format(resume_file))

    current_state = dict(
        unit_cube_samples=sampler.saved_u,
        physical_samples=sampler.saved_v,
        sample_likelihoods=sampler.saved_logl,
        sample_log_volume=sampler.saved_logvol,
        sample_log_weights=sampler.saved_logwt,
        cumulative_log_evidence=sampler.saved_logz,
        cumulative_log_evidence_error=sampler.saved_logzvar,
        cumulative_information=sampler.saved_h,
        id=sampler.saved_id,
        it=sampler.saved_it,
        nc=sampler.saved_nc,
        bound=sampler.bound,
        nbound=sampler.nbound,
        boundidx=sampler.saved_boundidx,
        bounditer=sampler.saved_bounditer,
        scale=sampler.saved_scale,
        sampling_time=sampling_time,
    )

    current_state.update(
        ncall=sampler.ncall,
        live_logl=sampler.live_logl,
        iteration=sampler.it - 1,
        live_u=sampler.live_u,
        live_v=sampler.live_v,
        nlive=sampler.nlive,
        live_bound=sampler.live_bound,
        live_it=sampler.live_it,
        added_live=sampler.added_live,
    )

    # Try to save a set of current posterior samples
    try:
        weights = np.exp(
            current_state["sample_log_weights"]
            - current_state["cumulative_log_evidence"][-1]
        )

        current_state["posterior"] = resample_equal(
            np.array(current_state["physical_samples"]), weights
        )
        current_state["search_parameter_keys"] = search_parameter_keys
    except ValueError:
        logger.debug("Unable to create posterior")

    with open(resume_file, "wb") as file:
        pickle.dump(current_state, file)

    # Try to create a checkpoint traceplot
    if no_plot is False:
        try:
            fig = traceplot(sampler.results, labels=sampling_keys)[0]
            fig.tight_layout()
            fig.savefig(filename_trace)
            plt.close("all")
        except Exception:
            pass


def read_saved_state(resume_file, sampler):
    """
    Read a saved state of the sampler to disk.

    The required information to reconstruct the state of the run is read from a
    pickle file.

    Parameters
    ----------
    resume_file: str
        The path to the resume file to read
    sampler: `dynesty.NestedSampler`
        NestedSampler instance to reconstruct from the saved state.

    Returns
    -------
    sampler: dynesty.NestedSampler
        If a resume file exists and was successfully read, the nested sampler
        instance updated with the values stored to disk. If unavailable,
        returns False
    sampling_time: int
        The sampling time from previous runs
    """

    if os.path.isfile(resume_file):
        logger.info("Reading resume file {}".format(resume_file))
        try:
            with open(resume_file, "rb") as file:
                saved = pickle.load(file)
            logger.info("Successfully read resume file {}".format(resume_file))
        except EOFError as e:
            logger.warning("Resume file reading failed with error {}".format(e))
            return False, 0

        sampler.saved_u = list(saved["unit_cube_samples"])
        sampler.saved_v = list(saved["physical_samples"])
        sampler.saved_logl = list(saved["sample_likelihoods"])
        sampler.saved_logvol = list(saved["sample_log_volume"])
        sampler.saved_logwt = list(saved["sample_log_weights"])
        sampler.saved_logz = list(saved["cumulative_log_evidence"])
        sampler.saved_logzvar = list(saved["cumulative_log_evidence_error"])
        sampler.saved_id = list(saved["id"])
        sampler.saved_it = list(saved["it"])
        sampler.saved_nc = list(saved["nc"])
        sampler.saved_boundidx = list(saved["boundidx"])
        sampler.saved_bounditer = list(saved["bounditer"])
        sampler.saved_scale = list(saved["scale"])
        sampler.saved_h = list(saved["cumulative_information"])
        sampler.ncall = saved["ncall"]
        sampler.live_logl = list(saved["live_logl"])
        sampler.it = saved["iteration"] + 1
        sampler.live_u = saved["live_u"]
        sampler.live_v = saved["live_v"]
        sampler.nlive = saved["nlive"]
        sampler.live_bound = saved["live_bound"]
        sampler.live_it = saved["live_it"]
        sampler.added_live = saved["added_live"]
        sampler.bound = saved["bound"]
        sampler.nbound = saved["nbound"]
        sampling_time = datetime.timedelta(
            seconds=saved["sampling_time"]
        ).total_seconds()
        return sampler, sampling_time

    else:
        logger.debug("No resume file {}".format(resume_file))
        return False, 0


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["MKL_DYNAMIC"] = "0"
os.environ["MPI_PER_NODE"] = "16"

parser = create_parser()
args = parser.parse_args()


def prior_transform_function(u_array):
    return priors.rescale(sampling_keys, u_array)


def likelihood_function(v_array):
    parameters = {key: v for key, v in zip(sampling_keys, v_array)}
    if priors.evaluate_constraints(parameters) > 0:
        likelihood.parameters.update(parameters)
        return np.nan_to_num(likelihood.log_likelihood())
    else:
        return np.nan_to_num(-np.inf)


def log_prior_function(v_array):
    params = {key: t for key, t in zip(sampling_keys, v_array)}
    return priors.ln_prob(params)


dynesty.dynesty._SAMPLING["rwalk"] = sample_rwalk_parallel_with_act
dynesty.nestedsamplers._SAMPLING["rwalk"] = sample_rwalk_parallel_with_act

model = SinglePulseFluxModel(n_shapelets=args.n_shapelets)

if ".pickle" in args.data_file:
    data = TimeDomainData.from_pickle(
        args.data_file, pulse_number=args.pulse_number)
elif ".txt" in args.data_file:
    data = TimeDomainData.from_txt(
        args.data_file, pulse_number=args.pulse_number)

likelihood = PulsarLikelihood(data, model)

tmax = data.time[np.argmax(data.flux)]
twidth = 0.2 * data.duration

priors = bilby.core.prior.PriorDict()
priors['base_flux'] = bilby.core.prior.Uniform(
    0, 0.05 * data.max_flux, 'base_flux', latex_label='base flux')
priors['toa'] = bilby.core.prior.Uniform(
    data.start, data.end, 'toa')
priors['beta'] = bilby.core.prior.LogUniform(1e-8, 1e-5, 'beta')
for i in range(args.n_shapelets):
    key = 'C{}'.format(i)
    priors[key] = SpikeAndSlab(
        spike=0, slab=bilby.core.prior.LogUniform(1e-5, 5), mix=0.5, name=key)
priors['sigma'] = bilby.core.prior.Uniform(0, 300, 'sigma')

sampling_keys = []
for p in priors:
    if isinstance(priors[p], bilby.core.prior.Constraint):
        continue
    if isinstance(priors[p], (int, float)):
        likelihood.parameters[p] = priors[p]
    elif priors[p].is_fixed:
        likelihood.parameters[p] = priors[p].peak
    else:
        sampling_keys.append(p)


t0 = datetime.datetime.now()
sampling_time = 0
with MPIPool() as pool:
    if not pool.is_master():
        pool.wait()
        sys.exit(0)
    POOL_SIZE = pool.size

    logger.info("Setting sampling seed = {}".format(args.sampling_seed))
    np.random.seed(args.sampling_seed)

    logger.info(f"sampling_keys={sampling_keys}")
    logger.info("Using priors:")
    for key in priors:
        logger.info(f"{key}: {priors[key]}")

    # Set up some labels
    outdir = 'outdir_single_pulse_{}'.format(args.pulse_number)
    bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
    label = 'single_pulse_{}_shapelets'.format(args.n_shapelets)

    filename_trace = "{}/{}_checkpoint_trace.png".format(outdir, label)
    resume_file = "{}/{}_checkpoint_resume.pickle".format(outdir, label)

    dynesty_sample = args.dynesty_sample
    dynesty_bound = args.dynesty_bound
    nlive = args.nlive
    walks = args.walks
    maxmcmc = args.maxmcmc
    nact = args.nact
    facc = args.facc
    min_eff = args.min_eff
    vol_dec = args.vol_dec
    vol_check = args.vol_check
    enlarge = args.enlarge

    init_sampler_kwargs = dict(
        nlive=nlive,
        sample=dynesty_sample,
        bound=dynesty_bound,
        walks=walks,
        maxmcmc=maxmcmc,
        nact=nact,
        facc=facc,
        first_update=dict(min_eff=min_eff, min_ncall=2 * nlive),
        vol_dec=vol_dec,
        vol_check=vol_check,
        enlarge=enlarge,
        save_bounds=False,
    )

    logger.info(
        "Initialize NestedSampler with {}".format(
            json.dumps(init_sampler_kwargs, indent=1, sort_keys=True)
        )
    )

    ndim = len(sampling_keys)

    sampler = dynesty.NestedSampler(
        likelihood_function,
        prior_transform_function,
        ndim,
        pool=pool,
        queue_size=POOL_SIZE,
        print_func=dynesty.results.print_fn_fallback,
        periodic=None,
        reflective=None,
        use_pool=dict(
            update_bound=True,
            propose_point=True,
            prior_transform=True,
            loglikelihood=True,
        ),
        **init_sampler_kwargs,
    )

    if os.path.isfile(resume_file) and args.clean is False:
        resume_sampler, sampling_time = read_saved_state(resume_file, sampler)
        if resume_sampler is not False:
            sampler = resume_sampler

    logger.info(
        f"Starting sampling for job {label}, with pool size={POOL_SIZE} "
        f"and n_check_point={args.n_check_point}"
    )
    old_ncall = sampler.ncall
    sampler_kwargs = dict(
        print_progress=True,
        maxcall=args.n_check_point,
        n_effective=args.n_effective,
        dlogz=args.dlogz,
    )

    while True:
        sampler_kwargs["add_live"] = False
        sampler_kwargs["maxcall"] += args.n_check_point
        sampler.run_nested(**sampler_kwargs)
        if sampler.ncall == old_ncall:
            break
        old_ncall = sampler.ncall

        sampling_time += (datetime.datetime.now() - t0).total_seconds()
        t0 = datetime.datetime.now()
        write_checkpoint(
            sampler,
            resume_file,
            sampling_time,
            sampling_keys,
            no_plot=False, # args.no_plot,
        )

    sampler_kwargs["add_live"] = True
    sampling_time += (datetime.datetime.now() - t0).total_seconds()

    out = sampler.results
    weights = np.exp(out["logwt"] - out["logz"][-1])
    nested_samples = DataFrame(out.samples, columns=sampling_keys)
    nested_samples["weights"] = weights
    nested_samples["log_likelihood"] = out.logl

    result = bilby.core.result.Result(
        label=label, outdir=outdir, search_parameter_keys=sampling_keys
    )
    result.priors = priors
    result.samples = dynesty.utils.resample_equal(out.samples, weights)
    result.nested_samples = nested_samples
    result.meta_data = {}
    result.meta_data["command_line_args"] = vars(args)
    result.meta_data["command_line_args"]["sampler"] = "parallel_bilby"
    result.meta_data["config_file"] = vars(args)
    result.meta_data["likelihood"] = likelihood.meta_data
    result.meta_data["sampler_kwargs"] = init_sampler_kwargs
    result.meta_data["run_sampler_kwargs"] = sampler_kwargs

    result.log_likelihood_evaluations = reorder_loglikelihoods(
        unsorted_loglikelihoods=out.logl,
        unsorted_samples=out.samples,
        sorted_samples=result.samples,
    )

    result.log_evidence = out.logz[-1]
    result.log_evidence_err = out.logzerr[-1]
    result.log_noise_evidence = likelihood.noise_log_likelihood()
    result.log_bayes_factor = result.log_evidence - result.log_noise_evidence
    result.sampling_time = sampling_time

    result.samples_to_posterior()

    logger.info(f"Saving result to {outdir}/{label}_result.json")
    result.save_to_file(extension="json")
    print(
        "Sampling time = {}s"
        .format(datetime.timedelta(seconds=result.sampling_time))
    )
    print(result)

    plot(data, model, result)
