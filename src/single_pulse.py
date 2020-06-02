""" Command line tool for single pulse analysis """
import os
import argparse

import numpy as np
import matplotlib as mpl

import bilby

from .flux_model import SinglePulseFluxModel
from .data import TimeDomainData
from .likelihood import PulsarLikelihood, NullLikelihood
from .priors import get_priors

mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = "Computer Modern"
mpl.rcParams["text.usetex"] = "True"


def get_args():
    parser = argparse.ArgumentParser(description='Run single pulse analysis')
    parser.add_argument('-p', '--pulse-number', type=int, default=None,
                        help='The pulse number to analyse')
    parser.add_argument('-s', '--n-shapelets', type=int, required=True,
                        help='The number of shapelets to use')
    parser.add_argument('-d', '--data-file', type=str, help='The data file',
                        required=True)
    parser.add_argument('--plot-corner', action='store_true', help='Create corner plots')
    parser.add_argument('--plot-fit', action='store_true', help='Create residual plots')
    parser.add_argument('--plot-run', action='store_true', help='Create run plots')

    parser.add_argument('--base-flux', action='store_true', help='Infer a base flux')
    parser.add_argument('--fractional-time-prior-width', type=float, default=0.1)
    parser.add_argument('--beta-max', type=float, default=1)
    parser.add_argument('--c-max-multiplier', type=float, default=0.1)
    parser.add_argument('--c-mix', type=float, default=0.5)
    parser.add_argument('--sigma-multiplier', type=float, default=0.1)

    parser.add_argument('--sampler', type=str, default="dynesty")
    parser.add_argument('--nlive', type=int, default=250)
    parser.add_argument('--walks', type=int, default=25)
    parser.add_argument('--cpus', type=int, default=4)

    args, _ = parser.parse_known_args()

    return args


def get_model_and_data(args):
    model = SinglePulseFluxModel(n_shapelets=args.n_shapelets)

    if ".pickle" in args.data_file:
        data = TimeDomainData.from_pickle(
            args.data_file, pulse_number=args.pulse_number)
    elif ".txt" in args.data_file:
        data = TimeDomainData.from_txt(
            args.data_file, pulse_number=args.pulse_number)

    return model, data


def run_analysis(args, data, model, priors):
    likelihood = PulsarLikelihood(data, model)

    run_sampler_kwargs = dict(
        sampler=args.sampler, nlive=args.nlive, walks=args.walks,
        queue_size=args.cpus)

    result = bilby.sampler.run_sampler(
        likelihood=likelihood, priors=priors, label=args.label,
        save=False, outdir=args.outdir, check_point_plot=args.plot_run,
        **run_sampler_kwargs)

    s = result.posterior.iloc[result.posterior.log_likelihood.idxmax()]
    residual = data.flux - model(data.time, **s)

    result.meta_data['args'] = args.__dict__
    result.meta_data['residual'] = residual
    result.meta_data['RMS_residual'] = np.sqrt(np.mean(residual**2))
    result.save_to_file()

    priors_null = bilby.core.prior.PriorDict()
    priors_null['sigma'] = priors['sigma']
    priors_null['base_flux'] = priors['base_flux']
    likelihood_null = NullLikelihood(data)
    result_null = bilby.sampler.run_sampler(
        likelihood=likelihood_null, priors=priors_null,
        label=args.label + '_null', outdir=args.outdir, save=False,
        check_point=False, check_point_plot=False,
        **run_sampler_kwargs)
    result_null = None

    return result, result_null


def save(args, data, result, result_null):
    rows = ['pulse_number', 'toa', 'toa_std', 'base_flux', 'base_flux_std',
            'beta', 'beta_std', 'sigma', 'sigma_std', 'log_evidence',
            'log_evidence_err', 'log_null_evidence', 'log_null_evidence_err',
            'toa_prior_width', 'normal_p_value']
    for i in range(args.n_shapelets):
        rows.append('C{}'.format(i))
        rows.append('C{}_err'.format(i))

    summary_outdir = 'single_pulse_summary'
    bilby.core.utils.check_directory_exists_and_if_not_mkdir(summary_outdir)
    filename = '{}/{}_shapelets.summary'.format(
        summary_outdir, args.n_shapelets)
    if os.path.isfile(filename) is False:
        with open(filename, 'w+') as f:
            f.write(','.join(rows) + '\n')

    p = result.posterior
    toa_prior_width = result.priors['toa'].maximum - result.priors['toa'].minimum
    row_list = [args.pulse_number, p.toa.median(), p.toa.std(), p.base_flux.median(),
                p.base_flux.std(), p.beta.median(),
                p.beta.std(), p.sigma.median(), p.sigma.std(),
                result.log_evidence, result.log_evidence_err,
                result_null.log_evidence, result_null.log_evidence_err,
                toa_prior_width, data.normal_pvalue]
    for i in range(args.n_shapelets):
        row_list.append(p['C{}'.format(i)].mean())
        row_list.append(p['C{}'.format(i)].std())

    with open(filename, 'a') as f:
        f.write(','.join([str(el) for el in row_list]) + '\n')


def main():
    args = get_args()

    args.outdir = 'outdir_single_pulse_{}'.format(args.pulse_number)
    args.label = 'single_pulse_{}_shapelets'.format(args.n_shapelets)

    model, data = get_model_and_data(args)
    priors = get_priors(args, data)
    result, result_null = run_analysis(args, data, model, priors)

    if args.plot_corner:
        result.plot_corner(priors=True)

    if args.plot_fit:
        data.plot_fit(result, model)

    save(args, data, result, result_null)
