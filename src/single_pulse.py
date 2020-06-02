""" Command line tool for single pulse analysis """
import os
import argparse

import numpy as np
import bilby
from bilby.core.prior import Uniform, LogUniform

from .flux_model import SinglePulseFluxModel
from .data import TimeDomainData
from .likelihood import PulsarLikelihood, NullLikelihood
from .priors import SpikeAndSlab


def get_inputs():
    parser = argparse.ArgumentParser(description='Run single pulse analysis')
    parser.add_argument('-p', '--pulse-number', type=int, default=None,
                        help='The pulse number to analyse')
    parser.add_argument('-s', '--n-shapelets', type=int, required=True,
                        help='The number of shapelets to use')
    parser.add_argument('-d', '--data-file', type=str, help='The data file',
                        required=True)
    parser.add_argument('--plot', action='store_true', help='Create plots')
    parser.add_argument('--null', action='store_true', help='')
    args, _ = parser.parse_known_args()

    # Set up some labels
    args.outdir = 'outdir_single_pulse_{}'.format(args.pulse_number)
    args.label = 'single_pulse_{}_shapelets'.format(args.n_shapelets)

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


def get_priors(args, data):
    priors = bilby.core.prior.PriorDict()
    priors['base_flux'] = 0 #bilby.core.prior.Uniform(0, data.max_flux, 'base_flux', latex_label='base flux')
    priors['toa'] = data.max_time  # Uniform(data.start, data.end, 'toa')
    priors['beta'] = LogUniform(1e-8, 1e-1, 'beta')
    for i in range(args.n_shapelets):
        key = 'C{}'.format(i)
        priors[key] = SpikeAndSlab(
            slab=Uniform(0, data.max_flux, key), name=key, mix=0.0)
        priors[key] = Uniform(1e-4, 1, key)

    priors['sigma'] = Uniform(0, .1 * data.max_flux, 'sigma')
    return priors


def run_analysis(inputs, data, model, priors):
    likelihood = PulsarLikelihood(data, model)

    run_sampler_kwargs = dict(
        sampler='dynesty', nlive=250, walks=25, queue_size=4)

    result = bilby.sampler.run_sampler(
        likelihood=likelihood, priors=priors, label=inputs.label,
        save=False, outdir=inputs.outdir, **run_sampler_kwargs)

    s = result.posterior.iloc[result.posterior.log_likelihood.idxmax()]
    residual = data.flux - model(data.time, **s)
    result.meta_data['residual'] = residual
    result.meta_data['RMS_residual'] = np.sqrt(np.mean(residual**2))
    result.save_to_file()

    if inputs.null:
        priors_null = bilby.core.prior.PriorDict()
        priors_null['sigma'] = priors['sigma']
        priors_null['base_flux'] = priors['base_flux']
        likelihood_null = NullLikelihood(data)
        result_null = bilby.sampler.run_sampler(
            likelihood=likelihood_null, priors=priors_null,
            label=inputs.label + '_null',
            outdir=inputs.outdir, **run_sampler_kwargs)
    else:
        result_null = None

    return result, result_null


def plot(data, model, result, result_null):
    xlims = (result.priors['toa'].minimum, result.priors['toa'].maximum)
    xlims = None
    result.plot_corner(priors=True)
    data.plot(result, model, xlims=xlims)
    if result_null is not None:
        result_null.plot_corner(priors=True)


def save(inputs, data, result, result_null):
    rows = ['pulse_number', 'toa', 'toa_std', 'base_flux', 'base_flux_std',
            'beta', 'beta_std', 'sigma', 'sigma_std', 'log_evidence',
            'log_evidence_err', 'log_null_evidence', 'log_null_evidence_err',
            'toa_prior_width', 'normal_p_value']
    for i in range(inputs.n_shapelets):
        rows.append('C{}'.format(i))
        rows.append('C{}_err'.format(i))

    summary_outdir = 'single_pulse_summary'
    bilby.core.utils.check_directory_exists_and_if_not_mkdir(summary_outdir)
    filename = '{}/{}_shapelets.summary'.format(
        summary_outdir, inputs.n_shapelets)
    if os.path.isfile(filename) is False:
        with open(filename, 'w+') as f:
            f.write(','.join(rows) + '\n')

    p = result.posterior
    toa_prior_width = result.priors['toa'].maximum - result.priors['toa'].minimum
    row_list = [inputs.pulse_number, p.toa.median(), p.toa.std(), p.base_flux.median(),
                p.base_flux.std(), p.beta.median(),
                p.beta.std(), p.sigma.median(), p.sigma.std(),
                result.log_evidence, result.log_evidence_err,
                result_null.log_evidence, result_null.log_evidence_err,
                toa_prior_width, data.normal_pvalue]
    for i in range(inputs.n_shapelets):
        row_list.append(p['C{}'.format(i)].mean())
        row_list.append(p['C{}'.format(i)].std())

    with open(filename, 'a') as f:
        f.write(','.join([str(el) for el in row_list]) + '\n')


def main():
    inputs = get_inputs()
    model, data = get_model_and_data(inputs)
    priors = get_priors(inputs, data)
    result, result_null = run_analysis(inputs, data, model, priors)
    if inputs.plot:
        plot(data, model, result, result_null)
    # save(inputs, data, result, result_null)
