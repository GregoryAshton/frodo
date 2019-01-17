import sys
import os
import argparse

import numpy as np
import bilby

from .flux_model import SinglePulseFluxModel
from .data import TimeDomainData
from .likelihood import PulsarLikelihood, NullLikelihood


def get_inputs():
    parser = argparse.ArgumentParser(description='Run single pulse analysis')
    parser.add_argument('-p', '--pulse-number', type=int, required=True,
                        help='The pulse number to analyse')
    parser.add_argument('-s', '--n-shapelets', type=int, required=True,
                        help='The number of shapelets to use')
    parser.add_argument('-d', '--data-file', type=str, help='The data file',
                        required=True)
    parser.add_argument('--plot', action='store_true', help='Create plots')
    args, _ = parser.parse_known_args()
    return args

    # Set up some labels
    args.outdir = 'outdir_single_pulse_{}'.format(args.n_shapelets)
    args.label = 'single_pulse_{}_shapelets'.format(args.pulse_number)


def get_model_and_data(args):
    model = SinglePulseFluxModel(n_shapelets=args.n_shapelets)

    data = TimeDomainData.from_pickle(
        args.data_file, pulse_number=args.pulse_number)

    dt = 0.02
    sample_rate = data.N / data.duration
    dw = int(dt / 2 * sample_rate)
    idxmid = int(0.5 * data.N)
    tmin = idxmid - dw
    tmax = idxmid + dw
    data.time = data.time[tmin: tmax]
    data.flux = data.flux[tmin: tmax]
    return model, data


def get_priors(args, data):
    priors = bilby.core.prior.PriorDict()
    priors['base_flux'] = bilby.core.prior.Uniform(
        0, data.max_flux, 'base_flux', latex_label='base flux')
    priors['tauP'] = bilby.core.prior.Uniform(
        data.start + 0.25 * data.duration, data.end - 0.25 * data.duration, 'tauP')
    priors['beta'] = bilby.core.prior.LogUniform(1e-6, 5e-4, 'beta')
    for i in range(args.n_shapelets):
        key = 'C{}'.format(i)
        if i == 0:
            priors[key] = bilby.core.prior.LogUniform(1e-8, 1, key)
        else:
            priors[key] = bilby.core.prior.LogUniform(1e-8, 5e-2, key)

    priors['sigma'] = bilby.core.prior.Uniform(0, 5, 'sigma')
    return priors


def run_analysis(inputs, data, model, priors):
    likelihood = PulsarLikelihood(data, model)

    run_sampler_kwargs = dict(
        sampler='dynesty', walks=50, nlive=1000, check_point=True)

    result = bilby.sampler.run_sampler(
        likelihood=likelihood, priors=priors, label=inputs.label,
        save=False, outdir=inputs.outdir, **run_sampler_kwargs)

    s = result.posterior.iloc[result.posterior.log_likelihood.idxmax()]
    residual = data.flux - model(data.time, **s)
    result.meta_data['residual'] = residual
    result.meta_data['RMS_residual'] = np.sqrt(np.mean(residual**2))
    result.save_to_file()

    priors_null = bilby.core.prior.PriorDict()
    priors_null['sigma'] = priors['sigma']
    priors_null['base_flux'] = priors['base_flux']
    likelihood_null = NullLikelihood(data)
    result_null = bilby.sampler.run_sampler(
        likelihood=likelihood_null, priors=priors_null,
        label=inputs.label + '_null',
        outdir=inputs.outdir, **run_sampler_kwargs)

    return result, result_null


def plot(data, model, result, result_null):
    xlims = (result.priors['tauP'].minimum, result.priors['tauP'].maximum)
    xlims = None
    result.plot_corner(priors=True)
    data.plot(result, model, xlims=xlims)
    result_null.plot_corner(priors=True)


def save(inputs, data, result, result_null):
    rows = ['pulse_number', 'toa', 'toa_std', 'base_flux', 'base_flux_std',
            'beta', 'beta_std', 'sigma', 'sigma_std', 'log_evidence',
            'log_evidence_err', 'log_null_evidence', 'log_null_evidence_err',
            'toa_prior_width', 'normal_p_value']
    for i in range(inputs.n_shapelets):
        rows.append('C{}'.format(i))
        rows.append('C{}_err'.format(i))

    filename = 'vela_single_pulse_{}_shapelets.summary'.format(inputs.n_shapelets)
    if os.path.isfile(filename) is False:
        with open(filename, 'w+') as f:
            f.write(','.join(rows) + '\n')

    p = result.posterior
    toa_prior_width = result.priors['tauP'].maximum - result.priors['tauP'].minimum
    row_list = [inputs.pulse_number, p.tauP.median(), p.tauP.std(), p.base_flux.median(),
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
    save(inputs, data, result, result_null)
