#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
import scipy.special as spc
import scipy.optimize as opt
import brewer2mpl


plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size':       16.,
    'axes.labelsize':  16.,
    'legend.fontsize': 16.,
    'xtick.labelsize': 10.,
    'ytick.labelsize': 10.,
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'sans',
    'mathtext.it': 'sans:italic',
    'mathtext.bf': 'sans:bold',
    'mathtext.sf': 'sans',
})


def logpdf(theta, x):
    """
    Rice / Bessel-Gaussian log PDF.

        theta = (A, s)
        f(x; A, s) = x/s^2 * exp(-(x^2 + A^2)/(2s^2)) * I_0(x*A/s^2)

    """

    A, s = theta
    s2 = s*s

    return -.5*(x*x+A*A)/s2 + np.log(x/s2*spc.i0(x*A/s2))


def nll(theta, x):
    """Negative log likelihood with penalty."""

    return -logpdf(theta, x).sum() if (theta >= 0).all() else np.inf


def _fit(x, theta0):
    """
    Simple wrapper to find maximum likelihood parameters.

    Uses Nelder-Mead / downhill simplex algorithm -- function values only, not
    derivatives.  In testing it was equal to or faster than Jacobian-based
    methods.

    Returns SciPy result object if successful, otherwise throws an error.

    """

    res = opt.minimize(nll, theta0, args=(x,), method='Nelder-Mead')

    if not res.success:
        raise RuntimeError(res)

    return res


def fit(x, M, test=False):
    """
    Find corrected parameters for flow distribution `x` with multiplicity `M`.

    Returns various parameters depending on the test option.


    The likelihood function of a Rice distribution often has two local maxima
    with different signal-to-noise ratios (SNR):

        -- high-SNR: (A, s) both finite
        -- low-SNR: (epsilon, s1) with epsilon nearly zero and s1 > s

    In the context of flow distributions, the high-SNR case corresponds to
    systematically driven flow (e.g. peripheral v_2) and the low-SNR case means
    fluctuation-only (e.g. central v_2).

    Finite multiplicity smearing reduces the SNR and makes it difficult to
    distinguish the two cases.

    The following prescription seems to work in nearly all cases:

        -- perform the high-SNR fit
        -- if the distribution is "noisy", also do the low-SNR fit
        -- compare the likelihoods of the two fits
        -- if they are very close, choose low-SNR

    When the likelihoods are similar, the low-SNR option tends to provide more
    sensible results after finite-multiplicity correction.

    The "noisy" criterion is based on the following observations

        -- The signal (Rice `A` parameter) cannot be reliably measured below
            the multiplicity flutuation 1/sqrt(M).
        -- SNR cannot be reliably determined below ~1.

    A distribution is "noisy" if the signal and SNR are small.

    Even in the noisy case, the multiplicity fluctuation can still be as large
    as the observed distribution.  This means effectively zero flow was
    observed within statistical error.  In this case, flow is reverted to a
    reasonable minimum.

    """

    # estimate parameters using sample moments
    # sigma ~ standard deviation
    # mu_2 = A^2 + 2*sigma^2
    s0 = x.std()
    A0 = max(np.square(x).mean()-2.*s0*s0, 0.)**.5
    x0 = np.array((A0, s0))

    # high-SNR fit
    res = _fit(x, x0)
    A, s = res.x

    # check if this is a "noisy" distribution
    if A < M**(-.5) and A/s < 1.2:
        # low-SNR fit
        s1 = (s**2. + .5*A**2.)**.5
        res1 = _fit(x, (1e-3, s1))

        ll = res.fun
        ll1 = res1.fun

        # choose low-SNR if the two likelihoods are very close
        if abs((ll-ll1)/ll) < 5.e-3:
            res = res1

    # extract flow dist. params
    vnrp, dvnobs = res.x

    # correct for finite multiplicity
    # revert to a sensible minimum in the zero-flow scenario
    dvn2 = dvnobs*dvnobs - .5/M
    if dvn2 < 0:
        dvn2 = 1.e-5
        vnrp = 0.

    dvn = dvn2**.5
    vnrp2 = vnrp * vnrp

    # calculate mean and width
    mean = (.5*np.pi)**.5 * dvn * spc.eval_laguerre(.5, -.5*vnrp2/dvn2)
    if np.isnan(mean):
        mean = vnrp + .5*dvn2/vnrp
    width = np.sqrt(2.*dvn2 + vnrp2 - mean*mean)

    if test:
        return vnrp, dvnobs, dvn, mean, width
    else:
        return mean, width


def main():
    import argparse
    import os
    import os.path

    parser = argparse.ArgumentParser(
        description='Fit flow distributions and make plots.'
    )
    parser.add_argument('--datadir',
                        default=os.path.expanduser('~/xtmp/ebe-data/flow/'),
                        help='Top-level path to flow distributions, default \
                        %(default)s.')
    parser.add_argument('--vnmin', type=int, default=2,
                        help='Minimum v_n, default %(default)s.')
    parser.add_argument('ic', help='Initial condition.')
    parser.add_argument('cent', help='Centrality range.')
    parser.add_argument('test', type=int, metavar='N',
                        help='Design point.')
    args = parser.parse_args()

    # construct sorted list of flow dist. files
    flowdir = os.path.join(args.datadir, args.ic, args.cent)
    basenames = sorted(os.listdir(flowdir), key=lambda s: int(s.split('.')[0]))
    files = [os.path.join(flowdir, f) for f in basenames]

    resultspath = os.path.expanduser('~/qcd/ebe/results/model/')

    # load corresponding average multiplicities
    mult = np.loadtxt(
        resultspath + 'mult/{ic}/{cent}.dat'.format(**vars(args)), usecols=[0]
    )

    # make a nice test plot
    dists = np.loadtxt(files[args.test], unpack=True)
    M = mult[args.test]

    fig, axes = plt.subplots(ncols=dists.shape[0], figsize=(12, 4.5))
    colors = [
        brewer2mpl.get_map(i, 'Sequential', 3).hex_colors[1]
        for i in ('Blues', 'Greens', 'Oranges')
    ]

    for n, (dist, ax, color) in enumerate(zip(dists, axes, colors)):
        vnrp, dvnobs, dvn, mean, width = fit(dist, M, test=True)

        vn = args.vnmin + n

        ax.hist(dist, bins=40, normed=True,
                color=color, ec='white')

        X = np.linspace(1e-8, dist.max(), 300)

        plots = (
            ((vnrp, dvnobs), 'Fit', 'dashed'),
            ((0., (2.*M)**(-.5)), 'Response', 'dotted'),
            ((vnrp, dvn), 'Corrected', 'solid')
        )

        for theta, label, ls in plots:
            ax.plot(X, np.exp(logpdf(theta, X)), label=label,
                    color='.25', lw=1, ls=ls)

        ax.set_xlim(0, dist.max())
        ax.set_xlabel('$v_{}$'.format(vn))

        if ax.is_first_col():
            ax.set_ylabel('$P(v_n)$')
            ax.legend()

        ax.set_yticks([])

        for spine in 'top', 'right':
            ax.spines[spine].set_visible(False)

    plt.tight_layout(pad=0.)
    plt.savefig('fit_{ic}_{cent}_{test}.pdf'.format(**vars(args)))
    plt.close()


if __name__ == "__main__":
    main()
