#!/usr/bin/env python3


import itertools

import numpy as np
import scipy.special as spc
import matplotlib.pyplot as plt
import brewer2mpl


resolution = 72.27
textwidth = 307.28987/resolution
textheight = .85*261.14662/resolution
textiny, texsmall, texnormal, texlarge = 5.5, 7., 8., 9.

plt.rcParams.update({
    'font.family': 'sans-serif',
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'sans',
    'mathtext.it': 'sans:italic',
    'mathtext.bf': 'sans:bold',
    'mathtext.sf': 'sans',
    'font.size': texnormal,
    'axes.titlesize': texnormal,
    'axes.labelsize': texnormal,
    'legend.fontsize': texnormal,
    'xtick.labelsize': textiny,
    'ytick.labelsize': textiny,
    'lines.linewidth': .5,
    'figure.figsize': [textwidth, textheight],
})


def pdf(theta, x):
    """
    Rice / Bessel-Gaussian log PDF.

        theta = (A, s)
        f(x; A, s) = x/s^2 * exp(-(x^2 + A^2)/(2s^2)) * I_0(x*A/s^2)

    """

    A, s = theta
    s2 = s*s

    return x/s2 * np.exp(-.5*(x*x+A*A)/s2) * spc.i0(x*A/s2)


def ll(theta, x):
    """ Log likelihood. """

    return np.log(pdf(theta, x)).sum()


def main():
    M = 300.
    dM = (2.*M)**(-.5)

    A, s = .05, .01

    Xmax = .15
    X = np.linspace(1e-8, Xmax, 300)

    plt.rc('axes', color_cycle=tuple(
        itertools.chain.from_iterable(
            zip(*(brewer2mpl.get_map(i, 'Sequential', 5, reverse=True).mpl_colors for i in
                 ('Blues', 'Oranges'))
                )
        )
    ))

    plt.figure(figsize=(.7*textwidth,.7*textheight))

    for l, c in zip(('a', 'b', 'c'), (1., .9, 0.)):
        d2 = .5*(1-c*c)*A*A + s*s
        plt.plot(X, pdf((c*A, d2**.5), X),
                 label='$v_n^\mathrm{true}$' if c == 1. else None, zorder=10)
        plt.plot(X, pdf((c*A, (d2 + dM*dM)**.5), X),
                 zorder=20,
                 label='$v_n^\mathrm{obs}$' if c == 1. else None)

    plt.plot(X, pdf((0, dM), X), color='black',
             lw=.6, ls='dashed', dashes=(2, 2),
             zorder=15, label='Response')

    plt.xlim(0, Xmax)
    plt.ylim(0, 1.05*pdf((A, s), A))

    plt.xlabel('$v_n$')
    plt.ylabel('$P(v_n)$')

    plt.xticks([])
    plt.yticks([])

    for spine in 'top', 'right':
        plt.gca().spines[spine].set_visible(False)

    plt.legend(loc='lower left', bbox_to_anchor=(.5, .5))

    plt.tight_layout(pad=0.)
    plt.savefig('problem.pdf')
    plt.close()


if __name__ == "__main__":
    main()
