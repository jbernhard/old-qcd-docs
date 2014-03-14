#!/usr/bin/env python

from glob import glob

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import scipy.stats as spst
import scipy.special as spsp



_res = 72.27
_width = 342.2953/_res
_height = 235/_res
_fontsize = 6.5
_linewidth = .8
_colors = ('#33b5e5','#99cc00','#ff4444','#aa66cc','#ffbb33')

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['CMU Sans Serif'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'sans',
    'mathtext.it': 'sans:italic',
    'figure.facecolor': '1.0',
    'font.size': _fontsize,
    'axes.titlesize': _fontsize,
    'axes.labelsize': _fontsize,
    'legend.fontsize': _fontsize,
    'xtick.labelsize': _fontsize,
    'ytick.labelsize': _fontsize,
    'axes.color_cycle': _colors,
    'figure.figsize': [_width,_height],
    'lines.linewidth': _linewidth,
    'lines.markersize': 2,
    'patch.linewidth': _linewidth,
})


samples = ((2,'00-05'),(2,'40-45'))
nsamples = len(samples)


datadir = '../../results/'


def datafiles(pattern):
    return sorted(glob(datadir + pattern))


def expvnparams(vn,cent):
    with open(datadir + 'exp-atlas-vnparams.dat') as f:
        for l in f:
            if l.startswith('{} {}'.format(vn,cent)):
                return tuple(map(float, l.split()[2:]))

def expvnparamsdict():
    from collections import OrderedDict

    params = OrderedDict()

    with open(datadir + 'exp-atlas-vnparams.dat') as f:
        for l in f:
            fields = l.split()
            vn = int(fields[0])
            cent = fields[1]
            p = tuple(map(float,fields[2:]))

            centlow,centhigh = map(int,cent.split('-'))
            if centhigh-centlow == 5:
                for _ in range(2):
                    try:
                        params[vn][cent] = p
                    except KeyError:
                        params[vn] = OrderedDict()

    return params


def gengamma_mean(a,c,s,gamma=spsp.gamma):
    return s*gamma(a+1/c)/gamma(a)



redoall = False


if False or redoall:
    from scipy.stats import gengamma

    fig,axes = plt.subplots(ncols=nsamples,figsize=(_width,.4*_height))

    for sample,ax in zip(samples,axes):
        vn,cent = sample

        params = np.loadtxt(datadir + 'v{}params-cent_{}.dat'.format(vn,cent))
        params = np.delete(params,params[:,3].argmin(),0)

        expparams = expvnparams(vn,cent)

        xmax = 1.3*gengamma.ppf(.999,*expparams)
        X = np.linspace(1e-10,xmax,200)

        for p in params:
            ax.plot(X,gengamma.pdf(X,*p),lw=.2,alpha=.3)

        ax.plot(X,gengamma.pdf(X,*expparams),'k',lw=1)

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax.set_xlim(xmax=xmax)
        if ax.is_first_col():
            ax.set_ylabel('$P(v_2)$'.format(vn))
            ax.annotate(
                'thick black line = ATLAS\nthin colored lines = model',
                (.33,.75),xycoords='axes fraction')

        ax.set_title('$v_{}$ {}%'.format(vn,cent))


    plt.tight_layout(pad=.1,w_pad=2)
    plt.savefig('allcurves.pdf')



if False or redoall:
    from scipy.stats import gengamma

    cents = ('00-05','20-25','40-45')
    vns = (2,3,4)

    fig,axes = plt.subplots(ncols=len(vns),figsize=(_width,.5*_height))#,sharey='row')

    for vn,ax in zip(vns,axes):

        handles = []

        for cent in cents:
            expparams = expvnparams(vn,cent)
            xmax = 1.3*gengamma.ppf(.99,*expparams)
            X = np.linspace(1e-10,xmax,200)
            ax.plot(X,gengamma.pdf(X,*expparams),'k')

            x,y,*err = np.loadtxt(datadir + 'exp-atlas-data/v{}_{}.dat'.format(vn,cent),unpack=True)
            errhigh = np.sqrt(err[0]**2 + err[1]**2)
            errlow = np.sqrt(err[0]**2 + err[2]**2)
            h = ax.errorbar(x,y,yerr=(errlow,errhigh),fmt='.',capsize=1,ms=2,lw=.5)
            handles.append(h)

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        if ax.is_first_col():
            ax.set_ylabel('Probability')

        ax.set_xlabel('$v_{}$'.format(vn))

        #ax.set_title('$v_{}$ {}%'.format(vn,cent))

    plt.figlegend(handles,(c+'%' for c in cents),'upper center',ncol=len(cents))

    plt.tight_layout(pad=.13,w_pad=.7)
    fig.subplots_adjust(top=.82)

    plt.savefig('atlasgengamma.pdf')


if False or redoall:

    fig,axes = plt.subplots(ncols=nsamples,figsize=(_width,.4*_height))

    for sample,ax in zip(samples,axes):
        vn,cent = sample

        params = np.loadtxt(datadir + 'v{}params-cent_{}.dat'.format(vn,cent))
        params = np.delete(params,params[:,3].argmin(),0)

        a,c,l,s = params.T
        mean = gengamma_mean(a,c,s)

        ax.hist(mean, bins=25, normed=True, label='model')

        smooth = spst.gaussian_kde(mean)
        X = np.linspace(mean.min(),mean.max(),100)
        ax.plot(X,smooth(X),'k',label='trendline')


        a,c,l,s = expvnparams(vn,cent)
        ax.axvline(gengamma_mean(a,c,s),color='k',lw=2,ls='--',label='ATLAS')


        ax.set_xlabel(r'$\langle v_{} \rangle$'.format(vn))
        if ax.is_first_col():
            ax.set_ylabel(r'$P(\langle v_{} \rangle)$'.format(vn))
            ax.legend()

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    plt.tight_layout(pad=.1,w_pad=2)
    plt.savefig('avgcomparison.pdf')



if False or redoall:
    from scipy.stats import gengamma

    fig,axes = plt.subplots(ncols=nsamples,nrows=2,figsize=(_width,_height))

    for sample,ax in zip(samples,axes[0]):
        vn,cent = sample

        params = np.loadtxt(datadir + 'v{}params-cent_{}.dat'.format(vn,cent))
        params = np.delete(params,params[:,3].argmin(),0)

        expparams = expvnparams(vn,cent)

        xmax = 1.3*gengamma.ppf(.999,*expparams)
        X = np.linspace(1e-10,xmax,200)

        for p in params:
            ax.plot(X,gengamma.pdf(X,*p),lw=.2,alpha=.3)

        ax.plot(X,gengamma.pdf(X,*expparams),'k',lw=1)

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax.set_xlabel('$v_{}$'.format(vn))

        ax.set_xlim(xmax=xmax)
        if ax.is_first_col():
            #ax.set_ylabel('$P(v_2)$'.format(vn))
            ax.set_ylabel('Probability')
            ax.annotate(
                'thick black line = ATLAS\nthin colored lines = model',
                (.40,.72),xycoords='axes fraction')

        ax.set_title('$v_{}$ {}%'.format(vn,cent),fontsize='large')


    for sample,ax in zip(samples,axes[1]):
        vn,cent = sample

        params = np.loadtxt(datadir + 'v{}params-cent_{}.dat'.format(vn,cent))
        params = np.delete(params,params[:,3].argmin(),0)

        a,c,l,s = params.T
        mean = gengamma_mean(a,c,s)

        ax.hist(mean, bins=25, normed=True, label='model')

        smooth = spst.gaussian_kde(mean)
        X = np.linspace(mean.min(),mean.max(),100)
        ax.plot(X,smooth(X),'k',lw=1.5,label='trend')


        a,c,l,s = expvnparams(vn,cent)
        ax.axvline(gengamma_mean(a,c,s),color=_colors[2],lw=1.5,ls='-',label='ATLAS')


        ax.set_xlabel('Average $v_{}$'.format(vn))
        if ax.is_first_col():
            ax.set_ylabel('Probability')
            ax.legend()

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))


    plt.tight_layout(pad=.15,w_pad=1,h_pad=1)
    plt.savefig('allcurvesavgs.pdf')


if False or redoall:
    fig,axes = plt.subplots(nrows=nsamples,figsize=(.6*_width,_height))

    for sample,ax in zip(samples,axes):
        vn,cent = sample

        params = np.loadtxt(datadir + 'v{}params-cent_{}.dat'.format(vn,cent))
        outlier = params[:,3].argmin()
        params = np.delete(params,outlier,0)

        a,c,l,s = params.T
        mean = gengamma_mean(a,c,s)

        ks = np.delete(np.loadtxt(datadir + 'v{}ks-cent_{}.dat'.format(vn,cent),unpack=True,usecols=(0,)),outlier)

        ax.plot(mean,ks,'o',c=_colors[0],ms=2.5,mew=.2,label='model')

        a,c,l,s = expvnparams(vn,cent)
        ax.axvline(gengamma_mean(a,c,s),color='k',lw=1.1,ls='-',label='ATLAS')

        ax.set_ylabel('KS')
        if ax.is_last_row():
            ax.set_xlabel('Average $v_{}$'.format(vn))
            ax.legend()

        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax.annotate(cent+'%',(.5,.85),xycoords='axes fraction',ha='center',fontsize='large')

        ax.set_ylim(0,.5)

    plt.tight_layout(pad=.15,h_pad=.8)
    plt.savefig('ksvsavg.pdf')


if False or redoall:
    from scipy.interpolate import UnivariateSpline

    for vn,cent in samples:

        params = np.loadtxt(datadir + 'v{}params-cent_{}.dat'.format(vn,cent))
        outlier = params[:,3].argmin()
        params = np.delete(params,outlier,0)

        a,c,l,s = params.T
        mean = gengamma_mean(a,c,s)

        ks = np.delete(np.loadtxt(datadir + 'v{}ks-cent_{}.dat'.format(vn,cent),unpack=True,usecols=(0,)),outlier)

        params = (a,c,s,mean,ks)
        paramnames = ('Shape 1','Shape 2','Scale','Average $v_{}$'.format(vn),'KS')
        N = len(params)

        design = np.delete(np.loadtxt(datadir+'lhs-design-glb-256.dat',skiprows=1),outlier,0).T
        designnames = ('Normalization', r'$\alpha$', r'$\tau_0$', r'$\eta/s$', r'$\tau_\Pi$')
        M = design.shape[0]

        a,c,l,s = expvnparams(vn,cent)
        expparams = (a,c,s,gengamma_mean(a,c,s),0)

        fig,axes = plt.subplots(N,M,sharex='col',sharey='row')

        for n,row in enumerate(axes):
            for m,ax in enumerate(row):
                x = design[m]
                y = params[n]

                hmodel, = ax.plot(x,y,'o',c=_colors[0],ms=1.6,mew=.2)

                spline = UnivariateSpline(x,y)
                X = np.linspace(x.min(),x.max(),100)
                htrend, = ax.plot(X,spline(X),'k',lw=1.5)

                hatlas = ax.axhline(expparams[n],c=_colors[2],lw=1)

                dx = .06*(x.max()-x.min())
                dy = .06*(y.max()-y.min())
                ax.set_xlim(x.min()-dx,x.max()+dx)
                ax.set_ylim(y.min()-dy,y.max()+dy)

                ax.set_xticks([])
                ax.set_yticks([])

                if ax.is_last_row():
                    ax.set_xlabel(designnames[m])
                if ax.is_first_col():
                    ax.set_ylabel(paramnames[n])

        fig.suptitle('$v_{}$ {}%'.format(vn,cent),fontsize='large')
        plt.figlegend((hmodel,htrend,hatlas),('model','trend','ATLAS'),'upper right',ncol=3)

        fig.tight_layout(pad=.1)
        fig.subplots_adjust(hspace=0,wspace=0,top=.93)

        plt.savefig('scatters_v{}_{}.pdf'.format(vn,cent))




    x = design[3]
    xlabel = designnames[3]

    fig,axes = plt.subplots(2,nsamples,sharex='col',sharey='row')

    for row in axes:
        for sample,ax in zip(samples,row):
            vn,cent = sample

            if ax.is_first_row():
                a,c,l,s = np.delete(np.loadtxt(datadir + 'v{}params-cent_{}.dat'.format(vn,cent)),outlier,0).T
                y = gengamma_mean(a,c,s)
                a,c,l,s = expvnparams(vn,cent)
                exp = gengamma_mean(a,c,s)
                ylabel = 'Average $v_{}$'.format(vn)
                ylim = (.02,.15)
            else:
                y = np.delete(np.loadtxt(datadir + 'v{}ks-cent_{}.dat'.format(vn,cent),unpack=True,usecols=(0,)),outlier)
                #exp = None
                exp = 0
                ylabel = 'KS (goodness of fit)'
                ylim = (-.004,.55)


            hmodel, = ax.plot(x,y,'o',c=_colors[0],ms=2,mew=.2)

            spline = UnivariateSpline(x,y)
            X = np.linspace(x.min(),x.max(),100)
            htrend, = ax.plot(X,spline(X),'k',lw=1.5)

            if exp is not None:
                hatlas = ax.axhline(exp,c=_colors[2],lw=1)

            ax.set_ylim(*ylim)

            ax.xaxis.set_minor_locator(AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

            if ax.is_first_row():
                ax.set_title('$v_{}$ {}%'.format(vn,cent),fontsize='large')
            if ax.is_last_row():
                ax.set_xlabel(xlabel)
            if ax.is_first_col():
                ax.set_ylabel(ylabel)

            if ax.is_first_col() and ax.is_first_row():
                ax.legend((hmodel,htrend,hatlas),('model','trend','ATLAS'))

    fig.tight_layout(pad=.1)
    fig.subplots_adjust(hspace=0,wspace=.1,top=.93)

    plt.savefig('avgksvsetas.pdf'.format(vn,cent))



if False or redoall:

    params = expvnparamsdict()

    cents = tuple(params[2].keys())
    centidx = np.arange(len(cents))

    ks = np.loadtxt(datadir + 'v23ks-avg.dat')

    eta = np.loadtxt(datadir+'lhs-design-glb-256.dat',skiprows=1,usecols=(3,))
    etabins = ((0.,.1),(.1,.2),(.2,.3))
    #etabins = ((0.,.06),(.06,.12),(.12,.18),(.18,.24))

    vns = (2,3)

    fig,axes = plt.subplots(ncols=len(vns),figsize=(_width,.79*_height))

    for vn,ax in zip(vns,axes):
        a,c,l,s = np.array(list(params[vn].values())).T
        mean = gengamma_mean(a,c,s)

        #plt.figure(figsize=(.6*_width,.7*_height))

        ax.plot(mean, 'k-o', ms=3, label='ATLAS')

        #for etamin,etamax in etabins:
        for nbin,(etamin,etamax) in enumerate(etabins):
            ksargmin = ( ks == ks[(etamin <= eta) & (eta < etamax)].min() )

            data = []
            for fn in datafiles('v{}params-cent_*.dat'.format(vn)):
                cent = fn.rstrip('.dat').split('_')[-1]
                a,c,l,s = np.loadtxt(fn)[ksargmin].flat
                mean = gengamma_mean(a,c,s)
                data.append([cents.index(cent),mean])
            X,Y = np.array(data).T

            ax.plot(X,Y,'-o',ms=3,label=r'${:.1f} \leq \eta/s < {:.1f}$'.format(etamin,etamax))
            '''
            best = np.any([ks == i for i in np.sort(ks[(etamin <= eta) & (eta < etamax)])[:3]], axis=0)

            idx = []
            mean = []
            for fn in datafiles('v{}params-cent_*.dat'.format(vn)):
                idx.append( cents.index( fn.rstrip('.dat').split('_')[-1] ) )
                a,c,l,s = np.loadtxt(fn)[best].T
                mean.append(gengamma_mean(a,c,s))
            
            X = np.array(idx)
            for k,Y in enumerate(np.array(mean).T):
                plt.plot(X,Y,'-o',c=_colors[nbin],lw=1,ms=3)
            '''

        ax.set_title('$v_{}$'.format(vn),size='large')

        ax.set_xlabel('Centrality')
        if ax.is_first_col():
            ax.set_ylabel('Average $v_n$')
            ax.legend(loc='best')

        ax.set_xlim(centidx.min()-.7,centidx.max()+.7)
        ax.set_xticks(centidx)
        ax.set_xticklabels(cents,rotation=50)

        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax.annotate('PRELIMINARY',(.07,.92),xycoords='axes fraction',va='top',alpha=.2,size='large',weight='bold',zorder=-1)

    plt.tight_layout(pad=.2,w_pad=.7)

    plt.savefig('bestavgvn.pdf')



    plt.clf()

    allvns = (2,3,4)

    plt.figure(figsize=(_width,.95*_height))
    #fig,axes = plt.subplots(ncols=len(allvns),figsize=(_width,.95*_height))
    axes = [
        plt.subplot2grid((2,2),(0,0),rowspan=2),
        plt.subplot2grid((2,2),(0,1)),
        plt.subplot2grid((2,2),(1,1))
    ]

    for vn,ax in zip(allvns,axes):
        a,c,l,s = np.array(list(params[vn].values())).T
        mean = gengamma_mean(a,c,s)

        ax.plot(mean, 'k-o', ms=3, label='ATLAS')

        #for etamin,etamax in etabins:
        for nbin,(etamin,etamax) in enumerate(etabins):
            ksargmin = ( ks == ks[(etamin <= eta) & (eta < etamax)].min() )

            data = []
            for fn in datafiles('v{}params-cent_*.dat'.format(vn)):
                cent = fn.rstrip('.dat').split('_')[-1]
                a,c,l,s = np.loadtxt(fn)[ksargmin].flat
                mean = gengamma_mean(a,c,s)
                data.append([cents.index(cent),mean])
            X,Y = np.array(data).T

            ax.plot(X,Y,'-o',ms=3,label=r'${:.1f} \leq \eta/s < {:.1f}$'.format(etamin,etamax))

        #ax.set_title('$v_{}$'.format(vn))

        ax.set_xlabel('Centrality')
        if ax.is_first_col():
            ax.set_ylabel('Average $v_n$')
            ax.legend(loc='best')

        ax.set_xlim(centidx.min()-.7,centidx.max()+.7)
        ax.set_xticks(centidx)
        ax.set_xticklabels(cents,rotation=50)
        if ax.is_first_row() and ax.is_last_col():
            ax.set_xticklabels([])
            ax.set_xlabel('')


        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        ax.annotate('PRELIMINARY',(.07,.92),xycoords='axes fraction',va='top',alpha=.2,size='large',weight='bold',zorder=-1)
        ax.annotate('$v_{}$'.format(vn),(.11,.80),xycoords='axes fraction',va='top',size='xx-large',weight='bold')

    plt.tight_layout(pad=.2,w_pad=.7,h_pad=.5)

    plt.savefig('bestavgvnwithv4.pdf')


    """
    plt.clf()

    fig,axes = plt.subplots(ncols=len(vns),figsize=(_width,.9*_height))

    for vn,ax in zip(vns,axes):
        a,c,l,s = np.array(list(params[vn].values())).T
        mean = gengamma_mean(a,c,s)

        ax.plot(mean, 'k-o', ms=3, label='ATLAS')

        for nbin,(etamin,etamax) in enumerate(etabins):
            #ksargmin = ( ks == ks[(etamin <= eta) & (eta < etamax)].min() )

            data = []
            for fn in datafiles('v{}params-cent_*.dat'.format(vn)):
                cent = fn.rstrip('.dat').split('_')[-1]
                ksmult = np.mean([ks,1-np.exp(-1.*np.square(np.loadtxt(datadir+'mult_z-cent_{}.dat'.format(cent))))],axis=0)
                argmin = ( ksmult == ksmult[(etamin <= eta) & (eta < etamax)].min() )
                a,c,l,s = np.loadtxt(fn)[argmin].flat
                mean = gengamma_mean(a,c,s)
                data.append([cents.index(cent),mean])
            X,Y = np.array(data).T

            ax.plot(X,Y,'-o',ms=3,label=r'${:.1f} \leq \eta/s < {:.1f}$'.format(etamin,etamax))

        ax.set_title('$v_{}$'.format(vn))

        ax.set_xlabel('Centrality')
        if ax.is_first_col():
            ax.set_ylabel('Average $v_n$')
            ax.legend(loc='best')

        ax.set_xlim(centidx.min()-.7,centidx.max()+.7)
        ax.set_xticks(centidx)
        ax.set_xticklabels(cents,rotation=50)

        ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        plt.tight_layout(pad=.2,w_pad=.7)

        plt.savefig('bestavgvnwithmult.pdf')
    """


    plt.clf()

    nbest = 5
    handles = [None for _ in vns]

    plt.figure(figsize=(.5*_width,.95*_height))

    for vn in vns:

        #data = []
        #for fn in datafiles('v{}ks-cent_*.dat'.format(vn)):
        #    cent = fn.rstrip('.dat').split('_')[-1]
        #    best = np.loadtxt(fn,usecols=(0,)).argmin()
        #    data.append([cents.index(cent),eta[best]])

        #X,Y = np.array(data).T
        #plt.plot(X,Y,'-o',label='$v_{}$'.format(vn))

        idx = []
        data = []
        for fn in datafiles('v{}ks-cent_*.dat'.format(vn)):
            cent = fn.rstrip('.dat').split('_')[-1]
            best = np.loadtxt(fn,usecols=(0,)).argsort()[:nbest]
            idx.append(cents.index(cent))
            data.append(eta[best])

        X = np.array(idx)
        for Y in np.array(data).T:
            handles[vn-2], = plt.plot(X,Y,'-o',c=_colors[vn-2],ms=3)


    plt.xticks(centidx[:9],cents[:9],rotation=35)
    plt.xlim(centidx[0]-.5,centidx[8]+.5)
    plt.xlabel('Centrality')

    plt.gca().yaxis.set_minor_locator(AutoMinorLocator(2))
    plt.ylabel(r'Preferred $\eta/s$')
    plt.legend(handles,['$v_{}$'.format(vn) for vn in vns],loc='best')

    plt.annotate('best {} points'.format(nbest),(.5,.85),
            xycoords='axes fraction',ha='center',fontsize='large')
    plt.annotate('PRELIMINARY',(.4,.4),xycoords='axes fraction',va='top',alpha=.2,size='large',weight='bold',zorder=-1)

    plt.tight_layout(pad=.2)

    plt.savefig('bestetas.pdf')



if False or redoall:
    from scipy.interpolate import UnivariateSpline

    for cent in (i[1] for i in samples):

        #mu,sigma = np.loadtxt(datadir + 'mult_stats-cent_{}.dat'.format(cent),unpack=True)
        #z = np.loadtxt(datadir + 'mult_z-cent_{}.dat'.format(cent))
        y = np.loadtxt(datadir+'mult_stats-cent_{}.dat'.format(cent),usecols=(0,))

        ylabel = 'Average multiplicity'
        N = 1

        design = np.loadtxt(datadir+'lhs-design-glb-256.dat',skiprows=1,unpack=True)
        designnames = ('Normalization', r'$\alpha$', r'$\tau_0$', r'$\eta/s$', r'$\tau_\Pi$')
        M = design.shape[0]

        with open(datadir+'exp-atlas-mult_stats.dat') as f:
            for l in f:
                if l.startswith(cent):
                    expy = float(l.split()[1])
                    break

        fig,axes = plt.subplots(N,M,sharex='col',sharey='row')

        for n,ax in enumerate(axes):
            x = design[n]
            hmodel, = ax.plot(x,y,'o',c=_colors[0],ms=2,mew=.2)

            order = np.argsort(x)
            spline = UnivariateSpline(x[order],y[order],s=1e10,k=3)
            X = np.linspace(x.min(),x.max(),100)
            htrend, = ax.plot(X,spline(X),'k',lw=1.5)

            hatlas = ax.axhline(expy,c=_colors[2],lw=1)

            dx = .06*(x.max()-x.min())
            dy = .06*(y.max()-y.min())
            ax.set_xlim(x.min()-dx,x.max()+dx)
            ax.set_ylim(y.min()-dy,y.max()+dy)

            ax.set_xticks([
                [30,50],
                [.1,.2],
                [.4,.8],
                [0,.1,.2,.3],
                [.4,.8]
                ][n])
            #ax.set_yticks([])
            ax.xaxis.set_minor_locator(AutoMinorLocator(4))

            ax.set_xlabel(designnames[n])
            #if n % 2:
            #    ax.xaxis.set_ticks_position('top')
            #else:
            #    ax.xaxis.set_ticks_position('bottom')

            if ax.is_first_col():
                ax.set_ylabel(ylabel)
            ax.yaxis.set_minor_locator(AutoMinorLocator(2))

        fig.suptitle(cent+'%',fontsize='large')
        plt.figlegend((hmodel,htrend,hatlas),('model','trend','ATLAS'),'upper right',ncol=3)

        fig.tight_layout(pad=.2)
        fig.subplots_adjust(hspace=0,wspace=0,top=.9)

        plt.savefig('scatters_mult_{}.pdf'.format(cent))
