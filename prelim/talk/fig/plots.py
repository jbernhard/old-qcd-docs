#!/usr/bin/env python3

import re
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


resolution = 72.27
textwidth = 307.28987/resolution
textheight = .85*261.14662/resolution
textiny, texsmall, texnormal, texlarge = 5.5, 7., 8., 9.
colors = ('#33b5e5','#99cc00','#ff4444','#aa66cc','#ffbb33')

plt.rcParams.update({
    #'font.family': 'CMU Sans Serif',
    'font.family': 'sans-serif',
    'font.sans-serif': ['CMU Sans Serif'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'sans',
    'mathtext.it': 'sans:italic',
    'figure.facecolor': '1.0',
    'font.size': texsmall,
    'axes.titlesize': texsmall,
    'axes.labelsize': texsmall,
    'legend.fontsize': texsmall,
    'xtick.labelsize': textiny,
    'ytick.labelsize': textiny,
    'axes.linewidth': .4,
    'xtick.major.size': 1.7,
    'ytick.major.size': 1.7,
    'xtick.minor.size': 1.2,
    'ytick.minor.size': 1.2,
    'axes.color_cycle': colors,
    'figure.figsize': [textwidth,textheight],
})


path = '../../ebe/results/'

def loadflowparams(fn):
    ic,vn,cent = re.search(r'flowparams_(glb|kln)_v([0-9])_([0-9]{2}-[0-9]{2}).dat',fn).groups()
    vn = int(vn)
    params = np.loadtxt(fn,unpack=True)
    return ic,vn,cent,params


def allflowparams():
    return (loadflowparams(fn) for fn in
            sorted(glob(path+'fits/flowparams_*_v*_*.dat')))


def loadatlasflowparams():
    from collections import OrderedDict

    params = OrderedDict()

    with open(path+'exp/atlas/fits/flowparams-all.dat') as f:
        for l in f:
            fields = l.split()
            vn = int(fields[0])
            cent = fields[1]
            p = tuple(map(float,fields[2:]))

            for _ in range(2):
                try:
                    params[vn][cent] = p
                except KeyError:
                    params[vn] = OrderedDict()

    return params

def loaddesign(ic):
    names = (
        'Normalization',
        {'glb':r'$\alpha$','kln':r'$\lambda$'}[ic],
        r'$\tau_0$',
        r'$\eta/s$',
        r'$\tau_\Pi$'
    )

    design = np.loadtxt(path+'design/lhs-design-{}-256.dat'.format(ic),skiprows=1,unpack=True)

    return names, design



#
# all curves
#

if False:
    import scipy.special as sp

    def rice_pdf(v, vrp, dv, exp=np.exp, i0=sp.i0, n2n=np.nan_to_num):
        dv2 = dv*dv
        return v / dv2 * exp(-0.5*(v*v+vrp*vrp)/dv2) * n2n(i0(v*vrp/dv2))

    atlasparams = loadatlasflowparams()

    fig,axes = plt.subplots(nrows=2,ncols=2,sharex='col',sharey='row',figsize=(textwidth,textheight))

    vn = 2

    Xmax = .29

    #for ic,row in zip(,axes):
        #dnames, design = loaddesign(ic)
        #for cent,ax in zip(('00-05','40-45'),row):
    for cent,row in zip(('00-05','40-45'),axes):
        for ic,ax in zip(('glb','kln'),row):
            dnames, design = loaddesign(ic)

            atlas = atlasparams[vn][cent][:2]

            X = np.linspace(1e-6,Xmax,300)
            Y = rice_pdf(X,*atlas)

            params = np.loadtxt(path+'fits/flowparams_{}_v{}_{}.dat'.format(ic,vn,cent),usecols=(0,1))
            for p in params:
                ax.plot(X,rice_pdf(X,*p),lw=.2,alpha=.3)

            ax.plot(X,Y,'k',lw=.8)

            ax.set_xlim(0,Xmax)

            if ax.is_first_col():
                ax.set_ylabel('$P(v_{})$'.format(vn))
            if ax.is_last_row():
                ax.set_xlabel('$v_{}$'.format(vn))
                ax.set_ylim(0,17)
            else:
                ax.set_ylim(0,36)

            ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

            ax.annotate(
                '{} $v_{}$ {}%'.format('Glauber' if ic=='glb' else 'KLN',vn,cent.replace('-','–')),
                (.95,.93), xycoords='axes fraction', va='top', ha='right', size=texnormal)

            if ax.is_first_col() and ax.is_first_row():
                ax.annotate(
                    'thick black line = ATLAS\nthin colored lines = model',
                    (.95,.70), xycoords='axes fraction', va='top', ha='right', size=texnormal)

            #plt.title('{} $v_{}$ {}%'.format(ic,vn,cent).replace('-','–'))
            
    plt.tight_layout(pad=0)

    plt.savefig('allcurves.pdf')


#
# preferred eta/s
#

if False:

    bestN = 10

    avn,acent,*aparams = np.loadtxt(path+'exp/atlas/fits/flowparams-5.dat',
            dtype='i1,S5,f8,f8,f8,f8,f8', unpack=True)
    amean = aparams[2]

    acent = acent.astype(str)
    cents = np.unique(acent).tolist()[:11]
    centidx = np.arange(len(cents))


    dnames,design = loaddesign('glb')
    etas = design[3]

    fig,axes = plt.subplots(figsize=(textwidth,0.8*textheight),ncols=2,sharey=True)

    for ax,ic,title in zip(axes,('glb','kln'),('Glauber','KLN')):

        data = {}

        for _ic,vn,cent,params in allflowparams():
            if _ic != ic:
                continue

            mean = params[2]
            a = amean[(avn == vn) & (acent == cent)]
            order = np.abs(mean - a).argsort()
            best = order[:bestN]

            for _ in range(2):
                try:
                    data[vn].append([cents.index(cent),etas[best]])
                except KeyError:
                    data[vn] = []
                else:
                    break


        for vn,xy in data.items():
            xy12 = []
            for x,y in xy:
                #xy12.append([x+.15*(vn-3),y.mean(),y.min(),y.max()])
                xy12.append([x+.20*(vn-3),y.mean(),y.std()])

            #X,Y,Ymin,Ymax = np.array(xy12).T
            X,Y,Yerr = np.array(xy12).T

            #plt.errorbar(X,Y,yerr=(Y-Ymin,Ymax-Y),
            ax.errorbar(X,Y,yerr=Yerr,
                    fmt='-o',ms=3,capsize=2,label='$v_{}$'.format(vn),zorder=10-2*vn)

        ax.axhline(.08 if ic == 'glb' else .2,c='k',ls='--',lw=.6)

        if ax.is_first_col():
            ax.set_ylabel(r'Preferred $\eta/s$')
            #ax.legend(loc='upper center',bbox_to_anchor=(.40,1),ncol=3)
            ax.legend(loc='upper right',ncol=3).get_frame().set_lw(.4)

        ax.set_xlabel('Centrality')
        ax.set_xlim(centidx.min()-.5,centidx.max()+.5)
        ax.set_xticks(centidx)
        ax.set_xticklabels(cents, rotation=50)


        ax.set_ylim(0,.3)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        #if ax.is_first_row():
        #    ax.annotate(title,(.95,.93),xycoords='axes fraction',ha='right',va='top')
        #else:
        #    ax.annotate(title,(.95,.07),xycoords='axes fraction',ha='right',va='bottom')
        ax.set_title(title,size=texnormal)

    plt.tight_layout(pad=.1)
    plt.savefig('preferredeta.pdf')
    plt.close()


#
# best <v_n>
#

if False:
    avn,acent,*aparams = np.loadtxt(path+'exp/atlas/fits/flowparams-5.dat',
            dtype='i1,S5,f8,f8,f8,f8,f8', unpack=True)
    amean = aparams[2]

    acent = acent.astype(str)
    cents = np.unique(acent).tolist()
    centidx = np.arange(len(cents))


    allflows = tuple(allflowparams())


    #ic = 'glb'

    fig,axes = plt.subplots(figsize=(.6*textwidth,.97*textheight),nrows=2,sharex='col')

    for ax,ic,title in zip(axes,('glb','kln'),('Glauber','KLN')):

        diff = np.zeros_like(allflows[0][-1][0])
        for _ic,vn,cent,params in allflows:
            if _ic != ic:
                continue

            mean = params[2]
            a = amean[(avn == vn) & (acent == cent)]
            diff += np.abs(mean - a)

        best = diff.argmin()


        #plt.figure(figsize=(textwidth,.7*textwidth))

        for vn in np.unique(avn):
            ax.plot(amean[avn == vn],'k-o',ms=3,label='ATLAS' if vn == 2 and ax.is_first_row() else '')

            data = []
            for _ic,_vn,cent,params in allflows:
                if _vn != vn or _ic != ic:
                    continue
                mean = params[2]
                data.append([cents.index(cent),mean[best]])
            X,Y = np.array(data).T

            ax.plot(X,Y,'-o',label='$v_{}$'.format(vn),ms=3)

        dnames,design = loaddesign(ic)
        #ax.annotate(title,(.05,.93),xycoords='axes fraction',ha='left',va='top',size=texnormal)
        ax.annotate(r'{} $\eta/s = {:.2f}$'.format(title,design[3][best]),
                (.05,.93),xycoords='axes fraction',ha='left',va='top',size=texnormal)

        if ax.is_first_row():
            leg = ax.legend(loc='right',ncol=2,bbox_to_anchor=(1,.50))
            leg.get_frame().set_lw(.4)
        else:
            ax.set_xlabel('Centrality')
            ax.set_xlim(centidx.min()-.5,centidx.max()+.5)
            ax.set_xticks(centidx)
            ax.set_xticklabels(cents, rotation=50)


        ax.set_ylim(0,.145)
        ax.set_ylabel(r'$\langle v_n \rangle$')
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        #dnames,design = loaddesign(ic)
        #plt.annotate('design point: '+str(best),(.6,.73),xycoords='axes fraction',ha='left',va='bottom')
        #plt.annotate('\n'.join(('{} = {:.3f}'.format(i,j) for i,j in zip(dnames,design.T[best]))),
        #    (.6,.7),
        #    xycoords='axes fraction',ha='left',va='top')

    #plt.figlegend((hmodel,htrend,hatlas),('model','trend','ATLAS'),'upper right',ncol=3,frameon=False)

    plt.tight_layout(pad=0,w_pad=.3)
    plt.savefig('bestavg.pdf')
    plt.close()

#
# parameter scatter plots
#

if False:
    from scipy.interpolate import UnivariateSpline

    vn = 2
    cent = '20-25'

    with open(path+'exp/atlas/fits/flowparams-all.dat') as f:
        for l in f:
            fields = l.split()

            if fields[0] == str(vn) and fields[1] == cent:
                atlas = tuple(map(float,fields[2:]))
                break

    #for ic in 'glb','kln':
    for ic in 'glb',:
        dnames,design = loaddesign(ic)

        params = np.loadtxt(path+'fits/flowparams_{}_v{}_{}.dat'.format(ic,vn,cent),unpack=True)

        labels = (s.format(vn) for s in (
            r'$v_{}^\mathrm{{RP}}$',
            r'$\delta_{{v_{}}}$',
            r'$\langle v_{} \rangle$',
            r'$\sigma_{{v_{}}}$',
            r'$\sigma_{{v_{0}}} / \langle v_{0} \rangle$',
            ))

        fig,axes = plt.subplots(nrows=len(params), ncols=len(design),
                figsize=(textwidth,textheight),
                sharex='col',sharey='row')

        for row,y,ylabel,a in zip(axes,params,labels,atlas):
            for ax,x,xlabel in zip(row,design,dnames):
                hmodel = ax.scatter(x,y,s=3,c=colors[0],linewidths=.3)

                spline = UnivariateSpline(x,y)
                X = np.linspace(x.min(),x.max(),100)
                htrend, = ax.plot(X,spline(X),'k')

                hatlas = ax.axhline(a,c=colors[2])

                dx = .07*(x.max()-x.min())
                dy = .07*(y.max()-y.min())
                ax.set_xlim(x.min()-dx,x.max()+dx)
                ax.set_ylim(y.min()-dy,y.max()+dy)

                if ax.is_last_row():
                    ax.set_xlabel(xlabel)
                    ax.xaxis.set_major_locator(ticker.MaxNLocator(3))
                if ax.is_first_col():
                    ax.set_ylabel(ylabel)
                    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))

        fig.suptitle('{} $v_{}$ {}%'.format('Glauber' if ic=='glb' else 'KLN',vn,cent.replace('-','–')),x=.30,y=.975)
        leg = plt.figlegend((hmodel,htrend,hatlas),('model','trend','ATLAS'),'upper right',ncol=3)
        leg.get_frame().set_lw(.4)

        fig.tight_layout(pad=0)
        fig.subplots_adjust(hspace=0,wspace=0,top=.93)

        plt.savefig('scatters-{}.pdf'.format(ic))
        plt.close()



#
# IC/hydro profiles
#

if False:
    from matplotlib import cm, gridspec

    ic = 'glb'

    plt.figure(figsize=(1.01*textwidth/3,textwidth/3))
    plt.axes((lambda x,y: [x,y,1-x,1-y])(.20,.20))

    Z = np.loadtxt(ic+'.dat').T
    Z[Z < 1.] = np.nan
    im = plt.imshow(Z,cmap=cm.hot,extent=(-13,13,-13,13))

    #clb = plt.colorbar(im,ticks=[],shrink=.82)
    #clb.set_label('Entropy density [arb. units]')

    plt.xlabel('$x$ [fm]',labelpad=3)
    plt.ylabel('$y$ [fm]',labelpad=-1)

    plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    #plt.tight_layout(pad=0,rect=(-.02,-.05,1,1.05))

    plt.savefig(ic+'.pdf')


if False:
    from matplotlib import cm, gridspec

    ic = 'glb'

    fig = plt.figure(figsize=(textwidth,1/3*textwidth))

    #gs = gridspec.GridSpec(1,2,width_ratios=(.80,1))
    #axes = tuple(map(plt.subplot,gs))

    plt.axes([.053,.2,.3,.8])

    Z = np.loadtxt(ic+'.dat').T
    Z[Z < 1.] = np.nan
    im = plt.imshow(Z,cmap=cm.hot,extent=(-13,13,-13,13))

    #clb = plt.colorbar(im,ticks=[],shrink=.82)
    #clb.set_label('Entropy density [arb. units]')

    plt.xlabel('$x$ [fm]',labelpad=3)
    plt.ylabel('$y$ [fm]',labelpad=-1)

    plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    #ax = axes[0]
    ax = plt.axes([.338,.2,.3,.8])

    Z = np.loadtxt('hydro0.dat',usecols=[2]).reshape((261,261))*1000.
    Z = Z.T
    Z[Z < 1.] = np.nan
    im = ax.imshow(Z,cmap=cm.coolwarm,extent=(-13,13,-13,13))

    ax.set_yticklabels([])
    ax.set_xlabel('$x$ [fm]',labelpad=3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))


    #ax = axes[1]
    ax = plt.axes([.568,.2,.4,.8])

    Z = np.loadtxt('hydro1.dat',usecols=[2]).reshape((261,261))*1000.
    Z = Z.T
    Z[Z < 1.] = np.nan
    im = ax.imshow(Z,cmap=cm.coolwarm,extent=(-13,13,-13,13))

    clb = fig.colorbar(im,ax=ax,ticks=range(0,161,40),shrink=1)
    clb.set_label('Temperature [MeV]')
    clb.ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    ax.set_yticklabels([])

    #for ax in axes:
    ax.set_xlabel('$x$ [fm]',labelpad=3)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    #    ax.set_xlim(-14,14)
    #    ax.set_ylim(-14,14)



    #plt.subplots_adjust(bottom=.1,top=.9)
    #plt.tight_layout(pad=.5,w_pad=0,h_pad=0)

    plt.savefig('hydro.pdf')


if False:
    from matplotlib import cm, gridspec

    ic = 'glb'

    #fig = plt.figure(figsize=(textwidth,.8*textwidth))
    #fig,axes = plt.subplots(figsize=(textwidth,.30*textwidth),ncols=3,sharey='row')
    fig = plt.figure(figsize=(textwidth,.33*textwidth))

    gs = gridspec.GridSpec(1,3,width_ratios=(1,.80,1))
    axes = tuple(map(plt.subplot,gs))
    #ax0 = plt.subplot(gs[0])
    #axes = (ax0,plt.subplot(gs[1],sharey=ax0),plt.subplot(gs[2],sharey=ax0))
    #axes.append()
    #axes.append(plt.subplot(gs[0],sharey=ax0))

    # initial condition
    ax = axes[0]

    Z = np.loadtxt(ic+'.dat')
    Z[Z < 1.] = np.nan
    im = ax.imshow(Z,cmap=cm.hot,extent=(-13,13,-13,13))

    #fig.colorbar(im,ax=ax,label='Entropy density [arbitrary units]',ticks=[])
    clb = fig.colorbar(im,ax=ax,ticks=[],shrink=.82)
    #clb = fig.colorbar(im,cax=axes[1],ticks=[])
    clb.set_label('Entropy density [arb. units]')

    ax.set_title('Initial condition')

    ax.set_ylabel('$y$ [fm]')
    #ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    #ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    # hydro eta/s = 0
    ax = axes[1]
    #ax = axes[2]

    Z = np.loadtxt('hydro0.dat',usecols=[2]).reshape((261,261))*1000.
    Z[Z < 1.] = np.nan
    im = ax.imshow(Z,cmap=cm.coolwarm,extent=(-13,13,-13,13))

    ax.set_title(r'hydro $\eta/s = 0.04$')
    ax.set_yticklabels([])

    #ax.set_xlim(-7.5,7.5)
    #ax.set_ylim(-7.5,7.5)

    # hydro eta/s = 0.16
    ax = axes[2]
    #ax = axes[3]

    Z = np.loadtxt('hydro1.dat',usecols=[2]).reshape((261,261))*1000.
    Z[Z < 1.] = np.nan
    im = ax.imshow(Z,cmap=cm.coolwarm,extent=(-13,13,-13,13))

    clb = fig.colorbar(im,ax=ax,ticks=range(0,161,40),shrink=.82)
    #clb = fig.colorbar(im,cax=axes[4],ticks=range(0,161,40))
    clb.set_label('Temperature [MeV]')
    clb.ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    ax.set_title(r'hydro $\eta/s = 0.24$')
    ax.set_yticklabels([])

    #ax.set_xlim(-7.5,7.5)
    #ax.set_ylim(-7.5,7.5)

    for ax in axes:
        ax.set_xlabel('$x$ [fm]')
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    #    ax.set_xlim(-14,14)
    #    ax.set_ylim(-14,14)



    #plt.subplots_adjust(bottom=.1,top=.9)
    plt.tight_layout(pad=0)#,w_pad=0,h_pad=0)

    plt.savefig(ic+'-hydro.pdf')
    plt.close()



#
# toy Glauber
#

if False:
    from numpy.random import random

    A = 208
    #A = 100
    rmax = 15
    a = 1.
    r0 = 1.25
    R = r0*A**.3333

    #b = 8
    for b in (0,8):

        N1 = np.empty((A,2))
        N2 = np.empty((A,2))

        for k,N in ((-1,N1),(1,N2)):
            for i in range(A):
                while True:
                    r = rmax*random()
                    if random() < 1/(1 + np.exp(r-R/a)):
                        phi = 2*np.pi*random()
                        N[i] = [k*b/2 + r*np.cos(phi),r*np.sin(phi)]
                        break
                    else:
                        continue

        import itertools
        BC = []
        for n1,n2 in itertools.product(N1,N2):
            if np.sum(np.square(n1-n2)) < 1.:
                BC.append(np.mean([n1,n2],axis=0))
        BC = np.array(BC)

        fig = plt.figure(figsize=(textwidth/3,textwidth/3),frameon=False)
        ax = fig.add_axes([0,0,1,1])
        ax.axis('off')

        ax.scatter(*N1.T,s=30,facecolors='none',edgecolors=colors[0],linewidths=.6,label='Nucleus 1')
        ax.scatter(*N2.T,s=30,facecolors='none',edgecolors=colors[1],linewidths=.6,label='Nucleus 2')
        ax.scatter(*BC.T,s=30,facecolors=colors[2],alpha=.5,linewidths=.4,label='Overlap')

        plt.xlim(-rmax,rmax)
        plt.ylim(-rmax,rmax)
        #plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
        #plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

        #plt.xlabel('x [fm]')
        #plt.ylabel('y [fm]')
        #plt.title('$b = {}$ fm'.format(b))
        #plt.legend()

        #plt.tight_layout(pad=0)
        plt.savefig('toyglauber{}.pdf'.format(b))


#
# LHS
#

if False:
    import io

    x1,x2 = np.loadtxt(io.StringIO( """
        0.42822634417098 0.388032464543357
        0.0505391324986704 0.643827478808817
        0.925781215541065 0.930126322433352
        0.567627242649905 0.0458342873607762
        """), unpack=True)

    x3,x4 = np.loadtxt(io.StringIO( """
        0.467693515337305 0.416897835751297
        0.811550274310866 0.887674003874417
        0.134896172577282 0.281725546065718
        0.267213938158238 0.788450275320793
        0.119147670204984 0.14299195408239
        0.420533414883539 0.548948839178774
        0.661207941290922 0.675455083075212
        0.645005048101302 0.449090315663489
        0.595322484802455 0.207589823781746
        0.335605888802093 0.854251757776365
        0.889827489800518 0.586786111368565
        0.484148854773957 0.802034507854842
        0.296338533825474 0.617782153387088
        0.835916681273375 0.394606126128929
        0.780648298666347 0.198461579694413
        0.689335947699146 0.0508677422243636
        0.742244255490368 0.969069675717037
        0.0505514958931599 0.373340009583626
        0.0944041407201439 0.0113130449783057
        0.912365568743553 0.26434143926017
        0.547827016550582 0.652481837617233
        0.241147826868109 0.23083459212794
        0.425271303381305 0.92553154064808
        0.15695037539117 0.645144444721518
        0.770222741126781 0.562770926463418
        0.399981403892161 0.325445039471379
        0.304257830092683 0.0374131604155991
        0.868277710827533 0.754888813971775
        0.0164965114905499 0.721470291883452
        0.360886075830786 0.739259034156566
        0.707556788570946 0.312581436958862
        0.603360302228248 0.826944183651358
        0.203156583954114 0.469311968213879
        0.960078882222297 0.481454085197765
        0.500592260801932 0.123469021456549
        0.571802014403511 0.512846936099231
        0.187819873809349 0.916014516277937
        0.984916324150981 0.0860356904217042
        0.0454414102481678 0.15946246702224
        0.946790276549291 0.982403350580716
        """), unpack=True)


    fig,axes = plt.subplots(ncols=2,sharey='row',figsize=(.8*textwidth,.43*textwidth))

    axes[0].plot(x1,x2,'o',ms=5)
    axes[0].set_title('4 points')
    axes[0].set_ylabel('$y$')
    axes[0].set_xticks([0,.25,.5,.75,1])

    axes[1].plot(x3,x4,'o',ms=2.5)
    axes[1].set_title('40 points')
    axes[1].set_xticks([.25,.5,.75,1])

    for ax in axes:
        ax.grid(ls=':',lw=.5)
        ax.set_yticks([.25,.5,.75,1])
        ax.set_xlabel('$x$')

    plt.tight_layout(pad=.1,w_pad=.5)
    plt.savefig('lhs.pdf')
    plt.close()



if True:
    from numpy.linalg import lstsq

    ic = 'glb'
    vn = 2
    cent = '20-25'

    dnames, design = loaddesign(ic)
    dnames = list(dnames)
    dnames[0] = 'Norm'
    design = design.T

    avg = np.loadtxt(path+'fits/flowparams_{}_v{}_{}.dat'.format(ic,vn,cent),usecols=(2,))
    sort = avg.argsort()

    x = lstsq(design,avg)[0]

    plt.plot(np.sum(design[sort]*x,axis=1),avg[sort],'o',ms=3)

    xlabel = ('' if x[0] > 0 else '-') + '{:.3f}{}'.format(abs(x[0]),dnames[0])
    for a,b in zip(x[1:],dnames[1:]):
        xlabel += (' + ' if a>0 else ' - ') + '{:.3f}{}'.format(abs(a),b)
    #plt.xlabel(' '.join('{:+4.2f}{}'.format(a,b) for a,b in zip(x,dnames)))
    plt.xlabel(xlabel)
    plt.ylabel(r'$\langle v_{} \rangle$'.format(vn))

    plt.xlim(0,.13)
    plt.ylim(0,.12)
    
    plt.gca().xaxis.set_minor_locator(ticker.AutoMinorLocator(2))
    plt.gca().yaxis.set_minor_locator(ticker.AutoMinorLocator(2))

    plt.annotate(
        r'$R^2 \sim\ 0.97$',
        (.05,.93), xycoords='axes fraction', va='top', ha='left', size=texnormal)

    plt.title('{} $v_{}$ {}%'.format('Glauber' if ic=='glb' else 'KLN',vn,cent.replace('-','–')))

    plt.tight_layout(pad=0)

    plt.savefig('linear.pdf')
