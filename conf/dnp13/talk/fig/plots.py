#!/usr/bin/env python


import matplotlib.pyplot as plt
import numpy as np




_res = 72.27
_figwidth = 307.28987/_res
#_figheight = 244.6939/_res
_figheight = 200.0/_res
#_texsmall = 9.24994
_texsmall = 7.5
_linewidth = .8

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['CMU Sans Serif'],
    'mathtext.fontset': 'custom',
    'mathtext.rm': 'sans',
    'mathtext.it': 'sans:italic',
    'figure.facecolor': '1.0',
    'font.size': _texsmall,
    'axes.titlesize': _texsmall,
    'axes.labelsize': _texsmall,
    'legend.fontsize': _texsmall,
    'xtick.labelsize': _texsmall,
    'ytick.labelsize': _texsmall,
    'axes.color_cycle': ['#33b5e5','#99cc00','#ff4444','#aa66cc','#ffbb33'],
    'figure.figsize': [_figwidth,_figheight],
    'lines.linewidth': _linewidth,
    'lines.markersize': 2,
    'patch.linewidth': _linewidth,
})





data = '/home/jonah/xtmp/ebe/data/'



#
# flow distributions
#

if False:
    for n in range(2):
        plt.clf()
        strn = str(n+2)

        for c in ['0-5','20-25']:

            flows = np.loadtxt(data + c + '_flows.dat', usecols=[n])
            counts, edges = np.histogram(flows, bins=15, density=True)
            centers = (edges[1:] + edges[:-1])/2
            plt.semilogy(centers, counts, zorder=0)

            vn,pvn,stat,syshigh,syslow = np.loadtxt(
                '../../data/v' + strn + '_' + c + '.dat', unpack=True, skiprows=1
            )

            yerr = [np.sqrt(np.square(stat)+np.square(syslow)), np.sqrt(np.square(stat)+np.square(syshigh))]

            plt.errorbar(vn, pvn, yerr=yerr, fmt='o', label=c + '%',
                    color={'0-5':'#33b5e5','20-25':'#99cc00'}[c], lw=.5, capsize=1)


        plt.xlabel(r'$v_' + strn + '$')
        plt.ylabel(r'$P(v_' + strn +  ')$')
        plt.annotate('charged particles\n$p_T > 0.5$ GeV\n$|\\eta| < 2.5$', 
                xy=(10,7), xycoords='axes points', ha='left', va='bottom')
        plt.legend()
        plt.tight_layout(pad=.3)

        plt.savefig('v' + strn + '.pdf')



#
# identified particle spectra
#

if False:

    ID = {
        211:  {'pT': [], 'color': '#33b5e5', 'label': r'$\pi$'},
        321:  {'pT': [], 'color': '#99cc00', 'label': r'$K$'},
        2212: {'pT': [], 'color': '#ff4444', 'label': r'$p$'},
    }
    #pT = {211: [], 321: [], 2212: []}
    #color = {211: '#33b5e5', 321: '#99cc00', 2212: '#ff4444'}

    #for c in ['0-5','20-25']:
    for c in ['0-5']:
        #plt.subplot(121) if c == '0-5' else plt.subplot(122)

        nev = 1000

        with open(data + c + '_charged-eta0.5.dat') as f:
            for l in f:
                if l.strip():
                    fields = l.split()
                    _ID = abs(int(fields[0]))
                    if _ID in ID.keys():
                        ID[_ID]['pT'].append(float(fields[1]))


        for _ID in [211,321,2212]:
            counts, edges = np.histogram(ID[_ID]['pT'], bins=64)
            centers = (edges[1:] + edges[:-1])/2
            err = np.maximum(1,np.sqrt(counts))
            #err = np.sqrt(counts)
            #print(err)
            #plt.semilogy(centers, counts/1000/{211:1,321:10,2212:1000}[ID], 
            #        color={211:'#33b5e5',321:'#99cc00',2212:'#ff4444'}[ID],
            #        label='$' + {211:r'\pi',321:'K/10',2212:'p/1000'}[ID] + '$' if
            #        c=='0-5' else None)
            #plt.semilogy(centers, counts/1000/{211:1,321:10,2212:1000}[ID],
            #        dashes=(4,2) if c == '0-5' else (),
            #        color={211:'#33b5e5',321:'#99cc00',2212:'#ff4444'}[ID],
            #        label='$' + {211:r'\pi',321:'K/10',2212:'p/1000'}[ID] + '$' if
            #        c=='20-25' else None)
            #        #{'0-5':'-','20-25':':'}[c], dashes=(5,2))
            #plt.semilogy()
            #plt.errorbar(centers, counts/1000, fmt='o', yerr=err/1000,
            #        capsize=1, lw=0.5)
            plt.semilogy(centers, counts/nev, color=ID[_ID]['color'],
                    label=ID[_ID]['label'])
            plt.fill_between(centers, (counts+err)/nev, np.maximum(1e-5,(counts-err)/nev),
                    alpha=0.25, lw=0, color=ID[_ID]['color'])



    plt.xlabel(r'$p_T\ \mathrm{[GeV]}$')
    plt.ylabel(r'$1/2\pi p_T dN/dp_Td\eta\ [\mathrm{GeV}^{-2}]$')
    plt.annotate('0-5%\n$|\\eta| < 0.5$', 
            xy=(15,10), xycoords='axes points', ha='left', va='bottom')
    #plt.annotate('solid: 0-5%\ndashed: 20-25%', 
    #        xy=(1.8,10), ha='left', va='bottom')
    plt.legend()
    plt.xlim(0,4.0)
    plt.ylim(1e-4,1e2)
    plt.tight_layout(pad=.1)

    plt.savefig('pT.pdf')



#
# multiplicity
#

if True:

    plt.figure(figsize=(0.75*_figwidth,0.75*_figheight))

    for c in ['0-5','20-25']:

        mult = np.loadtxt(data + c + '_multiplicity.dat')

        #counts, edges = np.histogram(mult, bins=30, density=True)
        #centers = (edges[1:] + edges[:-1])/2
        #plt.plot(centers, counts, label=c+'%')
        #plt.hist(mult, histtype='step', label=c+'%')
        plt.hist(mult, bins=20, label=c+'%')


    plt.xlabel(r'$N_\mathrm{ch}$')
    plt.ylabel('Number of events')
    plt.annotate('$p_T > 0.5$ GeV\n$|\\eta| < 2.5$', xy=(1200,50))
    plt.legend(loc='upper center')
    plt.tight_layout(pad=.1)

    plt.savefig('mult.pdf')




#
# GRW fits
#

if False:
    from scipy.optimize import minimize
    from math import gamma as gammafunc

    # Generalized Reverse Weibull [GRW] PDF
    def grw(X,m=0,s=0.01,a=2.0,g=1.0):
        XMS = (X-m)/s
        return a/(s*gammafunc(g)) * np.power(XMS,(a*g-1)) * np.exp(-np.power(XMS,a))


    # Hellinger distance between two arrays f1,f2
    def hellinger(f1,f2):
        return np.sqrt( np.sum( np.square( np.sqrt(f1) - np.sqrt(f2) ) ) )


    # small quantity > 0
    # needed to constrain fit params > 0
    delta = 1e-5

    # fit to GRW by minimizing Hellinger distance
    # use constrained L-BFGS-B algorithm
    def fitgrw(X,Y):
        return minimize(
                lambda x: hellinger(Y,grw(X,*x)),
                [-.01,0.05,2.0,1.0],
                method='L-BFGS-B',
                bounds=((-1.,min(X)-delta),(delta,1.),(delta,10.),(delta,5.)),
                )


    for n in range(2,4):
        strn = str(n)
        plt.clf()
        plt.figure(figsize=[.9*_figwidth,.75*_figheight])

        for c in ['0-1','10-15','30-35']:

            vn,pvn,stat,syshigh,syslow = np.loadtxt(
                '../../data/v' + strn + '_' + c + '.dat', unpack=True, skiprows=1
            )

            yerr = [np.sqrt(np.square(stat)+np.square(syslow)), np.sqrt(np.square(stat)+np.square(syshigh))]

            res = fitgrw(vn,pvn)
            dist = grw(vn,*res.x)

            plt.semilogy(vn, dist, 'k-')
            plt.errorbar(vn, pvn, yerr=yerr, fmt='o', label=c+'%', capsize=1)


        plt.xlabel(r'$v_' + strn + '$')
        plt.ylabel(r'$P(v_' + strn +  ')$')
        #plt.ylim(ymin=5e-3 if n==2 else 1e-3)
        plt.ylim(5e-3,1e2)
        plt.xlim(0,0.25)
        plt.legend()
        plt.tight_layout(pad=.1)

        plt.savefig('grw' + strn + '.pdf')



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

    plt.figure(figsize=(.9*_figheight,.85*_figheight))
    plt.plot(x1,x2,'o',ms=5)
    plt.grid(ls=':',lw=.5)
    plt.xticks([0,.25,.5,.75,1])
    plt.yticks([.25,.5,.75,1])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(str(x1.size) + ' points')
    plt.tight_layout(pad=.2)
    plt.savefig('lhs1.pdf')

    plt.figure(figsize=(.9*_figheight,.85*_figheight))
    plt.plot(x3,x4,'o',ms=2)
    plt.grid(ls=':',lw=.5)
    plt.xticks([0,.25,.5,.75,1])
    plt.yticks([.25,.5,.75,1])
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.title(str(x3.size) + ' points')
    plt.tight_layout(pad=.2)
    plt.savefig('lhs2.pdf')
