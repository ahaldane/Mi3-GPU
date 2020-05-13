#!/usr/bin/env python3
import numpy as np
import pylab as plt
from matplotlib.colors import Normalize, LinearSegmentedColormap
from matplotlib import cm, transforms
import matplotlib as mpl
from scipy.special import rel_entr
import sys, time, argparse

from mi3gpu.utils.potts_common import getLq, getUnimarg, indepF, getM, getXij
from mi3gpu.utils.potts_common import alpha21
import mi3gpu.utils.seqload as seqload
import mi3gpu.utils.changeGauge as changeGauge
from mi3gpu.utils.pseudocount import mutation_pc

o = -0.5 # coordinate offset in pairwise image plot

class DraggableColorbar(object):
    def __init__(self, cbar, mappable):
        self.cbar = cbar
        self.mappable = mappable
        self.press = None
        #self.cycle = sorted([i for i in dir(plt.cm)
        #                     if hasattr(getattr(plt.cm,i),'N')])
        #self.index = self.cycle.index(cbar.get_cmap().name)

    def connect(self):
        """connect to all the events we need"""
        self.cidpress = self.cbar.patch.figure.canvas.mpl_connect(
            'button_press_event', self.on_press)
        self.cidrelease = self.cbar.patch.figure.canvas.mpl_connect(
            'button_release_event', self.on_release)
        self.cidmotion = self.cbar.patch.figure.canvas.mpl_connect(
            'motion_notify_event', self.on_motion)
        #self.keypress = self.cbar.patch.figure.canvas.mpl_connect(
        #    'key_press_event', self.key_press)

    def on_press(self, event):
        """on button press we will see if the mouse is over us and store some data"""
        if event.inaxes != self.cbar.ax: return
        self.press = event.x, event.y

    #def key_press(self, event):
    #    if event.key=='down':
    #        self.index += 1
    #    elif event.key=='up':
    #        self.index -= 1
    #    if self.index<0:
    #        self.index = len(self.cycle)
    #    elif self.index>=len(self.cycle):
    #        self.index = 0
    #    cmap = self.cycle[self.index]
    #    self.cbar.set_cmap(cmap)
    #    self.cbar.draw_all()
    #    self.mappable.set_cmap(cmap)
    #    self.mappable.axes.set_title(cmap)
    #    self.cbar.patch.figure.canvas.draw()

    def on_motion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if self.press is None: return
        if event.inaxes != self.cbar.ax: return
        xprev, yprev = self.press
        dx = event.x - xprev
        dy = event.y - yprev
        self.press = event.x,event.y
        scale = self.cbar.norm.vmax - self.cbar.norm.vmin
        perc = 0.03
        if event.button==1:
            self.cbar.norm.vmin -= (perc*scale)*np.sign(dy)
            self.cbar.norm.vmax -= (perc*scale)*np.sign(dy)
        elif event.button==3:
            self.cbar.norm.vmin -= (perc*scale)*np.sign(dy)
            self.cbar.norm.vmax += (perc*scale)*np.sign(dy)
        self.cbar.draw_all()
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()

    def on_release(self, event):
        """on release we reset the press data"""
        self.press = None
        self.mappable.set_norm(self.cbar.norm)
        self.cbar.patch.figure.canvas.draw()

    def disconnect(self):
        """disconnect all the stored connection ids"""
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidpress)
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidrelease)
        self.cbar.patch.figure.canvas.mpl_disconnect(self.cidmotion)

def drawGrid(ax, x,y,d,nx,ny, content, rowlabel, collabel, title, rowsum, colsum, labeltext=None):
    for i in range(ny+1):
        ax.plot([x+i*d,x+i*d],[y,y+nx*d], 'k-')
    for i in range(nx+1):
        ax.plot([x,x+ny*d],[y+i*d,y+i*d], 'k-')
    fs = 12

    if labeltext is None:
        def labeltext(ax, *args, **kwargs):
            ax.text(*args, **kwargs)

    if content != None:
        for i in range(ny):
            labeltext(ax, x+0.5+i, y+0.5+nx, rowlabel[i],
                       horizontalalignment='center',
                       verticalalignment='bottom',
                       fontsize=fs, rotation='vertical')
            if rowsum is not None:
                ax.text(x+0.5+i, y-0.5, rowsum[i]['text'],
                           horizontalalignment='center',
                           verticalalignment='center',
                           fontsize=fs, color=rowsum[i]['color'])
            for j in range(nx):
                c = content[i][j]
                ax.text(x+0.5+i, y+0.5+ny-1-j, c['text'],
                           horizontalalignment='center',
                           verticalalignment='center',
                           fontsize=fs, color=c['color'])
        for j in range(nx):
            labeltext(ax, x-0.5, y+0.5+nx-1-j, collabel[j],
                       horizontalalignment='right',
                       verticalalignment='center',
                       fontsize=fs)
            if colsum is not None:
                ax.text(x+0.5+nx, y+0.5+nx-1-j, colsum[j]['text'],
                           horizontalalignment='center',
                           verticalalignment='center',
                           fontsize=fs, color=colsum[j]['color'])
    ax.text(x+ny/2, y+1.5+nx+1, title,
               horizontalalignment='center',
               verticalalignment='center',
               fontsize=fs)

def alphatext(ax,x,y,ls,**kw):
    """
    Take a list of strings ``ls`` and colors ``lc`` and place them next to each
    other, with text ls[i] being shown in color lc[i].
    """
    t = ax.transData
    #horizontal version
    if kw.get('rotation', 'horizontal') == 'horizontal':
        for s,c in ls:
            text = ax.text(x,y,s, alpha=c, transform=t, **kw)
            text.draw(ax.figure.canvas.get_renderer())
            ex = text.get_window_extent()
            t = transforms.offset_copy(text._transform, x=-ex.width,
                                       units='dots')
    else:
        for s,c in ls:
            text = ax.text(x,y,s,alpha=c, transform=t,
                    va='bottom',ha='center',**kw)
            text.draw(ax.figure.canvas.get_renderer())
            ex = text.get_window_extent()
            t = transforms.offset_copy(text._transform, y=ex.height,
                                       units='dots')

class PositionPicker:
    def __init__(self, contactfig, ax, alphacolor, J, h, ff, score):
        self.presstime = 0
        self.margax, self.Cax, self.Jax = ax
        contactfig.canvas.mpl_connect('button_press_event', self.onpress)
        contactfig.canvas.mpl_connect('button_release_event', self.onrelease)
        self.i, self.j = None, None

        self.J = J
        self.h = h
        self.ff = ff
        self.score = score

        self.alphacolor = alphacolor

        L, q = getLq(J)
        pairs = [(i,j) for i in range(L-1) for j in range(i+1,L)]
        pp = dict((p,n) for n,p in enumerate(pairs))
        pp.update([((j,i), n) for (i,j),n in pp.items()])
        self.pp = pp
        self.L, self.q = L, q

    def onpress(self, event):
        self.presstime = time.time()

    def onrelease(self, event):
        if time.time() - self.presstime < 0.2:
            if event.xdata is not None:
                L, q = self.L, self.q

                i,j = int(event.xdata-o), int(event.ydata-o)
                if i == j:
                    return
                self.i, self.j = i, j

                pij = self.pp[(i,j)]
                ffij = self.ff[pij,:].reshape(q,q)
                Jij = self.J[pij,:].reshape(q,q)
                hi = self.h[i,:]
                hj = self.h[j,:]
                score = self.score[pij]

                if j < i:
                    ffij = ffij.T
                    Jij = Jij.T

                self.margax.clear()
                self.Cax.clear()
                self.Jax.clear()

                drawMarg(self.alphacolor, q, i,j, ffij, Jij, hi, hj, score,
                         self.margax, self.Cax, self.Jax)

                self.margax.figure.canvas.draw()
                self.Cax.figure.canvas.draw()
                self.Jax.figure.canvas.draw()

def drawMarg(alphacolor, q, i,j, marg, J, hi, hj, score, margax, Cax, Jax):
    graytext = lambda x: {'text': "{:.2f}".format(x), 
                          'color': cm.gray_r(fnorm(x))}
    bwrtext = lambda x: {'text': "{:.2f}".format(x), 'color': cm.bwr(fnorm(x))}
    rwbtext = lambda x: {'text': "{:.2f}".format(x), 'color': cm.bwr_r(fnorm(x))}
    mapi = alphacolor[i]
    mapj = alphacolor[j]

    for ax in (margax, Cax, Jax):
        ax.set_axis_off()
        ax.set_xlim(0, 6+q)
        ax.set_ylim(0, 1*(q+5))

    fnorm = Normalize(0, 0.1, clip=True)
    drawGrid(margax, 3, 1, 1, q, q,
             [[graytext(x) for x in r] for r in marg],
             mapi, mapj, '({}, {})   Bimarg'.format(i,j),
             list(map(graytext, np.sum(marg, axis=1))),
             list(map(graytext, np.sum(marg, axis=0))),
             labeltext=alphatext)

    fnorm = Normalize(-1, 1.0, clip=True)
    C = marg - np.outer(np.sum(marg, axis=1), np.sum(marg, axis=0))
    Cmax = np.max(np.abs(C))
    drawGrid(Cax, 3, 1, 1, q, q,
             [[rwbtext(x) for x in r] for r in C/Cmax],
             mapi, mapj, '({}, {})   C * {}'.format(i, j, str(Cmax)),
             None, None,
             labeltext=alphatext)

    fnorm = Normalize(-1, 1.0, clip=True)
    drawGrid(Jax, 3, 1, 1, q, q,
             [[bwrtext(x) for x in r] for r in J],
             mapi, mapj, '({}, {})   J score={:.2f}'.format(i, j, score),
             list(map(bwrtext, hi)), list(map(bwrtext, hj)),
             labeltext=alphatext)

    # Compute the KL terms that make up the MI scores
    #S = rel_entr(marg, outer(np.sum(marg, axis=1), np.sum(marg, axis=0)))
    #fnorm = Normalize(0, np.max(S), clip=True)
    #drawGrid(Jax, 3, 1, 1, q, q,
    #         [[graytext(x) for x in r] for r in S],
    #         mapi, mapj, '({}, {})   MIab'.format(i, j),
    #         None, None,
    #         labeltext=alphatext)

cdict = {'red':   ((0.0,  1.0, 1.0),
                   (1.0,  1.0, 1.0)),
         'green': ((0.0,  1.0, 1.0),
                   (1.0,  0.0, 0.0)),
         'blue':  ((0.0,  1.0, 1.0),
                   (1.0,  0.0, 0.0)),
         'alpha': ((0.0,  0.0, 0.0),
                   (1.0,  1.0, 1.0))}
alphared = LinearSegmentedColormap('AlphaRed', cdict)

def main():
    parser = argparse.ArgumentParser(description='Visualize Potts Models')
    parser.add_argument('bimarg')
    parser.add_argument('couplings')
    parser.add_argument('-alphamap', help='map from 21 letters to alpha')
    parser.add_argument('-unimarg21', help='21 letter univariate marginals')
    parser.add_argument('-contactfreq', help='contact frequency file')
    parser.add_argument('-contactmode',
                        choices=['split', 'overlay', 'splitoverlay'],
                        default='overlay',
                        help='how to draw contact map')
    parser.add_argument('-score', default='fbwsqrt',
                        choices=['fb', 'fbw', 'fbwsqrt', 'DI', 'MI', 'Xij'])
    parser.add_argument('-gauge', default='skip',
                        choices=['skip', 'nofield', '0', 'w', 'wsqrt'])
    parser.add_argument('-alpha', default=alpha21)
    parser.add_argument('-annotimg', help='annotation image')
    parser.add_argument('-title', help='Figure title')
    parser.add_argument('-grid', type=int, default=10, help='grid spacing')
    parser.add_argument('-cnsrv', help='show conservation score')
    parser.add_argument('-Xijseq', help='seq ')
    parser.add_argument('-deltaXijseq', help='seq ')
    parser.add_argument('-pcN', help='small jeffreys pseudocount to add')
    parser.add_argument('-outscore', help='output interaction scores to file')

    args = parser.parse_args(sys.argv[1:])

    alpha = args.alpha

    ff = np.load(args.bimarg)
    J = np.load(args.couplings)

    if args.pcN:
        N = float(args.pcN)
        ff = mutation_pc(ff, N)

    unimarg = getUnimarg(ff)

    L, q = getLq(J)

    if args.gauge == 'nofield':
        h, J = changeGauge.fieldlessEven(np.zeros((L,q)), J)
    elif args.gauge == '0':
        h, J = changeGauge.zeroGauge(np.zeros((L,q)), J)
    elif args.gauge == 'w':
        h, J = changeGauge.zeroGauge(np.zeros((L,q)), J, weights=ff)
    elif args.gauge == 'wsqrt':
        h, J = changeGauge.zeroGauge(np.zeros((L,q)), J, weights=np.sqrt(ff))
    else:
        h = np.zeros((L,q))

    if args.deltaXijseq or args.Xijseq:
        Xij, Xijab = getXij(J, ff)

        if args.deltaXijseq:
            if args.Xijseq:
                raise ValueError("deltaXijseq, Xijseq are mutually exclusive")
            seq = args.deltaXijseq
            Xijab = Xijab - Xij
        else:
            seq = args.Xijseq

        seq = np.array([alpha.index(c) for c in seq], dtype='u4')
        pottsScore = np.array([Xijab[n, q*seq[i] + seq[j]] for n,(i,j) in
                               enumerate((i,j) for i in range(L-1)
                                               for j in range(i+1,L))])
    elif args.score == 'fb':
        h0, J0 = changeGauge.zeroGauge(h, J)
        pottsScore = np.sqrt(np.sum(J0**2, axis=1))
    elif args.score == 'fbw':
        w = ff
        hw, Jw = changeGauge.zeroGauge(np.zeros((L,q)), J, weights=w)
        pottsScore = np.sqrt(np.sum((Jw*w)**2, axis=1))
    elif args.score == 'fbwsqrt':
        w = np.sqrt(ff)
        hw, Jw = changeGauge.zeroGauge(np.zeros((L,q)), J, weights=w)
        pottsScore = np.sqrt(np.sum((Jw*w)**2, axis=1))
    elif args.score == 'Xij':
        C = ff - indepF(ff)
        X = -np.sum(C*J, axis=1)
        pottsScore = X
    elif args.score == 'MI':
        pottsScore = np.sum(rel_entr(ff, indepF(ff)), axis=-1)
    else:
        raise Exception("Not yet implemented")

    if args.outscore:
        np.save(args.outscore, pottsScore)

    if args.alphamap:
        unimarg21 = np.load(args.unimarg21)

        with open(args.alphamap) as f:
            alphamap = [l.split()[1:] for l in f.readlines()]
            # replace "junk" entry in alpha map with '*'
            alphamap_color = []
            for l,a in enumerate(alphamap):
                clet = []
                for g in a:
                    if g == '*':
                        clet.append([])
                        continue
                    linds = [alpha.index(c) for c in g]
                    f = np.array([unimarg21[l,i] for i in linds])
                    let_dat = zip(g, f/np.sum(f))
                    let_dat.sort(key=lambda x: x[1], reverse=True)
                    clet.append(let_dat)
                alphamap_color.append(clet)
    else:
        alphamap_color = [[[(c, 1.0)] for c in alpha] for i in range(L)]

    ss = 0.4

    contactfig = plt.figure(figsize=(11, 10))
    xscale = 10.0/11.0
    yscale = 10.0/10.0

    main_ax = plt.axes((0.01*xscale, 0.1*yscale, 0.89*xscale, 0.89*yscale),
                       zorder=3)
    cbar_ax = plt.axes((10.05/11, 0.1*yscale, 0.3/11, 0.89*yscale), zorder=3)
    
    if args.annotimg:
        try:
            ssim = np.load(args.annotimg)
        except ValueError:
            from PIL import Image
            ssim = np.array(Image.open(args.annotimg))

        ax = plt.axes((0.01*xscale, 0.02*yscale, 0.89*xscale, 0.08*yscale),
                      sharex=main_ax)
        ax.set_axis_off()
        ax.imshow(ssim, extent=(-0.5,L-0.5, 0, 16), interpolation='bicubic',
                  aspect='auto')
        ax = plt.axes((0.9*xscale, 0.1*yscale, 0.08*xscale, 0.89*yscale),
                      sharey=main_ax)
        ax.set_axis_off()
        ax.imshow(ssim.transpose((1,0,2))[::-1,:,:], extent=(0,16,-0.5,L-0.5),
                  interpolation='bicubic', aspect='auto')


    if args.contactfreq and args.contactmode == 'overlay':
        cont = getM(np.load(args.contactfreq))
        # assume grayscale [0,1]
        cont = cm.gray_r(cont*0.2)
        main_ax.imshow(cont, origin='lower',
                       extent=(+o,L+o,+o,L+o), interpolation='nearest')


        scores = getM(pottsScore)
        img = main_ax.imshow(scores, origin='lower', cmap=alphared,
                             extent=(+o,L+o,+o,L+o), interpolation='nearest')
        cbar = contactfig.colorbar(img, cax=cbar_ax)
        cbar = DraggableColorbar(cbar, img)
        cbar.connect()
    elif args.contactfreq and args.contactmode == 'split':
        lotri = np.ones((L,L), dtype=bool)
        lotri[np.triu_indices(L,k=1)] = False
        hitri = np.zeros((L,L), dtype=bool)
        hitri[np.triu_indices(L,k=1)] =True

        upper = getM(pottsScore)
        upper[hitri] = np.nan
        img = main_ax.imshow(upper, origin='lower', cmap='Blues',
                             extent=(+o,L+o,+o,L+o), interpolation='nearest')
        cbar = contactfig.colorbar(img, cax=cbar_ax)
        cbar = DraggableColorbar(cbar,img)
        cbar.connect()

        lower = getM(np.load(args.contactfreq))
        lower[lotri] = np.nan
        main_ax.imshow(lower, origin='lower', cmap='Reds',
                       extent=(+o,L+o,+o,L+o), interpolation='nearest')
    elif args.contactfreq and args.contactmode == 'splitoverlay':
        lotri = np.ones((L,L), dtype=bool)
        lotri[np.triu_indices(L,k=1)] = False
        hitri = np.zeros((L,L), dtype=bool)
        hitri[np.triu_indices(L,k=1)] =True

        cont = getM(np.load(args.contactfreq))
        cont = cm.gray_r(cont*0.2)
        main_ax.imshow(cont, origin='lower',
                       extent=(+o,L+o,+o,L+o), interpolation='nearest')

        upper = getM(pottsScore)
        upper[hitri] = np.nan
        img = main_ax.imshow(upper, origin='lower', cmap=alphared,
                             extent=(+o,L+o,+o,L+o), interpolation='nearest')
        cbar = contactfig.colorbar(img, cax=cbar_ax)
        cbar = DraggableColorbar(cbar,img)
        cbar.connect()
    else:
        img = main_ax.imshow(getM(pottsScore), origin='lower', cmap='gray_r',
                             extent=(+o,L+o,+o,L+o), interpolation='nearest')
        cbar = contactfig.colorbar(img, cax=cbar_ax)
        cbar = DraggableColorbar(cbar, img)
        cbar.connect()

    if args.title:
        plt.title(args.title)

    if args.grid:
        main_ax.set_xticks(np.arange(0, L, 10))
        main_ax.set_yticks(np.arange(0, L, 10))
        main_ax.grid(color='black', linestyle=':', linewidth=0.2)

    gridfig_size = tuple(0.2*x for x in (6+q, 6+q))

    margfig = plt.figure(figsize=gridfig_size, facecolor='white')
    margax = plt.axes([0, 0, 1, 1])

    Cfig = plt.figure(figsize=gridfig_size, facecolor='white')
    Cax = plt.axes([0, 0, 1, 1])

    Jfig = plt.figure(figsize=gridfig_size, facecolor='white')
    Jax = plt.axes([0, 0, 1, 1])

    ijpicker = PositionPicker(contactfig, (margax, Cax, Jax), alphamap_color,
                              J, h, ff, pottsScore)

    plt.show()

if __name__ == '__main__':
    main()
