#!/usr/bin/env python3
from scipy import *
import pylab as plt
from matplotlib.colors import Normalize
from matplotlib import cm, transforms
import matplotlib as mpl
import numpy as np
import sys, time, argparse
from Bio.Alphabet import IUPAC
import seqload, changeGauge
from matplotlib.colors import LinearSegmentedColormap

def getL(size):
    return int(((1+sqrt(1+8*size))//2) + 0.5)

def getLq(J):
    return getL(J.shape[0]), int(sqrt(J.shape[1]) + 0.5)

def getUnimarg(ff):
    L, q = getLq(ff)
    ff = ff.reshape((L*(L-1)//2,q,q))
    marg = array([sum(ff[0],axis=1)] + [sum(ff[n],axis=0) for n in range(L-1)])
    return marg/(sum(marg,axis=1)[:,newaxis]) # correct any fp errors

def indepF(fab):
    L = int( (1+np.sqrt(1+8*fab.shape[0]))/2 + 0.5)
    nB = int(np.sqrt(fab.shape[1]) + 0.5)

    fabx = fab.reshape((fab.shape[0], nB, nB))
    fa1, fb2 = np.sum(fabx,axis=2), np.sum(fabx,axis=1)
    fafb = np.array([np.outer(fa, fb).flatten() for fa,fb in zip(fa1, fb2)])
    return fafb

def getM(x):
    L = getL(len(x))
    M = zeros((L,L))
    M[triu_indices(L,k=1)] = x
    return M + M.T

class DraggableColorbar(object):
    def __init__(self, cbar, mappable):
        self.cbar = cbar
        self.mappable = mappable
        self.press = None
        #self.cycle = sorted([i for i in dir(plt.cm) if hasattr(getattr(plt.cm,i),'N')])
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
            t = transforms.offset_copy(text._transform, x=-ex.width, units='dots')
    else:
        for s,c in ls:
            text = ax.text(x,y,s,alpha=c, transform=t,
                    va='bottom',ha='center',**kw)
            text.draw(ax.figure.canvas.get_renderer())
            ex = text.get_window_extent()
            t = transforms.offset_copy(text._transform, y=ex.height, units='dots')

class PositionPicker:
    def __init__(self, contactfig, margax, alphacolor, J, h, ff, score):
        self.presstime = 0
        self.margax = margax
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

                i,j = int(event.xdata-0.5), int(event.ydata-0.5)
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
                drawMarg(self.alphacolor, q, i,j, ffij, Jij, hi, hj, score, self.margax)
                self.margax.figure.canvas.draw()

def drawMarg(alphacolor, q, i,j, marg, J, hi, hj, score, ax):
    graytext = lambda x: {'text': "{:.2f}".format(x), 'color': cm.gray_r(fnorm(x))}
    bwrtext = lambda x: {'text': "{:.2f}".format(x), 'color': cm.bwr(fnorm(x))}
    rwbtext = lambda x: {'text': "{:.2f}".format(x), 'color': cm.bwr_r(fnorm(x))}
    mapi = alphacolor[i]
    mapj = alphacolor[j]

    ax.set_axis_off()
    ax.set_xlim(0, 3*(q+4))
    ax.set_ylim(0, 1*(q+5))

    ax.text((3*(q+4)+2)/2.0, q+17, "Pair {}, {}".format(i+1,j+1), 
               horizontalalignment='center', 
               verticalalignment='center',
               fontsize=16)

    fnorm = Normalize(0, 0.1, clip=True)
    drawGrid(ax, 3, 1, 1, q, q, 
             [[graytext(x) for x in r] for r in marg],
             mapi, mapj, 'Bimarg', 
             list(map(graytext, sum(marg, axis=1))), 
             list(map(graytext, sum(marg, axis=0))),
             labeltext=alphatext)
    
    fnorm = Normalize(-1, 1.0, clip=True)
    C = marg - outer(sum(marg, axis=1), sum(marg, axis=0))
    Cmax = np.max(np.abs(C))
    drawGrid(ax, q+6.5, 1, 1, q, q, 
             [[rwbtext(x) for x in r] for r in C/Cmax],
             mapi, mapj, 'C * {}'.format(str(Cmax)), None, None,
             labeltext=alphatext)

    fnorm = Normalize(-1, 1.0, clip=True)
    drawGrid(ax, q + 13, 1, 1, q, q, 
             [[bwrtext(x) for x in r] for r in J],
             mapi, mapj, 'J score={:.2f}'.format(score), 
             list(map(bwrtext, hi)), list(map(bwrtext, hj)),
             labeltext=alphatext)

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
    parser = argparse.ArgumentParser(description='Run DCA')
    parser.add_argument('bimarg')
    parser.add_argument('couplings')
    parser.add_argument('-alphamap', help='map from 21 letters to alpha')
    parser.add_argument('-unimarg21', help='21 letter univariate marginals')
    parser.add_argument('-contactfreq', help='contact frequency file')
    parser.add_argument('-contactmode',
                        choices=['split', 'overlay', 'splitoverlay'], 
                        default='overlay',
                        help='how to draw contact map')
    parser.add_argument('-score', 
                        choices=['fb', 'fbw', 'fbwsqrt', 'DI', 'Xij'], 
                        default='fbwsqrt')
    parser.add_argument('-gauge', choices=['nofield', 0, 'w', 'wsqrt'], 
                                             default='wsqrt')
    parser.add_argument('-alpha', default="ABCDEFGH")
    parser.add_argument('-regions', help='comma separated list of indices')
    parser.add_argument('-title', help='Figure title')

    args = parser.parse_args(sys.argv[1:])

    alpha = args.alpha
    alpha21 = '-' + IUPAC.protein.letters

    ff = np.load(args.bimarg)
    J = np.load(args.couplings)

    unimarg = getUnimarg(ff)
    
    L, q = getLq(J)

    if args.gauge == 'nofield':
        h, J = changeGauge.fieldlessEven(zeros((L,q)), J)
    elif args.gauge == '0':
        h, J = changeGauge.zeroGauge(zeros((L,q)), J)
    elif args.gauge == 'w':
        h, J = changeGauge.weightedGauge(zeros((L,q)), J, weights=ff)
    elif args.gauge == 'wsqrt':
        h, J = changeGauge.weightedGauge(zeros((L,q)), J, weights=sqrt(ff))

    if args.score == 'fb':
        h0, J0 = changeGauge.zeroGauge(h, J)
        pottsScore = sqrt(sum(J0**2, axis=1))
    elif args.score == 'fbw':
        w = ff
        hw, Jw = changeGauge.weightedGauge(zeros((L,q)), J, weights=w)
        pottsScore = sqrt(sum((Jw*w)**2, axis=1))
    elif args.score == 'fbwsqrt':
        w = sqrt(ff)
        hw, Jw = changeGauge.weightedGauge(zeros((L,q)), J, weights=w)
        pottsScore = sqrt(sum((Jw*w)**2, axis=1))
    elif args.score == 'Xij':
        C = ff - indepF(ff)
        X = np.sum(C*J, axis=1)
        pottsScore = -X
    else:
        raise Exception("Not yet implemented")
    save('score', pottsScore)
    
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
                    linds = [alpha21.index(c) for c in g]
                    f = array([unimarg21[l,i] for i in linds])
                    let_dat = zip(g, f/sum(f))
                    let_dat.sort(key=lambda x: x[1], reverse=True)
                    clet.append(let_dat)
                alphamap_color.append(clet)
    else:
        alphamap_color = [[[(c, 1.0)] for c in alpha] for i in range(L)]
    
    ss = 0.4
    
    contactfig = plt.figure()


    if args.contactfreq and args.contactmode == 'overlay':
        cont = getM(np.load(args.contactfreq))
        # assume grayscale [0,1]
        cont = cm.gray_r(cont*0.2)
        plt.imshow(cont, origin='lower',
                     extent=(+0.5,L+0.5,+0.5,L+0.5), interpolation='nearest')

        scores = getM(pottsScore)
        img = plt.imshow(scores, origin='lower', cmap=alphared,
                     extent=(+0.5,L+0.5,+0.5,L+0.5), interpolation='nearest')
        cbar = plt.colorbar()
        cbar = DraggableColorbar(cbar,img)
        cbar.connect()
    elif args.contactfreq and args.contactmode == 'split':
        lotri = ones((L,L), dtype=bool)
        lotri[triu_indices(L,k=1)] = False
        hitri = zeros((L,L), dtype=bool)
        hitri[triu_indices(L,k=1)] =True

        upper = getM(pottsScore)
        upper[hitri] = nan
        img = plt.imshow(upper, origin='lower', cmap='Blues', 
                     extent=(+0.5,L+0.5,+0.5,L+0.5), interpolation='nearest')
        cbar = plt.colorbar()
        cbar = DraggableColorbar(cbar,img)
        cbar.connect()

        lower = getM(np.load(args.contactfreq))
        lower[lotri] = nan
        plt.imshow(lower, origin='lower', cmap='Reds', 
                     extent=(+0.5,L+0.5,+0.5,L+0.5), interpolation='nearest')
    if args.contactfreq and args.contactmode == 'splitoverlay':
        lotri = ones((L,L), dtype=bool)
        lotri[triu_indices(L,k=1)] = False
        hitri = zeros((L,L), dtype=bool)
        hitri[triu_indices(L,k=1)] =True

        cont = getM(np.load(args.contactfreq))
        cont = cm.gray_r(cont*0.2)
        plt.imshow(cont, origin='lower',
                     extent=(+0.5,L+0.5,+0.5,L+0.5), interpolation='nearest')

        upper = getM(pottsScore)
        upper[hitri] = nan
        img = plt.imshow(upper, origin='lower', cmap=alphared,
                     extent=(+0.5,L+0.5,+0.5,L+0.5), interpolation='nearest')
        cbar = plt.colorbar()
        cbar = DraggableColorbar(cbar,img)
        cbar.connect()
    else:
        img = plt.imshow(getM(pottsScore), origin='lower', cmap='Blues', 
                     extent=(+0.5,L+0.5,+0.5,L+0.5), interpolation='nearest')
        cbar = plt.colorbar()
        cbar = DraggableColorbar(cbar, img)
        cbar.connect()

    if args.regions:
        regions = [int(x) for x in args.regions.split(',')]
        for r in regions:
            plt.axvline(r+0.5, color='k', alpha=0.2)
            plt.axhline(r+0.5, color='k', alpha=0.2)

    if args.title:
        plt.title(args.title)

    margfig = plt.figure(figsize=(ss*(3*(q+4)), ss*1*(q+5)), facecolor='white')
    margax = plt.axes([0, 0, 1, 1])
    ijpicker = PositionPicker(contactfig, margax, alphamap_color, J, h, ff, pottsScore)

    plt.show()

if __name__ == '__main__':
    main()
