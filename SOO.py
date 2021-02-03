
from numpy import split
from integro import odeDP5
from itertools import product
from odepc import OdePC

import bderv2 as bderv
import pdb

def drawmat(M, t='M', a = 0):
        
        fig1,ax1 = subplots()
        ax1.matshow(M,cmap="winter",alpha=a);title(t)
        d=len(M)
        show_val = 1
        if show_val == 1:
            def truncate(f, n):
                '''Truncates/pads a float f to n decimal places without rounding'''
                s = '{}'.format(f)
                if 'e' in s or 'E' in s:
                    return '{0:.{1}f}'.format(f, n)
                i, p, d = s.partition('.')
                return '.'.join([i, (d+'0'*n)[:n]])
            for i in range(d):
                for j in range(d):
                    ax1.text(j, i, str(truncate(M[i][j],2)), va='center', ha='center',color='k')
            al = linspace(0.5, d-1.5, d-1)
            for a in al:
                plot([a, a],[-0.5,d-0.5],'k',alpha=0.5)
                plot([-0.5,d-0.5],[a, a],'k',alpha=0.5)

            show()

class SOO(object):
    def __init__(self,alpha,beta,delta):
      self.alpha = alpha
      self.beta  = beta
      self.delta = delta
      assert self.delta < self.alpha, "Delta must be less than alpha"


    def F(self,t,xx,p,sgn):
      x,dx = split(xx,2)
      dxx = self.alpha*ones_like(x)-self.beta*dx-self.delta*sign(x)
      return r_[dx,dxx]

    def _call(self, IC, t0,tf):
        f = lambda t,y,p : self.F(t,y,p, sign(split(IC,2)[0]))
        self.s = odeDP5(f)
        def evt(t,y,p):
            yb,ya = y
            ay,ady = split(ya,2)
            by,bdy = split(yb,2)
            return any(sign(by) != sign(ay))
        self.s.event = evt
        t,y = self.s(IC,t0,tf)
        return t,y

    def __call__(self,IC,t0,tf):
        tc = t0
        tb = [t0]
        ICi = IC
        yy = []
        tt = []
        while tc < tf:
            t,y = self._call(ICi, 0,tf-tc)
            tb.append(t[-1])
            ICi = y[-1]
            tt.append(t+tc)
            yy.append(y)
            tc = sum(tb)
        return tt,yy

d = 2
al=1
bl=.4
dl = 0.5
S = SOO(al, bl, dl)
IC = -rand(2*d)
t,y = S(IC, 0, 5)


eps = 0.01
nu =.4
rho = r_[zeros(d), nu*ones(d)]
Fb = {}
for b in tuple(product(set((-1,1)),repeat=d)):
     ICi =  rho + r_[eps*array(b), zeros(d)]
     Fb[b] = S.F(0,ICi,0, sign(b))

Gb = {}
for b in tuple(product(set((-1,1)),repeat=d)):
    Gb[b] = Fb[b[0:d]]

DH = c_[eye(d), zeros((d,d))]
Bc = bderv.Bderv(Gb,DH,2)
v = array([-.1,-.1,0,0])
B,dv = Bc.Bm(v)
drawmat(B,'Sue',a=0.5)
