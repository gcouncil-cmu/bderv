#!/usr/bin/env python
# coding: utf-8
import sympy as sm
from integro import odeDP5
from bderv2 import Bderv
import numpy as np
from numpy import array,asfarray,pi,sin,cos,sqrt,diag,hstack,vstack,linspace,zeros,ones_like,angle,exp,real,imag
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import sympy as sm
from integro import odeDP5
from bderv2 import Bderv
import sympy as sym
from jacob import jacobian_cdas as jac



####Section 0
class chair(object):
   def __init__(self,m,I,th,k,b,l,IC,mm=False):
      self.m = m
      self.I = I
      self.th =th
      self.k  = k
      self.b  = b
      self.l  = l
      self.g = -9.8
      self.IC = IC
      self.rho = None
      self.M = diag([self.m,self.m,self.I])
      self.mm = mm
      self.blt = 0.01


   def _rho(self): 
       '''
       For a given aerial initial condition IC, solves for the point rho, where both legs touchdown at x,xdot =0, w,wdot = 0.
       Inputs: None
       Outputs: None
       '''
       def find_rho(y):
          return abs(self.H1(0,y,0)) + abs(self.H2(0,y,0))
       res=minimize(find_rho, .5, method='Nelder-Mead')
       y0 = res.x[0]
       self.rho = array([self.IC[0], y0, 0, 0, -sqrt(2*-self.g*y0),0])

   def build_sys(self):
     '''
     Generates numerical hybrid system for later integration from symbolic expressions
     Inputs: None
     Outputs: None
     '''
     self.air = odeDP5(self.air_vf)
     def air_td(t,y,p):
           y0 = y[0][0:3]
           y1 = y[1][0:3]
           v = (self.H1(*y0) < 0) and (self.H1(*y1) > 0)
           u = (self.H2(*y0) < 0) and (self.H2(*y1) > 0)
           return (v or u)
     self.air.event = air_td
     self.rl  = odeDP5(self.right_leg_vf)
     def rl_td(t,y,p):
           y0 = y[0][0:3]
           y1 = y[1][0:3] 
           return (self.H2(*y0) < 0) and (self.H2(*y1) > 0)
     self.rl.event = rl_td
     self.ll  = odeDP5(self.left_leg_vf)
     def ll_td(t,y,p):
           y0 = y[0][0:3]
           y1 = y[1][0:3] 
           return (self.H1(*y0) < 0) and (self.H1(*y1) > 0)
     self.ll.event = ll_td
     self.bl  = odeDP5(self.both_leg_vf)

   def build_sample(self):
     '''
     Evaluates non-linear vector fields F_b at self.rho to define sampled vf self.Ft. DH is already defined by self.build for efficiency reasons
     Inputs: None
     Outputs: None
     '''
     self.Ft = {(-1,-1): asfarray(self.air_vf(0,self.rho, 0)),
          (1,-1): asfarray(self.right_leg_vf(0,self.rho, 0)),
          (-1,1): asfarray(self.left_leg_vf(0,self.rho, 0)),
          (1,1):  asfarray(self.both_leg_vf(0,self.rho, 0))}
 
   def var(self,IC):
      '''
      Computes variational pre and post impact matrices Dphi(s,rho) and Dphi(t-s,x)
      Inputs : initial point -- presumably any x so that for some t \in (0,1), \phi_{air}(t, IC) = rho
      Outputs: dxd arrays D1, D2 -- D1 is pre impact D_xPhi, D2 is post-impact D_{rho} phi
      '''
      def _h1(t,x):
          return self.H1(*x[0:3])
      def _h2(t,x):
          return self.H2(*x[0:3])
      _,y = self.air(IC,0,1)
      idx= argmax(r_[self.H1(*y[-1,0:3]),self.H2(*y[-1,0:3])])
      if idx == 0:
        s,yr=self.air.refine(_h1)
      if idx == 1:
        s,yr=self.air.refine(_h2)
      def _aa(s):
          def aa(x):
            _,yr=self.air(x,0,s)
            return yr[-1]
          return aa
      Dair = jac(_aa(s), 1e-4*ones(6))
      D1 = Dair(IC)
      def _gg(x):
           _,y = self.bl(x, 0, self.blt,dt=0.001)
           return y[-1]
      Dbl = jac(_gg, 1e-4*ones(6))
      D2 = Dbl(self.rho)
      return D1,D2



   def go(self,IC,Tr=None):
       '''
       Integrates nonlinear vector field with odeDP5 with event detection
       Inputs: 
           dl -- d-length array, initial condition
       Outputs:
           [t0,t1,y2],[y0,y1,y2] -- length 3-lists of pairs ti,yi with are the times ti, and state yi, for a trajectory in smooth mode i
            simulating the chair starting in the air and ending with both legs touching down, .1 seconds after touchdown
       '''
       t,y=self.air(IC,0,1,dt=0.001)
       idx= argmax(r_[self.H1(*y[-1,0:3]),self.H2(*y[-1,0:3])])
       #print("H"  + str(idx) +  " crossing event")
       ###
       def _h1(t,x):
          return self.H1(*x[0:3])
       def _h2(t,x):
          return self.H2(*x[0:3])
       ###
       if idx == 0:
          tr,yy = self.air.refine(_h1)
          y[-1]=yy
          t1,y1=self.rl(yy,0,1,dt=.001)
          tb,yy =self.rl.refine(_h2)
       if idx == 1:
          _,yy = self.air.refine(_h2)
          y[-1]=yy
          t1,y1=self.ll(yy,0,1,dt=.001)
          _,yy = self.ll.refine(_h1)
       ###
       if 'y1' in locals():
          #print('bl')
          t2,y2 = self.bl(yy,0,self.blt)
          rt = t[-1]+t1[-1]+self.blt
          if Tr is not None:
           if rt < Tr:
             t2,y2 = self.bl(yy,0,Tr-rt)
             
       return [t,t1,t2],[y,y1,y2]


   def air_vf(self,t,yy,p):
         x,y,w,dx,dy,dw = yy
         return self.fa(*yy).flatten()

   def right_leg_vf(self,t,yy,p):
       x,y,w,dx,dy,dw = yy
       return  self.f1(*yy).flatten()


   def left_leg_vf(self,t,yy,p):
       x,y,w,dx,dy,dw = yy
       return  self.f2(*yy).flatten()
 
   def both_leg_vf(self,t,yy,p):
       x,y,w,dx,dy,dw = yy
       return self.fg(*yy).flatten()


   def build(self):
    '''
    Builds vector field and event functions symbolically, then caches lambdify'd versions for numerical evalution
    Inputs: None
    Outputs: None
    '''
    m, I, x, y, w, dx, dy, dw, l, g, th, kappa, beta = sym.symbols(
        r'm, I, x, y, \omega, \dot{x}, \dot{y}, \dot{\omega}, \ell, g, \theta, \alpha, \beta', 
        )

    q = sym.Matrix([[x,y,w]]).T
    dq = sym.Matrix([[dx,dy,dw]]).T
    M = sym.Matrix([[m,0,0],[0,m,0],[0,0,I]])
    f = sym.Matrix([[0,m*g,0]]).T
    a = -sym.Matrix([[ (x + l*sym.cos(th-w))**2 + (y - l*sym.sin(th-w)),(x - l*sym.cos(w+th))**2 + (y - l*sym.sin(w+th))]]).T
    Da = sym.Matrix.hstack(*[sym.diff(a,_) for _ in q])
    Dh = sym.Matrix.hstack(Da,sym.Matrix.zeros(2,3))
    

    ## vectors in kernel of Da
    v0=sym.Matrix([1/Da[0,0],0,-1/Da[0,-1]])
    v1=sym.Matrix([1/Da[1,0],0,-1/Da[1,-1]])
    Dap=sym.Matrix(hstack([v0,v1])).T

    # # vector fields

    # In[11]:
    B=sym.Matrix(diag([1.,1,1]))
    F = {}
    for b in [(-1,-1),(+1,-1),(-1,+1),(+1,+1)]:
      ddq = f
      for j in [0,1]:
        if b[j] > 0:
          ddq -= (kappa * a[j] + beta * (Da[j,:] * dq)[0]) * Da[j,:].T 
      F[b] = (sym.expand(sym.Matrix.vstack(dq,M.inv()*ddq)))


    ###mode-dependent  'inputs'
    if self.mm == True:
        F[(+1,-1)] += sym.Matrix(vstack([0*dq,+0.5*beta * (Da[0,:] * dq)[0]*Da[0,:].T]))
        F[(-1,+1)] += sym.Matrix(vstack([0*dq,+0.5*beta * (Da[1,:] * dq)[0]*Da[1,:].T]))

    ##saltation products
    I6 = sym.Matrix.eye(6)
    self.M12 = ( (I6 + (F[(+1,+1)] - F[(+1,-1)])*Dh[1,:]/Dh[1,:].dot(F[(+1,-1)])) 
      * (I6 + (F[(+1,-1)] - F[(-1,-1)])*Dh[0,:]/Dh[0,:].dot(F[(-1,-1)])) )
    self.M21 = ( (I6 + (F[(+1,+1)] - F[(-1,+1)])*Dh[0,:]/Dh[0,:].dot(F[(-1,+1)])) 
          * (I6 + (F[(-1,+1)] - F[(-1,-1)])*Dh[1,:]/Dh[1,:].dot(F[(-1,-1)])) )
    #build callables for numerical evaluation
    pars = {kappa:self.k, beta:self.b, th:self.th,l:self.l,m:self.m,I:self.I,g:self.g}
    ae = -a.subs(pars)
    Dhe = Dh.subs(pars)
    Fe = {b:(F[b].subs(pars)) for b in F}
    self.Fe = Fe
    self.H1 = sym.lambdify([x,y,w],-ae[0])
    self.H2 = sym.lambdify([x,y,w],-ae[1])
    self._rho() #solve for simulatenous touchdown point
    self.DH = asfarray(Dhe.subs({x:self.rho[0],y:self.rho[1],w:self.rho[2]}))
    self.fa = sym.lambdify([x,y,w,dx,dy,dw],Fe[(-1,-1)].T)
    self.f1 = sym.lambdify([x,y,w,dx,dy,dw],Fe[(+1,-1)].T)
    self.f2 = sym.lambdify([x,y,w,dx,dy,dw],Fe[(-1,+1)].T)
    self.fg = sym.lambdify([x,y,w,dx,dy,dw],Fe[(+1,+1)].T)
    self.M12 = self.M12.subs({I:self.I,m:self.m,x:self.rho[0],y:self.rho[1],w:self.rho[2],dx:self.rho[3],dy:self.rho[4],dw:self.rho[5], kappa:self.k,beta:self.b, th:self.th,l:self.l})
    self.M21 = self.M21.subs({I:self.I,m:self.m,x:self.rho[0],y:self.rho[1],w:self.rho[2],dx:self.rho[3],dy:self.rho[4],dw:self.rho[5], kappa:self.k,beta:self.b, th:self.th,l:self.l})
#Section 1
##non-smooth ouptut -- mode-dependent change in Beta
print("Building symbolic equations for forced")
IC= array([0,1.,0,0,0,0])
C = chair(1,1,pi/4,1,1,.5,IC,True)
C.build()
C.build_sys()
C.build_sample()
print("Building sampled forced system")
B = Bderv(C.Ft,C.DH,6)


##smooth output -- no mode dependent forcing
print("Building symbolic equations for unforced")
Cs = chair(1,1,pi/4,1,1,.5,IC,False)
Cs.build()
Cs.build_sys()
Cs.build_sample()
print("Building sampled unforced system")
Bs = Bderv(Cs.Ft,C.DH,6)


####Section 2
###visual B-derivative
dv = linspace(-1,1,101)
dvv= zeros(len(dv))
dvs = zeros(len(dv))
for ii in range(len(dv)):
    v = zeros(6);v[2] = dv[ii];
    dvv[ii] = B(v)[-2]
    dvs[ii] = Bs(v)[-2]
    
###Section 3
####jacobian comparison
D = []
def gf(IC): 
   Ta,a = C.air(IC,0,1)
   def _f(t,x): 
         return C.H1(*x[:3]) 
   t,a = C.air.refine(_f)
   tt,b = C.bl(a,0,C.blt+.005) #dosh!
   Tf = Ta[-1] + tt[-1]
   #print(str(Tf))
   def _gf(x):
    t,y =C.go(x, Tr=Tf) 
    D.append(t[-1][-1]-Tf)
    return y[-1][-1]-b[-1]
   return _gf


v = array([0,0,-.01,0,0,0])
IC = C.rho + asfarray([0,.02,0,0,0,0])
pp = gf(IC)     

print("Begining timing tests...")
import time
ta = time.time()
df = jac(pp, 1e-4*ones(6))

pt=IC+v
dfr = df(pt)
tb = time.time()

t0 = time.time()
D1,D2=C.var(IC)
from bderv2 import BdervExtra
te = time.time()
B = BdervExtra(C.Ft,C.DH,6)
M1 = B.Bd(v)
M1p = B.Bm(v, mode='sol') ##M1 better equal M1p equal C.M12!!!!
S = dot(D2,dot(M1,D1))
tee = time.time()
t1 = time.time()
print("Timing tests complete")
print("Finite difference took " + str(tb-ta) + " seconds")
print("B-derv took " + str(tee-te) + " seconds")



######################
#plotting
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches


font = {'family':'sans-serif', 'size':10}
mpl.rc('font', **font)
mpl.rcParams['lines.linewidth'] = 5
mpl.rcParams['lines.markersize'] = 20

slope_line_params = {'linestyle':'--', 'dashes':(1, 1)}
line_params = {'lw':8, 'ms':12}
pos_color = 'b'
pos_slope_color = (.5, .5, 1.)
neg_color = 'r'
neg_slope_color = (1, .5, .5)

open_circle_params = {'marker':'.', 'mfc':'w', 'mew':4}

ax_line_params = {'lw':4, 'color':'k'}
sim_color = 'purple'  

#parameters for drawings / animations
body_color = 'k'
left_color = 'b'
right_color = 'r'
default_leg_style = {'linestyle':'-', 'lw':5, 'color':'grey'}
default_gnd_style = {'facecolor':'brown', 'edgecolor':'black', 'hatch':'/', 'fill':True}

default_flywheel_style = {'marker':'o', 'markersize':20, 'color':'darkgrey',
                          'markeredgecolor':'dimgrey', 'mew':3}
default_foot_style = {'marker':'o', 'markersize':10}
default_spring_leg_style = {'lw':3, 'color':'grey'}
default_bdy_style = {'linestyle':'-', 'lw':10, 'color':'black'}
p = {}
p['mb'] = 1
p['mf'] = 1
p['Ib'] = 1
p['g'] = 1
p['l'] = 1/2.
p['wh'] = p['l']/2.
p['COM'] = array([0,.75])





###
def draw_config(p,ax,act=False):
    COM = p['COM']
    if ax is None:
      fig, ax = plt.subplots(2,2)
    handles = {}
    ax, gnd_handle = draw_ground(ax, p, z=0.25,depth=0.05,xc=0,width=5)
    handles.update(gnd_handle)
    ax, bdy_handle = draw_body(ax, p,COM)
    handles.update(bdy_handle)
    draw_leg(ax,p,act=act)
    #draw_dd(ax,p)
    return ax,handles

def draw_body(ax,p,COM):
      x,y = COM
      if 'bdy_style' in p:
        bdy_style = p['bdy_style']
      else:
        bdy_style = default_bdy_style
      rect = patches.Rectangle((x-.25, y), .5, .1, **bdy_style)
      ax.add_patch(rect)
      handles = {'body': rect} 
      return ax,handles

def draw_dd(ax,p):
     t = linspace(-1,1,5)
     #plot(t,-.1*t*t)
     v = asfarray([t, -.1*t*t]).T
     n = asfarray([-.2*t,ones_like(t)]).T
     for vi,ni in zip(v,n):
          print(vi[0])
          z = spring(0,1,a=1, b=1, c=.1)
          z.imag *= .5
          ang = angle(ni[0]+1j*ni[1])
          z = z*.3*exp(1j*ang)
          ax.plot(real(z)+vi[0],-imag(z)+vi[1],lw=2,color='blue')
          ax.plot(real(z[25]) + vi[0], -imag(z[25])+vi[1], marker='o',color='green')

def draw_leg(ax,p,act=False):
     xc,yc = p['COM']
     ll = p['l']
     l = linspace(0,ll,10)
     xhp1 = xc - .2
     xhp2 = xc + .2
     ax.plot([xc,xhp1],[yc,yc-l[-1]],color='gray');ax.plot(xhp1,yc-l[-1],color='red',marker="o", ms='10')
     ax.plot([xc,xhp2],[yc,yc-l[-1]],color='gray');ax.plot(xhp2,yc-l[-1],color='blue',marker="o",ms='10')
     ###actuator
     #if act:
     #    ax.plot(xhp1+0, yc-ll/2, marker='s',color='blue')
     #    ax.plot(xhp2+0, yc-ll/2, marker='s',color='blue')


     

def draw_ground(ax, p, z=.25, depth=1, xc=0, width=5,gnd='c'):
  '''
  Draw a patch representing the ground

  inputs:
    ax - axis to draw on
    p - parameter dict
    z - z height
    depth - how tall to draw the path
    xc - center of path
    width - width of the patch
  '''

  if 'gnd_style' in p:
    gnd_style = p['gnd_style']
  else:
    gnd_style = default_gnd_style
  if gnd == 'flat':
      rect = patches.Rectangle((xc-width/2, z-depth), width, depth, **gnd_style)
      ax.add_patch(rect)
      handles = {'ground':rect}
  elif gnd == 'c':
      t = linspace(-1,1,100)
      arc = ax.plot(t, -.1*t*t, lw='5',color='darkred');plt.axis([-1,1,-1,1])
      handles = {'ground':arc}
  else:
      raise
  return ax, handles




def spring(z0, z1, a=.2, b=.6, c=.2, h=1., p=4, N=100):
    '''
    Generate coordinates for a spring (in imaginary coordinates)

    From - Sam Burden

    input:
      z0, z1 - initial/ final coordinates (imaginary)
      a, b, c - relative ratio of inital straight to squiggle to final straight
                line segments
      h - width of squiggle
      N - number of points
    '''

    x = np.linspace(0., a+b+c, N)
    y = 0.*x
    mb = int(N*a/(a+b+c))
    Mb = int(N*(a+b)/(a+b+c))
    y[mb:Mb] = np.mod(np.linspace(0., p-.01, Mb-mb), 1.)-0.5
    z = ((np.abs(z1-z0)*x + 1.j*h*y)) * np.exp(1.j*np.angle(z1-z0)) + z0
    return z


fig,ax = plt.subplots(2,2)
draw_config(p,ax=ax[0,0],act=True);ax[0,0].axis([-1,1,-0.3,1]);ax[0,0].set_yticklabels([]);ax[0,0].set_xticklabels([]);ax[0,0].tick_params(bottom=False,left=False)
draw_config(p,ax=ax[0,1],act=False);ax[0,1].axis([-1,1,-0.3,1]);ax[0,1].set_yticklabels([]);ax[0,1].set_xticklabels([]);ax[0,1].tick_params(bottom=False,left=False)




lw=5
fs=18
plt.subplot(2,2,3);
plt.plot(dv[dv<=0],dvs[dv<=0],c='b',lw=lw);
plt.plot(dv[dv>0],dvs[dv>0],c='r',lw=lw);
#plt.xlabel(r'$\delta \theta^-$');
#plt.ylabel(r'$\delta \dot{\theta}^+$');
plt.axis([min(dv),max(dv),min(dvv),0.05]);
#plt.title(r'$C^r$ output')
plt.xticks([-1,-.5,0,.5,1],fontsize=fs)
plt.yticks([-0.5  , -0.375, -0.25 , -0.125, -0.   ],fontsize=fs)
#ax[1,1].set_yticklabels([])



plt.subplot(2,2,4);
plt.plot(dv[dv<=0],dvv[dv<=0],c='b',lw=lw);
plt.plot(dv[dv>0],dvv[dv>0],c='r',lw=lw);
#plt.xlabel(r'$\delta \theta^-$');
#plt.ylabel(r'$\delta \dot{\theta}^+$');
plt.axis([min(dv),max(dv),min(dvv),0.05]);
plt.xticks([-1,-.5,0,.5,1],fontsize=fs)
#plt.yticks(fontsize=15)
#plt.title(r'$PC^r$ output');
a = gca();a.set_yticklabels([])

plt.show()
plt.tight_layout()


