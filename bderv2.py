#
# This software is copyrighted material (C) George Council 2021
#
# The library is provided under the GNU Public License version 3.0,
#   as found at http://www.gnu.org/licenses/gpl-3.0-standalone.html
#
import pdb
import itertools
from itertools import product
import numpy as np
from numpy import *
from numpy.linalg import inv,pinv,svd




class Bderv(object):
     def __init__(self, F, DH,dim):
       '''
       Inputs: 
          F  -- dict of vector fields; keys are binary words indexing D_b, while F[b] should yield F_b(rho)
          DH -- n x d array of surface normals such that DH(F_b) > 0 for all b
          dim -- vector field dimension d
       '''
       m,d = shape(DH)
       self.xi = None
       self.xf = None
       self.F  = F 
       self.DH = DH
       self.dim = d
       self.ess_dim =  m 
       self.B = array(tuple(itertools.product(set((0,1)),repeat=self.ess_dim)))

     def f(self,b):
       '''
       Input: binary vector b of length self.dim
       Output: length-d array F[b]
           Wrapper function for vector field dictionary making it callable and accounts for index conventions in python vs Alg. 2.
       '''
       b = b.astype(int)
       b[b==0]=1
       return self.F[tuple(b)]

     def __call__(self,dx):
        '''
          Inputs: dx -- length d array ; equivalently \delta y^{-}
          Outputs: B(dx) -- length d array; aeuqivalently \delta y^{+}
          Comments: Alg. 1, so just image point
        '''
        _,dv = self.Bof(dx)
        return dv

     def Bof(self,dx,re=False):
        '''
          Inputs: dx -- length d array ; equivalently \delta y^{-}
          Outputs: xf -- length d array; aeuqivalently B(\delta y^{-}) = \delta y^{+}
                   dx -- Saltation update without pre/post flow
        '''
        dt = 0
        b = -np.ones(len(self.DH),dtype=np.int)
        dx = np.array(dx)
        while np.any(b<0):
             tau = -np.dot(self.DH,dx)/np.dot(self.DH,self.f(b))
             tau[b > 0] = np.inf
             j = np.argmin(tau)
             dt += tau[j]
             dx += self.f(b)*tau[j]
             b[j] = +1
             if re == True:
               print(tau,j,b,dt)
        xf = dx+(-dt)*self.f(b)
        return dx,xf


##TO DO: Modify simplex point / forward simplex point to solve non standard arrangement H's.
class BdervExtra(Bderv):
    def _f(self,t,x,p):
       '''
       Inputs: t -- time, x -- length d state, p -- params
       Output: F(t,x,p) for x in D_b
       '''
       x = sign(x)
       x[x==0] = 1
       x = array([int(xi) for xi in x])
       return self.f[tuple(x[0:self.dim])]

    def simplex_points(self,mode='orth'):
       '''
       Determine initial simplicial points (\zeta_b's) in D_(-1) 
       '''
       xx = {} 
       m,d = shape(self.DH)
       for b in self.B:
           b = asarray(b,bool)
           x = zeros(d)
           bc = ~b
           assert tuple(bc) in self.B, "No complement %s defined. Check dimension???" % repr(bc)
           bci = tuple( (-1 if bi else 1) for bi in b )  # tuple index of complement of b
           bi = tuple( (1 if bi else -1) for bi in b )  # tuple index  of b
           if mode == 'orth':
             x[:m][~b] = -1.*self.F[bi][:m][~b]
             assert all(x<=0), "0-simplex is not contained in the lower hyperoctant"
           elif mode == 'sol':
             sp = self.DH.T
             self.Pr=dot(sp, dot( inv(dot(sp.T,sp)), sp.T))
             _,_,V = svd(self.Pr)
             self.Nc=r_[self.Pr,V[(m+1):]].T
             v = zeros(m)
             v[:m][b] = 0
             v[:m][~b] = -dot(self.DH, self.F[bi])[:m][~b]# -self.F[bi][:m][~b]
             x = dot(pinv(self.DH),v)
           else:
               assert False, "Mode not valid"
           bb = array(b,dtype=int)
           bb[bb==0] = -1
           xx[tuple(bb)] = x 
       if (d-m) > 0:
          L=[]
          for kk in range(d-m):
             v = zeros(d)
             v[m+kk] = 1
             L.append(v)
          L = asarray(L)
          D=zeros((d-m, d))
          
          D[:,m:] = eye(d-m)
          if (d-m) < 2:
             D = D.flatten()
          D = svd(self.DH)[-1][m:]
          xx['L'] = D
       self.xi = xx
    
    def forward_simplex_points(self,mode):
        '''
        Map from initial \zeta_b's to \phi(1,\zeta_b)
        '''
        xf = {}
        m,d = shape(self.DH)
        for b,xi in self.xi.items():
          if b != 'L':
            xf[b] = xi+self.F[b]
          else:

            xf[b] = xi + self.F[tuple(ones(m))]#xi + self.F[tuple(ones(m))]
        self.xf = xf
    
    def cachePoints(self,mode='orth'):
         '''
         Pre-compute and cache all simplex points
         '''
         self.simplex_points(mode)
         self.forward_simplex_points(mode)

    def projfb(self,f,g,dim):
        '''
        Wrapper function I + f g^T
        Inputs: 
            f : length d array -- vector field value
            g : length d array -- surface normal
        Ouputs:
            S : d x d array -- outer product that appears in saltation matrix sandwich 
        '''
        return eye(dim+1)  + outer(r_[1,-f],r_[0,g])/dot(g,f)
 
    def Bd(self,dx):
      '''
      This returns the selection function (matrix (B(sigma(dx)), and dv, found via saltation sandwich
      Inputs:  
            dx -- length d array -- tangent vector \delta x
      Outputs:
            A -- linear piece of B-derivative such that dot(A,dx) = B(dx) for all vectors that share transition sequence with dx
      '''
      b = self.wordOf(dx)   ### transition sequence of dx
      d = len(dx)
      m,_ = shape(self.DH)
      et = []
      for i in range(len(b)-1):    ##index convention wrapper
        idx = where(b[i+1]>b[i])[0][0]
        et.append(idx)
      A = eye(d+1)
      for ei,bi in zip(et,b):
         A = dot(self.projfb(self.F[tuple(bi.astype(int))],self.DH[ei],d),A)  ##compute ''sandwich'' of saltation matrix outer products
      V = c_[self.F[tuple(-ones(m))], eye(d)]  ###pre-impact update -- account for non zero flow time
      U = c_[self.F[tuple(ones(m))], eye(d)]    ###post--impact update -- '' 
      A = dot(U,dot(A,vstack([zeros(d+1),V]))) ##compute complete sandwich 
      return A[:,1:] ###return state sub-block (as A natively operates on flow domain (dt,dx))

    def wordOf(self,dx):
        """
        Inputs:
          dx -- d-dimensional vector of initial point
        Outputs:
          w -- sequence of binary words indicating order of guard crossings
        """
        dt = 0
        bw = [-np.ones(len(self.DH),dtype=np.int)]
        dx = np.array(dx)
        b = bw[-1].copy()
        while np.any(b<0):
             tau = -np.dot(self.DH,dx)/np.dot(self.DH,self.f(b))
             tau[b > 0] = np.inf
             j = np.argmin(tau)
             dt += tau[j]
             dx += self.f(b)*tau[j]
             b[j] = +1
             bw.append(array(b.copy()))
        return bw

    def Bm(self,dx,mode='orth'):
       '''
       This returns the selection function (matrix (B(sigma(dx)), and dv, found via simplex points
       '''
       if (self.xi == None) or (self.xf == None):
          self.cachePoints(mode)
       bw = self.wordOf(dx)
       m,d = shape(self.DH)
       P = asfarray([self.xi[tuple(z)] for z in bw]).T
       Q = array([self.xf[tuple(z)] for z in bw]).T
       if (d-m) > 0:
          P = c_[P, self.xi['L'].T]
       P = vstack([P, ones(d+1)])
       if (d-m) > 0:
         Q = c_[Q, self.xf['L'].T]
       B= dot(Q, inv(P))[:,:-1]
       return B, dot(B,dx)
       

