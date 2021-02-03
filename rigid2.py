import sys
assert sys.version_info[0] == 2, "This is a python 2 file"

from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from integro import odeDP5
from scipy.linalg import expm,logm
#from util import Animation
from itertools import product, combinations
from odepc import OdePC
from numpy import *
from numpy.linalg import det,inv,norm, pinv
from scipy import *
from itertools import product
import pdb
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import itertools
import cPickle
import time


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

DEBUG = []
def vectorize(fun):
    def vecf(y):
     out = []
     if ndim(y) == 1:
       y = y[newaxis,...]
     return asfarray([ fun(yi) for yi in y]).squeeze().copy()
    #vecf.__doc__ = "(auto vectorized)"+fun.__doc__
    return vecf
class rigidPoints(object):
    def __init__(self,mp, frame,IC):
        '''
        mp :  n x 4 array of points and masses : first col is m_i, 1:3 is x,y,z in body frame B_b:
            m1 x1 y1 z1,
            m2 x2 y2 z2,
            ...
        frame : homogenous representation of member of SE3 chosen w.r.t world basis B_w
        NOTE : FRAME AND MP ARE IN DIFFERENT COORDINATES
        IC  = [v,w]
            v := initial translational velocity in B_w
            w := initial angular velocity in B_w
        '''
        self.body = mp[:,1:]
        self.mass = sum(mp[:,0])  #total mass of point cloud
        self.frame = frame
        self.Ibody = sum([m*(dot(r,r)*eye(3) - outer(r,r)) for r,m in zip(self.body, mp[:,0])], axis=0)
        self.invIbody = inv(self.Ibody)
        self.IC = IC ###spurious currently

        self.sys = odeDP5(self.newtonEuler)
    	self.sys.event = self.foot_contact_world
        self.sys.aux = self.auxil


        self.wfsB = eye(4)
        self.wfsB[-1,-1] = 1
        #self.legs   = array([ [1, 5.8*pi/3, 0], [1,5.8*pi/3, 2*pi/3], [1,5.8*pi/3, 4*pi/3]])
        #self.legs_c = array([ [1, pi/2, 0], [1,pi/2, 2*pi/3], [1,pi/2, 4*pi/3]])
        self.legs = array([[3.24, 3.8, 0], [3.24, 3.8, pi]])
        self.legs_c = array([ [0.41,pi/2, 0], [0.41, pi/2, pi]])
        self.leg_td = zeros_like(self.legs)
        self.EVT = None
        self.con = zeros(len(self.legs))

        self.spm = diag([1,1,1,0,0,0])
        self.um  = 20*diag([0,0,0,1,1,1])


    def getICWorldCoords(self, vb):
        '''
        Input: vb in body coords
        Output: T(vb) - vb in world coordinates
        '''
        return dot(self.frame, asarray(vstack([vb,1])))

    def R3_to_so3(self,w):
        '''
        Parameterization of so(3) by R^3
        Input: w \in R^3 --- type: ARRAY
        Output: element of so(3) in as 3x3 matrix =:A s.t A = -A.T --- type: ARRAY
        '''
        a1,a2,a3 = w
        w_hat = array([[0,-a3, a2], [a3,0,-a1],[-a2, a1,0]])
        return w_hat.copy()

    def so3_to_R3(self,A):
        '''
        Returns coordinates of so(3) in vector form
        Input : A - 3x3 skew symmetric matrix of so(3) ---type : ARRAY
        Ouput : w s.t R3_to_so3(w)  = A --- type: ARRAY
        '''
        return array([A[2, 1], A[0, -1], A[1, 0]])

    def R6_to_se3(self,vv):
        '''
        Parameterizatino of se(3) by R^6
        Input: vv : [v,w]
                w : parameters in R^3 of rotation matrix in so(3) \
                v : translation part in R^3
        Output    : element of se(3) in 4x4 homogeneous matrix form
        '''
        v = vv[0:3]
        w = vv[3:,]
        w_hat = self.R3_to_so3(w)
        xi = zeros((4,4));
        xi[0:3,0:3] = w_hat
        xi[0:3,-1]  = v
        return xi.copy()

    def se3_to_R6(self,A):
        '''
        Returns coordinates of se(3) in R^6 s..t R6_to_s3(se3_to_R6(A)) = A
        Input: homogeneous representation A of xi \in se(3) --- TYPE: ARRAY
        Output : [v,w] vector in R^6 -- type: ARRAY
        '''
        R = A[0:3,0:3]
        v = A[0:3, -1]
        w = self.so3_to_R3(R)
        return hstack(array([v,w]))

    def inv_SE3(self,g):
        '''
        Finds group inverse element of g \in SE3
        Input: g in 4x4 homogeneous representation --- TYPE: ARRAY
        Output: g^-1 in 4x4 homogeneous representatino --- type:ARRAY
        '''
        R = g[0:3,0:3]
        v = g[0:3,-1]
        iv = zeros((4,4))
        iv[0:3,0:3] = R.T
        iv[0:3,-1 ] = -dot(R.T, v)
        iv[-1,-1] = 1
        return iv

    def adjoint(self,g):
        '''
        Detemines adjoint action of SE(3) on se(3)
        Inputs: g \in SE(3) represeted as 6x6 matrix in homogeneous coords --- type: ARRAY
        Output: Matrix representation of Adjoint action --- type: ARRAY
        '''
        R = g[0:3,0:3]
        v = g[0:3, -1]
        adg = zeros((6,6))
        vhat = self.R3_to_so3(v)
        adg[0:3,0:3] = R
        adg[0:3,3:6] = dot(vhat, R)
        adg[3:6, 3:6] = R
        return adg


    def body_frame_force(self,f,p):
        #https://gist.github.com/iizukak/1287876
        def gs_cofficient(v1, v2):
            return np.dot(v2, v1) / np.dot(v1, v1)

        def multiply(cocfficient, v):
            return map((lambda x : x * cofficient), v)

        def proj(v1, v2):
            return multiply(gs_cofficient(v1, v2) , v1)

        def gs(X):
            Y = []
            for i in range(len(X)):
                temp_vec = X[i]
                for inY in Y :
                    proj_vec = proj(inY, X[i])
                    temp_vec = map(lambda x, y : x - y, temp_vec, proj_vec)
                Y.append(temp_vec)
            return Y

        A = zeros((3,3))
        A[0,:] = f
        u,s,vt = svd(A)
        nidx = where( s < 1e-6)
        ns  = vt[nidx].T
        A[1:, :] = ns.T
        A = array(gs(A)).T
        A = A/norm(A,axis=0)
        return self.homoCoords(A,p)

    def AWrench_to_BWrench(self,g,S):
        '''
        map from se3 -> to se3 that transforms a wrench Fa in spatial coordinates to body coords
        Inputs: g - group action in homogeneous 4x4 matrix --- type:ARRAY
                S - wrench in S coordiates (R6) --- type: ARRAY
        Outputs: Fb - wrench in B coordinates (R6) --- type: ARRAY
        '''
        Fb = dot(self.adjoint(inv(g)).T, S)
        return Fb


    def q_to_array(self,q):
        '''
        Quaterinion to matrix representation
        '''
        s,vx,vy,vz = q
        return array([ [1-2*vy*vy-2*vz*vz, 2*vx*vz - 2*s*vz, 2*vx*vz+2*s*vy],
                    [2*vx*vy+2*s*vz, 1 - 2*vx*vx-2*vz*vz, 2*vy*vz-2*s*vx],
                    [2*vx*vz - 2*s*vy, 2*vy*vz+2*s*vx, 1 - 2*vx*vx-2*vy*vy]])

    def homoCoords(self,R,v):
        '''
        Put R,v in 4x4 homogeneous coordinates
        '''
        frame = zeros((4,4))
        frame[0:3,0:3] = R
        frame[-1,-1] = 1
        frame[0:3,-1] = v
        return frame

    def q_mult(self,q1, q2):
        '''
        Point-wise multiplication of quaternions q1 and q2 as 4-tuples
        '''
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
        z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
        return w, x, y, z

    def getWorldCoords(self,frame,vb):
        return dot(frame, asarray(vstack([vb,1])))

    def getWorldForces(self,frame,xx):
        fi = self.force(xx)
        self.getWorldCoords(frame, fi)

    def worldWrench(self,x):
        return 0

    def homoexpm(self,xi):
        v = xi[0:3].copy()
        w = xi[3:].copy()
        #return expm(self.R6_to_se3(xi))
        n = sqrt(sum(w*w))
        egm = zeros((4,4))
        A = sin(n)/n
        B = (1-cos(n))/(n*n)
        C = (1-A)/(n*n)
        if n != 0:
          wh = self.R3_to_so3(w).copy()
          #so =  eye(3) + wh*(sin(n)/n) + dot(wh,wh)*(1- cos(n))/(n*n)
          so3 = eye(3) + A*wh + B*dot(wh,wh)
          #r3  = (dot(eye(3) - so, dot(wh,v)) + dot(outer(w,w),v))/n
          r3 = eye(3) + B*wh + C*dot(wh,wh)
          egm[0:3,0:3] = so3.copy()
          egm[0:3,-1]  = dot(r3, v).copy()
        else:
          egm[0:3,0:3] = eye(3)
          egm[0:3,-1]  = v
        egm[-1,-1 ]  = 1
        return egm.copy()

    def homologm(self,g):
        '''
        Computes a matrix log of g s.t. expm(homologm(g)) = g
        Input: homogeneous 4x4 representation of g -- type: ARRAY
        Output: matrix log of g -- type: ARRAY
        '''
        R = g[0:3,0:3]
        p = g[0:3,-1]
        lgm = zeros((4,4))
        if allclose(R ,eye(3),atol=1e-6,rtol=1e-6):
            lgm[0:3,-1] = p/norm(p)
            th = norm(p)
            return th*lgm
        if allclose(trace(R),-1,atol=1e-6,rtol=1e-6):# == -1:
            th = pi
            w = (1/sqrt(2*(1+R[2,2])))*array([R[0,2], R[1,2], 1+R[2,2]])
            lgm[0:3,0:3]= self.R3_to_so3(w)
            lgm[0:3,-1] = p/norm(p)
            return th*lgm
        else:
            arg = (trace(R)-1)/2.
            if abs(arg) >= 1:
                arg = int(arg)
            th = arccos(arg)
            w_hat = (0.5/sin(th))*(R-R.T)
            Ginv = (1./th)*eye(3) - .5*w_hat + (1/th-.5*(1/tan(th/2.)))*dot(w_hat,w_hat)
            v  = dot(Ginv,p)
            lgm[0:3,0:3] = w_hat
            lgm[0:3,-1] = v
            return th*lgm

    def appWrenchBody(self,t,x,p):
        trans = 'pass'
        F = zeros(6)
        xi = x.T[0:6] #p,th - pos,ang
        xid = x.T[6:12] #v,w - vel, angvel

        #spring-damper wrench in body
        #F = -dot(self.spm, xi)-dot(self.um, xid)

        g = dot( dot(self.wfsB, self.homoexpm(xi)), self.frame)

        adg = self.adjoint(g)
        xis = real(self.se3_to_R6(self.homologm(g)))
        xids = dot(adg, xid)
        #F = -dot(self.spm, xis)-dot(self.um, xids)
        F = self.spring_wrenches(t,x,p)
        F = dot(diag([1,0,1,0,1,0]),F) #only take x,z,w_y components
        gf = self.mass*dot(self.adjoint(self.frame).T,asfarray([0,0,-10,0,0,0]))  ##gravity
        F  +=  dot(self.adjoint(inv(self.homoexpm(xi))),gf)
        F  +=  dot(self.adjoint(inv(self.homoexpm(xi))),dot(diag([0,1000,0,1000,0,1000]),xis)) ##restoring spring for y = 0
        F  +=  dot(self.adjoint(inv(self.homoexpm(xi))),dot(diag([0,1000,0,1000,0,1000]),xids)) ##restoring damper for y = 0
        return F


    def newtonEuler(self,t,xx,p):
        pb = xx[0:3]
        thb = xx[3:6]
        vb = xx[6:9]
        wb = xx[9:]
        Fb = self.appWrenchBody(t,xx,p)
        dvb = (1./self.mass)*(Fb[0:3]-cross(wb, self.mass*vb))
        dwb = dot(self.invIbody,Fb[3:]-cross(wb, dot(self.Ibody, wb)))
        if any(isnan([dvb,dwb])):
            pdb.set_trace()
        return hstack(array([vb, wb, dvb, dwb]))

    def homepad(self,v):
        if ndim(v) > 1:
            return vstack([v, ones((1, v.shape[1]))])
        if ndim(v) == 1:
            return hstack([v,1])

    def apply_g(self,q,R):
        '''
        Applies affine transformation R to v
        Input: R  - element of SO(3) in 4x4 homogeneous cooridinates --- type:ARRAY
               q  - 3 x n column-wise list of elements of R^3 --- type :ARRAY
        Output: Rv column-wise --- type: ARRAY
        '''
        return dot(R, self.homepad(q))[0:3,]

    def trans_body_pts(self,pp):
        '''
        Transform the self.body pts in the body frame into via g specified by: vv = expm(hat([p,th]))
        Inputs : vv is 6 x n where a column is [p,th] \in R^ param of transformation of SE(3) --- type : ARRAY
        Output: n x 4 x m  - n (sample count) x 4 (homo cords) x m (body pts) --- type:ARRAY
        '''
        bb = vstack([self.body.T, ones(self.body.T.shape[1])])
        traj = []
        q = []
        maps = []
        #traj.append(bb)
        for n in range(0,shape(pp.T)[0]):
            g = expm(self.R6_to_se3(pp.T[n]))
            traj.append(dot(g, bb))
            q.append(det(g))
            maps.append(g)
        return traj,q,maps

    def integrate(self,IC,tstart, tend):
        t,y = self.sys(IC, tstart,tend)
        return t,y

    def foot_location_world(self,t,y,p):
        l = []
        for leg in self.legs:
            g =  self.homoexpm(y[0:6])
            l.append(dot(g,self.homepad(self.S2C_iso(*leg)))[0:3])
        return asfarray(l).reshape(len(self.legs)*3)

    def hip_location_world(self,t,y,p):
        l = []
        for leg in self.legs_c:
            g =  self.homoexpm(y[0:6])
            l.append(dot(g,self.homepad(self.S2C_iso(*leg)))[0:3])
        return asfarray(l).reshape(len(self.legs)*3)

    def auxil(self,t,y,p):
        a = self.foot_location_world(t,y,p)
        b = self.hip_location_world(t,y,p)
        c = self.spring_wrenches(t,y,p)
        return asfarray(hstack([a,b,c,self.con.copy()]))

    def h_i(self,y,i):
      if ndim(y) == 1:
        y = y[newaxis,...]
      out = []
      for yi in y:
        g0 = self.homoexpm(yi[0:6])
        o = dot(g0, self.homepad(self.S2C_iso(*self.legs[i])))[0:3]
        out.append(self.quad_td(o))
      return asfarray(out)


    def set_con(self,t,y,p):
        g0 = self.homoexpm(y[0:6])
        for i in range(len(self.legs)):
            o = dot(g0,self.homepad(self.S2C_iso(*self.legs[i])))[0:3]
            if self.quad_td(o) > 0:
                self.con[i] = 1


    def quad_td(self,xx):
       x,_,z = xx
       return -(z+(x*x)/10.)

    def foot_contact_world(self,t,y,p):
        g0 = self.homoexpm(y[0][0:6])
        g1 = self.homoexpm(y[1][0:6])
        l = []
        #for leg in self.legs:
        self.last_con = self.con.copy()
        for i in range(len(self.legs)):
          #o = dot(g0,self.homepad(self.S2C_iso(*self.legs[i])))[0:3]
          #n = dot(g1,self.homepad(self.S2C_iso(*self.legs[i])))[0:3]
          #if ((o[-1] > 0) and (n[-1] < 0)) and (self.con[i] == 0):# or (o[-1] < 0) and (n[-1] > 0): -- flat ground
          #if (self.quad_td(o) < 0) and (self.quad_td(n) > 0) and (self.con[i] == 0): ## -- quad ground
          if self.h_i(y[0],i) < 0 and self.h_i(y[1],i) > 0 and self.con[i] == 0:
             l.append(1)
             #import pdb;pdb.set_trace()
             self.con[i] = 1
          else:
             l.append(0)
          if any(l):
            self.EVT = ['con',l]
         ###
        o = dot(g0,self.homepad(asfarray([0,0,0])))[0:3]
        n = dot(g1,self.homepad(asfarray([0,0,0])))[0:3]
        pt = asfarray([dot(g1, self.homepad(b))[-2] for b in self.body])
        return any(l)*1

    def foot_refine(self,t,y):
        g0 = self.homoexpm(y[0:6])
        l = []
        for i in range((len(self.legs))):
          o = dot(g0,self.homepad(self.S2C_iso(*self.legs[i])))[0:3]
          #l.append(o[-1].copy())
          #l.append(self.quad_td(o))
          l.append(self.h_i(y,i))
        l = asfarray(l)
        idx=(self.con!=self.last_con)
        l = l[idx]
        if len(l)==len(self.legs):
            return -l[0]*l[0]+l[1]*l[1]
        else:
            return l



    def compression(self,t,y,p):
       F = []
       legs = []
       legs_c = []

       ###find position of leg tip in world coordinates
       for leg in self.legs:
           g =  self.homoexpm(y[0:6])
           legs.append(dot(g,self.homepad(self.S2C_iso(*leg)))[0:3])
       legs = asfarray(legs)

       ###find hip in world coordiantes
       for lc in self.legs_c:
           g = self.homoexpm(y[0:6])
           legs_c.append(dot(g,self.homepad(self.S2C_iso(*lc)))[0:3])
       legs_c = asfarray(legs_c)

       ###determine compression length leg wise
       for i in range(len(legs)):
           l = legs[i]
           lc = legs_c[i]
           #if (l[-1] > 0) and (self.con[i] == 0):
           if self.con[i] == 0:
               F.append(zeros(3)) #if still in air (l_z > 0), apply no force
           elif array_equal(self.leg_td[i], zeros(3)):
              #this runs ONCE per contact (per leg) - so once contact occurs, it is assumed unbreakable
              n = asfarray([0,0,1])
              l0 = lc-l
              l0 = l0/norm(l0)
              d = -dot(l,n)/dot(l0,asfarray([0,0,1]))
              lp = d*l0+l
              self.leg_td[i] = lp ###find point of first touchdown - use as attachment point for rest of contact
              F.append(l-lp) ### vector from current leg tip (l) to contact point (lp)
           else:
              #F.append(l - self.leg_td[i])
              q = l-lc
              A = outer(q,q)/sum(q*q)
              F.append( (norm(l-lc)- norm(dot(A,l - self.leg_td[i])))*(l-self.leg_td[i]))
       if any(isnan(F)):
           pdb.set_trace()
       return asfarray(F)

    def spring_wrenches(self,t,x,p):
        xi = x.T[0:6].copy() #p,th - pos,ang
        xid = x.T[6:12].copy() #v,w - vel, angvel
        dl = self.compression(t,x,p).copy()
        F = []
        ###compute applied wrench for each leg based on contact status (Spring-Damper)
        for dli,leg,lc,cn,td in zip(dl,self.legs, self.legs_c,self.con,self.leg_td):

            g0 = zeros((4,4))
            g0[0:3,0:3] = eye(3) ###hip is pure translation
            g0[0:3,-1] = td
            g0[-1,-1] = 1

            #g1 = dot(inv(g0), self.homoexpm(xi[0:6]))
            g1 = inv(g0)

            vl = asfarray(hstack([dli,0.,0.,0.])) ###direction of leg compression as pure translation
            #vl = asfarray(hstack([self.S2C_iso(*leg)-self.S2C_iso(*lc),0,0,0]))
            if norm(vl != 0): ###If not 0, project hip velocity onto leg-compression direction in se3
              pxids = dot(self.adjoint(inv(g0)),xid)
              pixds = dot(outer(vl,vl)/sum(vl*vl), pxids)
            else:
              pxids = zeros(6)
            #spring
            Fa  = -2*cn*asfarray(hstack([dot(inv(g0),self.homepad(dli))[0:3],0.,0.,0.])) ###if cn == True, apply spring force
            #damping
            Fa += -10.*cn*pxids#dot(adg,xid) ### if cn == True, apply damping
            #Fb = dot(self.adjoint(inv(g1)).T, Fa)
            Fb = dot(self.adjoint(dot(inv(self.homoexpm(xi)), inv(g0))).T, Fa)
            F.append(Fb)
            if any(isnan(Fa)):
                pdb.set_trace()
        return sum(F,0) ###add leg wrenches together

    def S2C_iso(self,r=0.1,theta=0,phi=-pi/2):
        x = r*sin(theta)*cos(phi)
        y = r*sin(theta)*sin(phi)
        z = r*cos(theta)
        return array([x,y,z])

    def C2S_iso(self,x,y,z):
        r = (x**2+y**2+z**2)**0.5
        t = arccos(z/r)
        p = arctan2(y,x)
        return r,t,p

    def execute(self,IC,t0,tf,refine=False):
      from copy import deepcopy
      tb = [0]
      tc = t0
      yy = []
      tt = []
      l = []
      ICi = IC
      self.IC = ICi.copy()
      try:
          while tc < tf:
             print "integration starting at : " + repr(tc) + " " + ": " + repr(self.con)
             t,y = self.integrate(ICi,0,tf-tc)
             if self.EVT is not None:
              if self.EVT[0] == 'con':
               l.append(deepcopy(self.con))
               self.EVT = None
              else:
               raise "UNKNOWN EVENT"
             if refine and (tc+t[-1] < tf):
                 tr,yr = self.sys.refine(self.foot_refine)
                 ICi = yr[0:12].copy()
                 #print "Refined to " + repr(ICi)
             self.last_con=self.con.copy()
            # ICi = y[-1,0:12].copy()
             self.IC = ICi.copy()
             yy.append(y.copy())
             tt.append(t.copy()+tc)
             tb.append(t[-1].copy())
             tc = sum(asfarray(tb))
             #import pdb
             #pdb.set_trace()
      except Exception, e:
          print e
      finally:
          return tt,yy,l


d = 3
#Box at Origin
### NOTE!!! - Convention taken is that the origin of the Body frame is included first in pi - graphing convenience.
h = sqrt(2)/10
mi = 2*array([0,1,1,1,1,1,1,1,1])
po = array([[0,0,0],[1,0,h], [0,1,h], [0,-1,h], [-1,0,h], [1,0,-h],[0,1,-h],[0,-1,-h],[-1,0,-h]])
mp = hstack([mi[...,newaxis], po])


###Initial frame - make it the 'world' frame.

frame = zeros((d+1,d+1))
theta = 0#pi/8.
frame[0:d,0:d] = eye(3)
frame[0:d,-1] = array([0,0,0])
frame[-1,-1] = 1

if det(frame) > 1.0:
   print bcolors.FAIL + "ERR: frame not rigid!!! - re-run to try new frame" + bcolors.ENDC
else:
    B  = rigidPoints(mp, frame,[0,0])

    #close('all')
def yh(y):
    #padding function to take xz to xyz
    if ndim(y) == 1:
       y = y[newaxis,...]
    return asfarray([[yi[0], 0, yi[1], 0 ,yi[2],0, yi[3], 0, yi[4], 0, yi[5], 0] for yi in y]).squeeze()
test = 1
if test == 1:
    #B  = rigidPoints(mp, frame,[0,0])
    #IC = [v,w]  - v is trans vel, w in angular vel
    IC = zeros(6)
    IC[2] = 6

    def h0(y):
        y = array([y[0], 0, y[1], 0,y[2], y[3], 0, y[4], 0, 0, y[5], 0])
        return B.h_i(y,0)

    def h1(y):
        y = array([y[0], 0, y[1], 0,y[2], y[3], 0, y[4], 0, 0, y[5], 0])
        return B.h_i(y,1)

    def h_select(y, i):
      if i == 0:
        return h0(y)
      if i == 1:
        return h1(y)

    eps = 1e-6

    dt = 0.01
    YY = []
    def execute(IC,t0,tf,refine=True, constraints=None):
        '''
        constraints = list of contact state for start and end of integration, e.g, [[0,0],[1,1]] checks if started in [0,0] and ended in [1,1]
        '''
        B  = rigidPoints(mp, frame,[0,0])
        g0 = B.homoexpm(yh(IC)[0:6])
        c = zeros(2)
        for i in range(len(B.legs)):
          #o = dot(g0,B.homepad(B.S2C_iso(*B.legs[i])))[0:3]
          if h_select(IC,i) > 0:#o[-1] < 0:
              c[i] = 1
        print  bcolors.OKGREEN + "Initialized in corner:"  + repr(c) + bcolors.ENDC
        B.con = c
        if constraints is not None:
            cons =[]
            cons.append(copy(c))
        #x,z,w,xd,zd,wd
        #x,y,z,wx,wy,wz,xd,yd,zd,wdx,wdy,wdz
        idx = [0,2,4,6,8,10]
        ICt = array([IC[0], 0, IC[1], 0,IC[2], 0,IC[3], 0, IC[4],0, IC[5], 0])
        t,y,l = B.execute(ICt,t0,tf,refine=refine)
        YY.append([t,y])
        res = array([norm(yy[:,1]) for yy in y])
        if norm(res) > 1:
            print bcolors.WARNING + "WARNING!!!! - xz plane not well stabilized" + bcolors.ENDC
        print bcolors.OKGREEN + "Terminated at : " + repr(B.con) + bcolors.ENDC
        if constraints is not None:
            cons.append(B.con.copy())
            for i,o in zip(constraints, cons):
                i = asfarray(i)
                o = asfarray(o)
                if not array_equal(i,o):
                    print bcolors.WARNING + "WARNING!!!! - constraints not met; "  + repr(i) +  "!=" + repr(o)  + bcolors.ENDC
        return t, [array(yy[...,idx]) for yy in y]

    #x,z,w,xd,zd,wd
    IC = asfarray([0,6,0,0,0,0])
    t,y = execute(IC,0, 2, refine=True)
    from copy import deepcopy
    tb = deepcopy(t)
    yb = deepcopy(y)
    rho  = yb[0][-1]
    from copy import deepcopy
    GG = deepcopy(YY);
    #rho = array([  0,   3.35370309e+00,  -4.90043349e-09,
    #     0,  -7.19912457e+00,   8.75448712e-04])
    dt = 0.01
    #from util import jacobian_cdas as jac
    from jacob import jacobian_cdas as jac
    def hp(y):
        y = array([y[0], 0, y[1], 0,y[2], y[3], 0, y[4], 0, 0, y[5], 0])
        return array([h0(y), h1(y)])


    def h(yi):
        return asfarray(hstack([h0(yi),h1(yi),yi[0],yi[3],yi[4:6]])).squeeze()


    def g(y):
        '''
        Function to distort corner vector fields  -- must be full rank
        '''
        scl = 20
        out = []
        if ndim(y) == 1:
          y = y[newaxis,...]
        for yi in y:
            f1= yi[0]
            f2= yi[1]
            f3=yi[0]+scl*yi[1]
            f4=yi[3]+scl*yi[1]
            f5=yi[4]+scl*yi[1]
            f6=yi[5]+scl*yi[1]
            out.append([f1,f2,f3,f4,f5,f6])
        return asfarray(out).squeeze().copy()

    def sc(y):
        '''
        Rescaling function
        '''
        #scl = 0.01
        scl = 1
        out = []
        if ndim(y) == 1:
          y = y[newaxis,...]
        return asfarray([dot(diag([1,1,scl,scl,scl,scl]),yi) for yi in y]).squeeze().copy()


    Q = asfarray([[  0.,  0.,   0.,   0.,   0.,      0.],
               [  0.,  0.,   0.,   0.,   0.,      0],
               [  1,   0.,   0.,   0.,   0 ,     0],
               [  0,   0.,   0.,   -1.,   0,      0],
               [  0.,  0.,   0.,   0,    1.,     0],
               [  0.,  0.,   0.,   0.,   0,      1]])
    def H(x):
        x = asfarray(x)
        return r_[h(x)[0:2].copy(), dot(Q,x-rho)[2:].copy()]


    H = vectorize(H)
    dH = jac(H,eps*ones(6))
    ts=time.time()
    dhr = dH(rho)
    tf  = time.time()
    time_dh = tf-ts

    def fu(x):
        x=asfarray(x)
        return norm(H(x),axis=0)
    from scipy.optimize import minimize
    rho = minimize(fu, rho)['x']
    #YY = []
    YP = []
    findBd = True
    corner_mode = "EXACT"
    print bcolors.OKBLUE + "Finding Corner Vector Fields" + bcolors.ENDC
    if findBd == True:
        idh = lambda x : pinv(dH(x))
        rr = 2
        Fbb = {}
        eps = 1e-6
        for b in tuple(product(set((-1,1)),repeat=rr)):
          ICi = rho.copy() #
          ICi +=  sum([-2*eps*b[i]*-dhr[i,:] for i in range(2)],axis=0)
          print "At corner " + repr(b) + " have IC: " + repr(ICi)
          print "sanity check :-- H has value   "  + repr(sign(H(ICi)))
          g0 = B.homoexpm(yh(ICi)[0:6])
          c = zeros(2)
          for i in range(len(B.legs)):
              o = dot(g0,B.homepad(B.S2C_iso(*B.legs[i])))[0:3]
              if H(ICi)[i] > 0:#o[-1] < 0:
                  c[i] = 1
          print repr(c) + " " + repr(b)
          B.con = c
          idx = [0,2,4,6,8,10]
          if corner_mode == "LINE_FIT":
            t,y = execute(ICi,0, 0.01,refine=False)
            YP.append([deepcopy(t),deepcopy(y)])
            yd=y[0][0:5]
            td=t[0][0:5]

            r = polyfit(td,yd,1,full=True)
            figure();
            for tt,yy in zip(t,y):
                plot(tt,yy, '.-')
            tt = linspace(t[0][0],t[-1][-1], 5)
            plot(tt,array([m*tt+s for m,s in zip(*r[0])]).T, '-x');title(repr(b))
            axvline(t[0][-1])
            print bcolors.HEADER + "Residuadls of fit: " +  repr(norm(r[1])) + bcolors.ENDC
            #print norm(B.newtonEuler(0,yh(ICi),0)[idx]-r[0][0])
            Fbb[b] = r[0][0]
          if corner_mode == "NUM_DIFF":
            t,y = execute(ICi,0, 1,refine=False)
            YP.append([deepcopy(t),deepcopy(y)])
            y=y[0]
            t=t[0]
            Fbb[b] = (y[1]-y[0])/(t[1]-t[0])
          if corner_mode == "EXACT":
             Fbb[b] = B.newtonEuler(0,yh(ICi),0)[idx]
          if corner_mode == "AVG":
             Fbb[b] = mean(array([(y[i]-y[i-1])/(t[i]-t[i-1]) for i in range(10)]),0)
        

        #F = array(Fbb.values()).T
        #M = dot(F.T,F)
        #C = 1+rand(4,4)
        #Q = dot(C, dot(pinv(M),F.T))

        plot_patches = 1
        if plot_patches:
            figure()

            for yp in YP:
              t,y = yp
              for tt,yy in zip(t,y):
                for n,yi in enumerate(yy.T):
                  figure(1);subplot(2,3,n+1)
                  plot(tt,yi)
                  figure(2);subplot(2,3,n+1)
                  plot(tt[1:],diff(yi)/diff(tt))

            lbl = ['$x$','$z$','$\omega_y$','$\dot x$','$\dot z$','$\dot \omega_y$']
            for n,l in enumerate(lbl):
                   subplot(2,3,n+1)
                   ylabel(l)
            suptitle("Corner Vector Field Traj Patches")

        idx = sign(dot(pinv(Fbb.values()),ones(4)))
        dtx = diag(idx)
        #Fbb = {b:dot(dtx,fb) for b,fb in Fbb.iteritems()}
        Fb = {}
        
        for b in tuple(product(set((-1,1)),repeat=6)):
             Fb[b] = Fbb[b[0:2]].copy()
        if 0:
            Gb = {}
            for b,f in Fb.iteritems():
                Gb[b] = dot(dhr, f)
            print bcolors.HEADER + "Signs of corner vector field:"+ bcolors.ENDC
            print repr(sign(dot(dhr,asfarray(Fbb.values()).T)))
            assert all(sign(dot(dhr,asfarray(Fbb.values()).T))>0), "Corner vector field is not coherently oriented"


            print bcolors.OKBLUE+"Starting Bderv"+bcolors.ENDC
            s = Bderv(6,Gb)
            print bcolors.OKBLUE+"Finding initial simplex points"+bcolors.ENDC
            s.simplex_points()
            print bcolors.OKBLUE+"Integrating forward"+bcolors.ENDC
            t0 = time.time()
            s.forward_simplex_points()
            print bcolors.OKBLUE+"Integrating test point forward"+bcolors.ENDC
            o = OdePC(s.f)
            ICt = array([  1.12176845e-04,   2.56341539e+00,   1.53796591e-04,
             6.67978652e-04,   3.54281742e-04,   3.60306925e-04])
            ICtra = H(asfarray(ICt))
            assert all(ICtra<0), "Initialize condition for PC vector field is not in B_{-1}"
            #from pconstant import PC as OC
            #o = OC(Gb)

            tl,yl = o(ICtra, 0, 0.006, dt=0.001)

            print bcolors.OKBLUE + "Finding B-Derv" + bcolors.ENDC
            tc,idx,si,so,Bd = s.Bd(tl,yl,0)



        #ICt = array([  0,   2.56341539+0.01,  0.001,
        # 0,   -8.2,   0])
        ICt = rho + array([0,.01,-0.001,0,0,0])
        ICtra = ICt
        ICtra = H(asfarray(ICt))
        #from bdervNG import Gbderv
        Tgo = 0.002005
        Tgo = 0.002
        import imp
        foo = imp.load_source('bderv2', '/home/george/code/B-Derv-Geom/code/bderv2.py')
    
        if 1:
            ts = time.time()
            Gbb = {b:dot(dhr, f) for b,f in Fbb.iteritems()}
            DG = {}
            for i in range(2):
                v = zeros(6)
                v[i] = 1
                DG[i] = v
            tf = time.time()
            time_Gbb = tf-ts
            tot = 0
            n = 0
            while n < 1000:
                ta = time.time()
                #G = Gbderv(Gbb,DG,6,2)
                tb = time.time()
                #G.build_simplices(delt=Tgo/10)
                #tc = time.time()
                #tl,yl = G.go(ICtra,0,Tgo+Tgo/10, Tgo/10)
                #td = time.time()
                #st,en,Bd= G(tl,yl)
                te = time.time()
                #ST,EN,Q = G.bd(ICtra)
                tf = time.time()
                #M = dot(inv(dhr), dot(Q, dhr))
                n += 1
                Bc = foo.Bderv(Gbb,array(DG.values()), 2)
                Bn,dv2 = Bc.Bm(IC)
                M = dot(inv(dhr), dot(Bn,dhr))
                tg = time.time()
                tot += (tg-ta)
            print tot/n+time_Gbb
               
        M[abs(M)<1e-9] = 0
        #raise

        D = []
        Y = []
        
        def yo(IC):
            #Tgo = .03
            t,y = execute(IC,0,Tgo,refine=True,constraints=[[0,0],[1,1]])
            t = deepcopy(t)
            y = deepcopy(y)
            D.append([t,y])
            return y[-1][-1].copy()
        #from jacobian_cdas_par import jacobian_cdas_par as jac
        print bcolors.OKBLUE + "Finding numerical jacobian" + bcolors.ENDC
        t0 = time.time()
        eps=1e-6
        do_jac = 1;
        if do_jac == 1:
            dyo = jac(yo,eps*ones(6))
            dyoc = dyo(ICt)
            t1 = time.time()
            matshow(dyoc,cmap="inferno");title("num jac")
            print repr(t1-t0)
        fig1,ax1 = subplots()
        #M = dot(inv(dhr), dot(Bd, dhr))
        ax1.matshow(M,cmap="inferno");title("bd-" + repr(ICt[2])) 
        show_val = 1
        if show_val == 1:
            def truncate(f, n):
                '''Truncates/pads a float f to n decimal places without rounding'''
                s = '{}'.format(f)
                if 'e' in s or 'E' in s:
                    return '{0:.{1}f}'.format(f, n)
                i, p, d = s.partition('.')
                return '.'.join([i, (d+'0'*n)[:n]])
            for i in range(6):
                for j in range(6):
                    ax1.text(j, i, str(truncate(dyoc[i][j],2)), va='center', ha='center',color='red')
        lin_check = 0;
        if lin_check ==1:
            figure();
            plot(tl,yl, 'rx',alpha=0.3)
            for t,y in D:
                y = array([H(yi) for yi in y])
                for tt, yy in zip(t,y):
                    plot(tt,yy)
            title("Linearity check")
            figure();
            for yp in D:
                dyl = asfarray([ (yl[i+1,:]-yl[i,:])/(tl[i+1]-tl[i])  for i in range(len(tl)-1)])
                t,y = yp
                for tt,yy in zip(t,y):
                 yy = H(yy)
                 for n,yi in enumerate(yy.T):
                   subplot(2,3,n+1)
                   plot(tt[1:],diff(yi)/diff(tt), 'r.-')
                   plot(tl[1:], dyl[:,n], 'b-')

            lbl = ['$x$','$z$','$\omega_y$','$\dot x$','$\dot z$','$\dot \omega_y$']
            for n,l in enumerate(lbl):
                   subplot(2,3,n+1)
                   ylabel(l)



        raise
raise

print "Visualization..."

for t,y in YY:
    def vis_frames(stj,axlim):
        xmin,xmax,ymin,ymax,zmin,zmax = axlim
        fig_stj = plt.figure()
        ax_stj = fig_stj.add_subplot(1,1,1, projection='3d')
        body_pts = asarray([b[0:3,:] for b in stj[:]])
        com = asarray([b.T[0] for b in body_pts])
        color = itertools.cycle(cm.rainbow(linspace(0,1, shape(B.body)[0])))
        for pts in body_pts:
            for pt in pts.T:
              col = next(color)
              ax_stj.scatter(pt[0],pt[1],pt[2], c=col)
            ax_stj.plot(pts.T[:,0], pts.T[:,1], pts.T[:,2])
            ax_stj.set_xlim3d(xmin,xmax)
            ax_stj.set_ylim3d(ymin,ymax)
            ax_stj.set_zlim3d(zmin,zmax)
            ax_stj.set_xlabel('x')
            ax_stj.set_ylabel('y')
            ax_stj.set_zlabel('z')
        plot(com[:,0], com[:,1], com[:,2], 'k-', lw = 5)
        xx,yy = meshgrid(linspace(xmin,xmax,3),linspace(ymin,ymax,3))
        ax_stj.plot_surface(xx,yy,zeros_like(xx),alpha=0.2)
        plt.show()
        return fig_stj

    def vis_framesxz(stj,axlim):
        xmin,xmax,ymin,ymax,zmin,zmax = axlim
        fig_stj = plt.figure()
        ax_stj = fig_stj.add_subplot(1,1,1)
        body_pts = asarray([b[0:3,:] for b in stj[:]])
        com = asarray([b.T[0] for b in body_pts])
        color = itertools.cycle(cm.rainbow(linspace(0,1, shape(B.body)[0])))
        for pts in body_pts:
            for pt in pts.T:
              col = next(color)
              ax_stj.scatter(pt[0],pt[2], c=col)
            ax_stj.plot(pts.T[:,0], pts.T[:,2], color='b')
            ax_stj.plot(pts.T[[-1,0],0], pts.T[[-1,0], 2], color='b')
        plot(com[:,0], com[:,2], 'k-', lw = 5)
        #xx,yy = meshgrid(linspace(xmin,xmax,3),linspace(ymin,ymax,3))
        #ax_stj.plot_surface(xx,yy,zeros_like(xx),alpha=0.2)
        axhline(0)
        plt.show();grid('on')
        return fig_stj

    def anim_frameszx(stj,axlim):
        A  = Animation("fall.avi")
        xmin,xmax,ymin,ymax,zmin,zmax = axlim
        fig_stj = plt.figure()
        ax_stj = fig_stj.add_subplot(1,1,1)
        body_pts = asarray([b[0:3,:] for b in stj[:]])
        com = asarray([b.T[0] for b in body_pts])
        color = itertools.cycle(cm.rainbow(linspace(0,1, shape(B.body)[0])))
        for j,pts in zip(range(len(ftb)),body_pts):
            fig_stj.clf()
            ax_stj = fig_stj.add_subplot(1,1,1)
            for pt in pts.T:
              col = next(color)
              ax_stj.scatter(pt[0],pt[2], c=col)
            ax_stj.plot(pts.T[:,0], pts.T[:,2], 'k')
            ax_stj.plot(pts.T[[-1,1],0], pts.T[[-1,1],2], 'k')
            fslc = ftb[j,:]
            hslc = htb[j,:]
            ax_stj.plot(fslc[:,0], fslc[:,2],marker='x',color='k',linestyle='None')
            if yi[j,-2] == 1:
                    ax_stj.plot(fslc[0,0], fslc[0,2],marker='x',color='r',linestyle='None')
            if yi[j,-1] == 1:
                    fig.gca().plot(fslc[1,0], fslc[1,2],marker='x',color='r',linestyle='None')
            ax_stj.plot(hslc[:,0], hslc[:,2],marker='x',color='b',linestyle='None')
            for i in range(2):
                plot([fslc[i,0], hslc[i,0]],[fslc[i,2], hslc[i,2]], 'k')
            plt.show()
            #raw_input()
            #fig_stj.clf()
            #ax_stj = fig_stj.add_subplot(1,1,1)
            A.step()
        #A.stop()


    axlim = array([-8,8,-8,8,-8,8])
    i = 0
    FT = []
    HP = []
    for ti,yi in zip(t,y):
        pb = yi[:,0:3].T
        th = yi[:,3:6].T
        btj,q,Rt = B.trans_body_pts(vstack([pb,th]))
        gi = B.inv_SE3(frame) #spatial to body map in R6
        stj = [dot(frame, pts) for pts in btj]
        ftb = yi[:,12:12+len(B.legs)*3]
        ftb = reshape(ftb, (len(ftb),2,3))
        htb = yi[:,12+len(B.legs)*3:12+len(B.legs)*3+6]
        htb = reshape(htb, (len(htb),2,3))
        FT.append(ftb)
        HP.append(htb)
        fig= vis_framesxz(stj,axlim)
        for j in range(len(ftb)):
            fslc = ftb[j,:]
            hslc = htb[j,:]
            fig.gca().plot(fslc[:,0], fslc[:,2],marker='x',color='k',linestyle='None')
            if yi[j,-2] == 1:
                    fig.gca().plot(fslc[0,0], fslc[0,2],marker='x',color='r',linestyle='None')
            if yi[j,-1] == 1:
                    fig.gca().plot(fslc[1,0], fslc[1,2],marker='x',color='r',linestyle='None')
            fig.gca().plot(hslc[:,0], hslc[:,2],marker='x',color='b',linestyle='None')
            for i in range(2):
                plot([fslc[i,0], hslc[i,0]],[fslc[i,2], hslc[i,2]])
        title(repr(i))
        i +=1

    raw_input("Hit Enter to see next trial")
    close('all');

alld=False
if alld:
    SM = array([dot(inv(dhr), sti)+rho for sti in st])
    nr = SM.shape[0]
    ba = 0.5
    nic = ones(nr)*(1-ba)/nr+identity(nr)*ba

    dpt = dot(nic,SM)

    K = []
    for pt in dpt:
      dyo = jac(yo,eps*ones(6))
      dyoc = dyo(pt)
      K.append(dyoc.copy())


    P =asarray([[svd(a-b)[1][0] for a in K] for b in K ])
    #P =[svd(a-b)[1][0] for a,b in combinations(K,2)]
