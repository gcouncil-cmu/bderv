from bderv2 import BdervExtra as Bderv
from bderv2 import Bderv as Bbase
from time import sleep
from pylab import *
import pdb

def drawmat(M, t='M'):
        
        fig1,ax1 = subplots()
        ax1.matshow(M,cmap="winter",alpha=0.0);title(t)
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

def ball(Bc):
    x = -0.5*Bc.F[(-1,-1)]
    V=array([0.18*array([cos(t),sin(t)]) for t in linspace(0,2*pi,100)])
    U = []
    for vi in V:
       y = Bc.Bof(vi)
       U.append(y[-1]+0.5*Bc.F[(1,1)])
    return V+x,array(U)
 

if __name__=="__main__":
  #cases: 2d-contract, 2d-asym, 2d-angled, 2d-nonlinear, 3d-contract, 3d-degen, rigid -- note that the 'squishy chair' is in rigid2.py, not here
  test = '2d-contract'
  if test == '2d-contract':
        dl = +0.25
        nu = .75
        Fb = {(-1,-1): array([nu+dl,nu+dl]),
          (-1,1) : array([nu+dl,nu-dl]),
          (1,-1) : array([nu-dl,nu+dl]),
          (1,1)  : array([nu-dl,nu-dl])}
        #for _,v in Fb.items():
        #     v += rand(2)
        IC = asfarray([-1,-.75])
        Dh = {}
        S = eye(2)*(nu-dl)/(nu+dl)
        def f(b):
           b=b.astype(int)
           b[b==0]=1
           return Fb[tuple(b)]
        s = linspace(-.75,.6,7)
        X,Y = meshgrid(s,s)
        FX = zeros_like(X)
        FY = zeros_like(Y)
        for i in range(len(s)):
           for j in range(len(s)):
            x = X[i,j]
            y = Y[i,j]
            b = sign(array([x,y]))
            fx,fy = f(b)
            FX[i,j] = fx
            FY[i,j] = fy
        close('all')
        lw = 3
        ms = 20
        figure()
        quiver(X,Y,FX,FY,color='gray',scale=20,width=.005,headwidth=5,headlength=5,headaxislength=5)
        axis('scaled')
        axhline(0,color='r',lw=1.5*lw);axvline(0,color='b',lw=1.5*lw)
        DH = eye(2)
        v = -array([0.25,0])
        Bc = Bderv(Fb,DH, 2)
        dx,dv =Bc.Bof(v) 
        B,dv2 = Bc.Bm(v)
        xp=-.5*f(-ones(2))
        plot([0,xp[0]],[0,xp[1]] ,'m.-',label='x(0)',color='purple',ms=ms,lw=lw);
        x1=.5*f(ones(2))
        plot([0,x1[0]],[0,x1[1]] ,'m.-',label='x(1)',color='purple',ms=ms,lw=lw);
        ####
        V,U = ball(Bc)
        VV=V+Bc.F[(-1,-1)]
        okl = ':'
        ###B_w
        D=array([dot(B,vi) for vi in V])+Bc.F[(1,1)]
        IDX1=V[:,0]<V[:,1]
        IDX2=V[:,0]>V[:,1]
        v1 = V[IDX1]
        v2 = V[IDX2]
        idx=argmax(diff(v2));v2=vstack([v2[idx:],v2[:idx]]);
        d1 = D[IDX1]
        d2 = D[IDX2];d2=vstack([d2[idx:],d2[:idx]]);
        plot(v1[:,0],v1[:,1], lw=lw, ls='-',alpha=1, color='gold')
        plot(d1[:,0],d1[:,1], lw=lw,ls='-', label='B_w',alpha=1,color='gold')
        #plot(d2[:,0],d2[:,1], lw=4,ls=okl, label='B_w',alpha=0.5,color='gold')
        ###B_w'
        Bw,_ = Bc.Bm(-v)
        D=array([dot(Bw,vi) for vi in V])+Bc.F[(1,1)]
        d1 = D[IDX1]
        d2 = D[IDX2];d2=vstack([d2[idx:],d2[:idx]]);
        plot(v2[:,0],v2[:,1], lw=lw, ls='-',alpha=1, color='green')
        plot(d2[:,0],d2[:,1], lw=lw,ls='-', label='B_w',alpha=1,color='green')
        plot(d1[:,0],d1[:,1], lw=4, ls=okl,label='B_w',alpha=0.5,color='green') 
        plot(d2[:,0],d2[:,1], lw=4,ls=okl, label='B_w',alpha=1,color='gold')
        ###
        #plot(U[:,0],U[:,1], lw=1, color='k',ls='-.')
        ###
        plt.xticks([-.6,-.3,0,.3,.6],fontsize=30)
        plt.yticks([-.6,-.3,0,.3,.6],fontsize=30)
        drawmat(B,test);show()
  if test == '2d-angled':
        dl = +0.25
        nu = .75
        Fb = {(-1,-1): array([nu+dl,nu+dl]),
          (-1,1) : array([nu+dl,nu-dl]),
          (1,-1) : array([nu-dl,nu+dl]),
          (1,1)  : array([nu-dl,nu-dl])}
        IC = asfarray([-1,-.75])
        DH = array([[1.1,.4],[.4,1.1]])
        S = eye(2)*(nu-dl)/(nu+dl)
        def f(b):
           b=b.astype(int)
           b[b==0]=1
           return Fb[tuple(b)]
        s = linspace(-1,1,25)
        X,Y = meshgrid(s,s)
        FX = zeros_like(X)
        FY = zeros_like(Y)
        for i in range(len(s)):
           for j in range(len(s)):
            x = X[i,j]
            y = Y[i,j]
            b = sign(dot(DH, array([x,y])))
            fx,fy = f(b)
            FX[i,j] = fx
            FY[i,j] = fy
        quiver(X,Y,FX,FY);
        plot(s,-.4/1.1*s,color='b')
        plot(s,-1.1/.4*s, color='orange')   
        v = -array([0.25,0])
        Bc = Bderv(Fb,DH, 2)
        dx,dv =Bc.Bof(v) 
        B,dv2 = Bc.Bm(v,mode='sol')
        plot(*-0.5*f(-ones(2)), 'mo', label='x(0)')
        plot(*(v-0.5*f(-ones(2))), 'mx', label='x(0)+\delta x')
        plot(*v, 'gx',label='delta x');plot(*dv, 'rx',label='delta x(1)');plot(*0.5*f(ones(2)), 'bo',label='x(1)');legend()
        print("dv: "  +str(dv)) 
        print("x(1) + \delta x - x(1)" + str(x - 0.5*f(ones(2))));axis([min(s),max(s),min(s), max(s)])
        V,U = ball(Bc)
        plot(V[:,0],V[:,1]);plot(U[:,0],U[:,1])
        drawmat(B, t=test);draw();show();
  if test == '2d-asym':
        dl = +0.25
        nu = .75
        Fb = {(-1,-1): array([nu+dl,nu+dl]),
          (-1,1) : array([nu+dl,nu]),
          (1,-1) : array([nu-dl,nu+dl]),
          (1,1)  : array([1,.5])}
        IC = asfarray([-1,-.75])
        Dh = {}
        S = eye(2)*(nu-dl)/(nu+dl)
        def f(b):
           b=b.astype(int)
           b[b==0]=1
           return Fb[tuple(b)]
        s = linspace(-.65,.65,7)
        X,Y = meshgrid(s,s)
        FX = zeros_like(X)
        FY = zeros_like(Y)
        for i in range(len(s)):
           for j in range(len(s)):
            x = X[i,j]
            y = Y[i,j]
            b = sign(array([x,y]))
            fx,fy = f(b)
            FX[i,j] = fx
            FY[i,j] = fy
        lw = 3
        ms = 20
        figure()
        quiver(X,Y,FX,FY,color='gray',scale=20,width=.005,headwidth=5,headlength=5,headaxislength=5)
        axis('scaled')
        axhline(0,color='r',lw=1.5*lw);axvline(0,color='b',lw=1.5*lw)
        DH = eye(2)
        v = -array([0.25,0])
        Bc = Bderv(Fb,DH, 2)
        dx,dv =Bc.Bof(v) 
        B,dv2 = Bc.Bm(v)
        xp=-.5*f(-ones(2))
        plot([0,xp[0]],[0,xp[1]] ,'m.-',label='x(0)',color='purple',ms=ms,lw=lw);draw()
        x1=.5*f(ones(2))
        plot([0,x1[0]],[0,x1[1]] ,'m.-',label='x(1)',color='purple',ms=ms,lw=lw);draw()
        print("dv: "  +str(dv)) 
        print("x(1) + \delta x - x(1)" + str(x - 0.5*f(ones(2))))
        ####
        V,U = ball(Bc)
        VV=V+Bc.F[(-1,-1)]
        okl = ':'
        ###B_w
        D=array([dot(B,vi) for vi in V])+Bc.F[(1,1)]
        IDX1=V[:,0]<V[:,1]
        IDX2=V[:,0]>V[:,1]
        v1 = V[IDX1]
        v2 = V[IDX2]
        idx=argmax(diff(v2));v2=vstack([v2[idx:],v2[:idx]]);
        d1 = D[IDX1]
        d2 = D[IDX2];d2=vstack([d2[idx:],d2[:idx]]);
        plot(v1[:,0],v1[:,1], lw=lw, ls='-',alpha=1, color='gold');draw()
        plot(d1[:,0],d1[:,1], lw=lw,ls='-', label='B_w',alpha=1,color='gold')
        plot(d2[:,0],d2[:,1], lw=4,ls=okl, label='B_w',alpha=0.5,color='gold')
        ###B_w'
        Bw,_ = Bc.Bm(-v)
        D=array([dot(Bw,vi) for vi in V])+Bc.F[(1,1)]
        d1 = D[IDX1]
        d2 = D[IDX2];d2=vstack([d2[idx:],d2[:idx]]);
        plot(v2[:,0],v2[:,1], lw=lw, ls='-',alpha=1, color='green')
        plot(d2[:,0],d2[:,1], lw=lw,ls='-', label='B_w',alpha=1,color='green')
        plot(d1[:,0],d1[:,1], lw=4, ls=okl,label='B_w',alpha=0.5,color='green') 
        ###
        #plot(U[:,0],U[:,1], lw=1, color='k',ls='-.')
        ###
        plt.xticks([-.6,-.3,0,.3,.6],fontsize=30)
        plt.yticks([-.6,-.3,0,.3,.6],fontsize=30)
        drawmat(B, t=test);
        draw();show();
  if test == '3d-degen':
        dl = +0.25
        nu = .75
        Fb = {(-1,-1): array([nu+dl,nu+dl,1]),
          (-1,1) : array([nu+dl,nu-dl,1]),
          (1,-1) : array([nu-dl,nu+dl,0.5]),
          (1,1)  : array([nu-dl,nu-dl,0.5])}
        IC = asfarray([-1,-.75])
        Dh = {}
        S = eye(2)*(nu-dl)/(nu+dl)
        def f(b):
           b=b.astype(int)
           b[b==0]=1
           return Fb[tuple(b)]
        DH = asfarray([[1,0,0],[0,1,0]])
        v = -array([0.25,0,.1])
        Bc = Bderv(Fb,DH, 2)
        dx,dv =Bc.Bof(v) 
        B,dv2 = Bc.Bm(v,mode='sol')
        fig = figure()
        ax = fig.gca(projection='3d')
        tt = linspace(-1,1,6)
        X,Y,Z = meshgrid(tt,tt,tt)
        DX = zeros_like(X)
        DY = zeros_like(Y)
        DZ = zeros_like(Z)
        d = range(len(tt))
        for i in range(len(tt)):
            for j in range(len(tt)):
                for k in range(len(tt)):
                    f = Bc.f(sign([X[i,j,k], Y[i,j,k]]))
                    DX[i,j,k] = f[0]
                    DY[i,j,k] = f[1]
                    DZ[i,j,k] = f[2]
        dx,dy = meshgrid(tt,tt)
        dz = 0*dx
        ax.plot_surface(dx,dy,dz,alpha=0.2)
        ax.plot_surface(dx,dz,dy,alpha=0.2)
        ax.plot_surface(dz,dy,dx,alpha=0.2)
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        drawmat(B, t=test)
  if test == 'rigid':
        Fb = {(-1,
          -1): array([8.20000002e+00, 8.19999999e+00, 8.05318842e-09, 1.23917213e-07,
            1.00000000e+01, 0.00000000e+00]),
         (-1,
          1): array([ 8.20000002e+00,  8.19999999e+00,  8.05318842e-09, -6.61014086e-01,
            -2.59727048e-01,  3.63723449e+01]),
         (1,
          -1): array([ 8.20000002e+00,  8.19999999e+00,  8.05318842e-09,  6.61014408e-01,
            -2.59727098e-01, -3.63723444e+01]),
         (1,
          1): array([ 8.20000002e+00,  8.19999999e+00,  8.05318842e-09,  2.28230856e-07,
            -1.05020000e+01,  1.04670386e-06])}

        IC=array([-4.06167703e-01, -4.01191016e-01,  2.39364421e-08, -1.14255012e-08,
        7.00000009e-01,  9.25543634e-09])
        DH = hstack([eye(2), zeros((2,4))])
        Bc = Bderv(Fb,DH, 2)
        B,dv2 = Bc.Bm(IC)
        dhr = array([[ 0.39648391, -1.        , -2.59415251,  0.        ,  0.        ,
                 0.        ],
               [-0.39648391, -1.        ,  2.59415251,  0.        ,  0.        ,
                 0.        ],
               [ 1.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                 0.        ],
               [ 0.        ,  0.        ,  0.        , -1.        ,  0.        ,
                 0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ,  1.        ,
                 0.        ],
               [ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
                 1.        ]])
        M = dot(inv(dhr), dot(B, dhr))
        drawmat(M, title=test)

  if test == '2d-nonlinear':
        n = 15
        s = linspace(-1,1,n)
        X,Y = meshgrid(s,s)
        FX = zeros_like(X)
        FY = zeros_like(Y)
        FSX = zeros_like(X)
        FSY = zeros_like(Y)
        H1 = zeros_like(X)
        H2 = zeros_like(X)
        def h1(xx):
           x,y = xx
           return x**3 - y
        def h2(xx): 
           x,y = xx
           a = -pi/4
           R = asfarray([[cos(a), sin(a)],[-sin(a), cos(a)]])
           u,v = dot(R, array([x,y]))
           return u*u-v
        DH = array([[0, 1],[1,1]])
        sf = 0.75
        Fnl = {(-1,-1): lambda xx: (1+.5*sf*cos(xx[0]),1+.5*sf*sin(xx[1])),
              (1, -1): lambda xx:  (.5+0.5*sf*cos(2*pi*xx[0]),1+0.5*sf*sin(2*pi*xx[1])),
              (-1 ,1): lambda xx:  (1+0.5*sf*cos(xx[0]),.5+.5*sf*sin(xx[1])),
              (1,  1): lambda xx:   (.5+.25*sf*cos(xx[0]),.5+.25*sf*sin(xx[1]))}
        def f(xx):
            b = sign(array([h1(xx),h2(xx)])).astype(int)
            b[b==0] = 1
            v = Fnl[tuple(b)](xx)
            print(norm(v))
            return v
        def samp():
           S = {}
           for i in Fnl.items():
              b,fx = i 
              S[b] = asfarray(fx(zeros(2)))
           return S
        Fb = samp()
        def fs(b):
           b=b.astype(int)
           b[b==0]=1
           return Fb[tuple(b)]
        for i in range(n):
           for j in range(n):
            x = X[i,j]
            y = Y[i,j]
            fx,fy = f(array([x,y]))
            fsx,fsy = fs(sign(dot(DH,array([x,y]))))
            FX[i,j] = fx
            FY[i,j] = fy
            FSX[i,j] = fsx
            FSY[i,j] = fsy
            H1[i,j] = h1(array([x,y]))
            H2[i,j] = h2(array([x,y]))
        quiver(X,Y,FX,FY,alpha=0.25);
        quiver(X,Y,FSX,FSY,alpha=1);axis('scaled');
        contour(X,Y, H1, levels=[0],colors='blue',alpha=0.25)
        contour(X,Y, H2, levels=[0],colors='orange',alpha=0.25)
        plot(s, -s, color='red'); plot(s, 0*s,color='blue') ## linear 
        v = -array([0.25,0])
        Bc = Bderv(Fb,DH, 2)
        dx,dv =Bc.Bof(v) 
        for it in Fb.items():
             b,ekr = it
             print(dot(DH,ekr))
        B,dv2 = Bc.Bm(v,mode='sol')
        for pts in zip(Bc.xi.items(),Bc.xf.items()):
            (_,xi),(_,xf) = pts
            xi *= .25
            xf *= .25
            plot(xi[0], xi[1], 'mo')
            plot(xf[0], xf[1], 'mx')
            plot([xi[0],xf[0]],[xi[1],xf[1]], '--',color='green')
        drawmat(B, t=test);show()

  if test == '3d-contract':
        dim = 3
        Fb = {(-1, -1, -1): array([1,  1,  1 ]),
         (-1, -1, 1): array([ 1, 1, .4]),
         (-1, 1, -1): array([1,  .4,  1]),
         (-1, 1, 1): array([ 1,  .4,  .4]),
         (1, -1, -1): array([ .4, 1,  1]),
         (1, -1, 1): array([ .4 ,  1,  .4]),
         (1, 1, -1): array([ .4,  .4,  1]),
         (1, 1, 1): array([ .4, .4,  .4])}
        DH = eye(3)
        v = -array([0.25,0,.1])
        Bc = Bderv(Fb,DH, 3)
        dx,dv =Bc.Bof(v) 
        Bd,dv2 = Bc.Bm(v)
        print(repr(Bd))
        drawmat(Bd,test)
        fig = figure()
        ax = fig.gca(projection='3d')
        tt = linspace(-1,1,6)
        X,Y,Z = meshgrid(tt,tt,tt)
        DX = zeros_like(X)
        DY = zeros_like(Y)
        DZ = zeros_like(Z)
        d = range(len(tt))
        for i in range(len(tt)):
            for j in range(len(tt)):
                for k in range(len(tt)):
                    f = Bc.f(sign([X[i,j,k], Y[i,j,k], Z[i,j,k]]))
                    DX[i,j,k] = f[0]
                    DY[i,j,k] = f[1]
                    DZ[i,j,k] = f[2]
        ax.quiver(X,Y,Z,DX,DY,DZ,alpha=0.4)
        dx,dy = meshgrid(tt,tt)
        dz = 0*dx
        ax.plot_surface(dx,dy,dz,alpha=0.2)
        ax.plot_surface(dx,dz,dy,alpha=0.2)
        ax.plot_surface(dz,dy,dx,alpha=0.2)
        ax.set_xlim3d(-1,1)
        ax.set_ylim3d(-1,1)
        ax.set_zlim3d(-1,1)
        show()


