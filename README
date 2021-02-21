<center>File description and minimal example of B-derivative code.</center>

<center>**George Council, Shai Revzen, Sam Burden  
Representing and computing the B-derivative of an ECr vector field’s PCr flow**</center>

* * *

### Requirements:

The main algorithms are implemented in Python 3.6.9, though it should be backwards compatible with Python 2.7.5\. The examples are inteded to be run in IPython 3, vers. 5.5.0, though should be executable as standalone python scripts, e.g. "python filename.py". The core alogrithms are contained in the module bderv2.py, which should be imported (and therefore in the python path) for use.  

The most up-to-date version of this code is available at : https://github.com/gcouncil-cmu/bderv.git

* * *

### Files

We will give a high-level overview of each file and its purpose; A complete enuermation of file/class contents can be obtained with the 'pdoc' python packge, which can be installed via ``pip install pdoc''. E.g., an html can then be obtained via ``pdoc --html bderv2.py''

1.  bderv.py  
    Numerical B-differential alogrithm. Contains classes "Bderv" and "BdervExtra". Each class instance minimally has internal states of F and DH, and methods to evalute the B-derivative.
    1.  Bderv -- direct implemetation of Alg. 2 -- __call__ method maps dx^- to dx^+
    2.  BdervExtra -- inherits Bderv; also possesses methods Bd and Bm.
        *   Bd -- for given sequence w experienced by x+dx, computes matrix B_w via sequence of rank-1 updates.
        *   Bm -- for given sequence w experienced by x+dx, computes matrix B_w via simplicial map

2.  test-bd.py  
    Test suite for Bderv. Contains a collection examples. Each is executed by declaring the 'test' variable in line 42 to one of the following options, the running the file.

    *   2d-contract

    *   2d-asym

    *   2d-angled

    *   2d-nonlinear

    *   3d-contract

    *   3d-degen

3.  chair.py  
    Code assocaited to Section 3.2, where a planar chair/table is striking a soft ground. This file contains several sections:
    *   Section 0: python class ``chair'' -- this is a numerical object that builds and simulates the dynamics of the chair hitting the ground.
    *   Section 1: Builds b-derivatives for both the mode-depedent and mode-independent schemes.
    *   Section 2: Evaluates associated B-derivates along linspace for Fig. in Section 3.3 of paper.
    *   Section 3: A speed comparison for computing the varational equation of the flow -- we would expect a finite difference method to produce an matrix quite close to our elements for small time, yet will be much slower as such a method needs to consider at least 2^m points and integrate the nonlinear flow numerically.  
        E.g., the python variables will be in the namespace if you run in ipython, and should all coincide :
        *   C.M12 -- exact matrix computed symbolically (same as chair-sym notebook).
        *   M12 -- matrix from sequence of outer product -- Alg. 2.2
        *   M12p -- matrix computed from simplicial algorithm used in proof; equal, but slower, to Alg 2.2
        *   dfr -- numerical jacobian of \phi(T,x + \delta x) - \phi(t,x); since this accounts for nonlinearities away from \rho, this will be close saltation matrix for small time near \rho, but less close further away.
    *   Section 4: Plotting.
4.  chair-sym.ipynb  
    Python notebook that contains the symbolic computations of Section 3 in a tutorial form.
5.  SOO.py  
    Second-order oscillator example from 'Event-selected vector fields yield piecewise differentiable flows' (10.1137/15M1016588) Section 8.2
6.  integro.py

### Basic Usage

The Bderv class constructor is requires the sampled vector field F as dictionary indexed by tuples (+/- 1, +/- ) so that e.g., F[(-1,1)] = F_(-1,1), and collection of surface normals DH as m x d array, where row mi is the surface normal for plane H_i. Critically, this data must be for the sampled system -- for an example of how to generate the sampled data from the nonlinear vector field, see the text or the ``2d-nonlinear'' example in test-bd.py A minimal example is the following, where Bderv and BdervExtra are both shown:  
```
dim = 2 #systemdimension  
dl = +0.25 #vf parameter  
nu = .75 #vf parameter  
Fb = {(-1,-1): array([nu+dl,nu+dl]), (-1,1) : array([nu+dl,nu-dl]), (1,-1) : array([nu-dl,nu+dl]), (1,1) : array([nu-dl,nu-dl])} #sampled vf definition  
DH = eye(2) # two orthogonal event surfaces  
v = -array([0.25,0]) #test tangent vector  
B = Bderv(Fb,DH, dim)  
dv =B(v)#Alg. 2 method  
Bc = BdervExtra(Fb,DH, dim)  
B1,dv2 = Bc.Bm(v) #B is full matrix, and dv2=dv=dot(B1,v)=B(v)  
```

* * *
Note: rigid2.py is a python2 application.
