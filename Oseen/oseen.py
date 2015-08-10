import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from  matplotlib.pyplot import contour
from mpl_toolkits.mplot3d.axes3d import Axes3D
import scipy.sparse.linalg
from math import sqrt,log,fabs
from math import log
from numpy import meshgrid
def dele(ro,col,dat,*args):
    for arg in args:
        ro[13*arg:13*(arg + 1)] = 0
        col[13*arg:13*(arg + 1)] = 0
        dat[13*arg:13*(arg + 1)] = 0
def assign(ro,col,dat,val,*args):
    dele(ro,col,dat,*args)
    for arg in args:
        ro[13*arg] = arg
        col[13*arg] = arg
        dat[13*arg] = val
def on_boundary(i,bsize,n,h):
    return ((fabs(height(i % n,n,h)) == bsize) and (fabs(height(i // n,n,h)  )<= bsize)
            or (fabs(height(i // n,n,h)) == bsize and fabs(height(i % n,n,h)) < bsize))
def height(i,n,h):
    return ((n - 1)/2 - i)*h
def visualise(x,n,h):
    #Visualisization of the streamlines
    ht = lambda i: ((n - 1)/2 - i)*h
    y_axis = np.zeros([n])
    x_axis = np.zeros([n])
    for i in range(n):
        y_axis[i] = ht(i)
        x_axis[i] = i*h
    xx,yy = meshgrid(x_axis,y_axis)
    Z = np.zeros([n,n])
    for i in range(n**2):
        Z[i // n,i % n] = x[i]
    fig = plt.figure(figsize = (10,10))
    ay = fig.add_subplot(1,1,1)
    ay.contour(xx,yy,Z,121)
def mult_assign(col,dat,val,i,ic):
    col[i] = ic
    dat[i] = val
def iteroverRe(n,b,V,U):
    #Iteration in order to take into Reynolds number influence
     for i in range(2, n - 2):
       for j in range(2, n - 2):
           if abs(height(i // n)>st_range and  abs(height(i % n)))>st_range:
               b[i*n + j] = Re*(V[i,j]*(U[i + 1,j] + U[i - 1,j] + U[i,j + 1] + U[i, j - 1]))
               b[i*n + j] = b[i*n + j]  -  Re*(U[i, j]*(V[i + 1,j] + V[i - 1,j] + V[i,j + 1]))
def convert(V,h1,h2):
    #Converting set of velocities from one precision(h1 - gridsize) to another with gridsize h2
    L = (len(V) - 1)*h1
    n = int(L / h2) + 1
    V_new = np.zeros([n])
    for i in range(n):
        V_new[i] = V[int(round(i*h2 / h1))]        
    return V_new
def stokesanal(loc,n,h,Re): # value of the analytical solution in the stokes range for point with location "loc"  
    height = lambda i: ((n - 1)/2 - i)*h    
    a = loc // n 
    b1 = loc % n      
    r = sqrt (height(a)**2 + height(b1)**2)
    if (r !=  0): si = height(a)/r
    else: si = 0 
    if r == 0: res = 0
    else : res =  - ( 1.0 / log(Re)) * (si*(r*log(r) - r / 2.0 + 1 / (2.0 *r)))
    return res
class OseenSolver:
    def __init__(self,Re,L,h):
        self.Re = Re
        self.L = L
        self.h = h 
        self.grid = int(L / h + 1)
        self.row = np.zeros([13*self.grid*self.grid])
        self.data = np.zeros([13*self.grid*self.grid])
        self.column = np.zeros([13*self.grid*self.grid])
        self.b = np.zeros([self.grid*self.grid])
    def assignFDM(self):
        #Assigning 13-point stencil to the matrix
        for i in range(2*self.grid,(self.grid**2 - 2*self.grid)):
            assign(self.row,self.column,self.data,20,i)
            self.row[13*i:13*i + 13] = i
            if i % self.grid != 0:
                 mult_assign(self.column,self.data,2,13*i + 1,i + self.grid - 1)
                 mult_assign(self.column,self.data,2,13*i + 2,i - self.grid - 1)
                 mult_assign(self.column,self.data,-8,13*i + 3,i - 1)
            if (i + 1) % self.grid != 0:
                 mult_assign(self.column,self.data,2,13*i + 4,i - self.grid + 1)
                 mult_assign(self.column,self.data,2,13*i + 5,i + self.grid + 1)
                 mult_assign(self.column,self.data,-8,13*i + 6,i + 1)
            mult_assign(self.column,self.data,1,13*i + 7,i - 2*self.grid)
            mult_assign(self.column,self.data,1,13*i + 8,i + 2*self.grid)
            mult_assign(self.column,self.data,-8,13*i + 9,i + self.grid)
            mult_assign(self.column,self.data,-8,13*i + 10,i - self.grid)
            if (i + 2) % self.grid != 0 and (i + 1) % self.grid != 0:
                mult_assign(self.column,self.data,1,13*i + 11,i + 2)
            if (i - 1) % self.grid != 0  and i % self.grid != 0:
                mult_assign(self.column,self.data,1,13*i + 12,i - 2)
    def setBoundaryInner(self,size,V_x,V_y):
        #Setting Neuman boundary condition on the rectangle with side - 2*size
        ht = lambda i: ((self.grid - 1)/2 - i)*self.h
        n = self.grid
        for j in range(self.grid**2):
            if on_boundary(j,size,n,self.h):
                if (fabs(ht(j % n)) == size):
                    dele(self.row,self.column,self.data,j)
                    self.column[13*j],self.row[13*j],self.row[13*j + 1],self.row[13*j + 2]= j,j,j,j
                    self.data[13*j] = 3 
                    self.data[13*j + 1] = - 4
                    self.data[13*j + 2] = 1
                    self.column[13*j + 1] = j - n
                    self.column[13*j + 2] = j - 2*n
                    self.b[j] =   V_y[j // n ,j % n]
                if (fabs(ht(j // n)) == size) and (fabs(ht(j % n))) != size:
                    dele(self.row,self.column,self.data,j)
                    self.column[13*j],self.row[13*j],self.row[13*j + 1],self.row[13*j + 2]= j,j,j,j
                    self.data[13*j] = 3 
                    self.data[13*j + 1] = - 4
                    self.data[13*j + 2] = 1
                    self.column[13*j + 1] = j + 1
                    self.column[13*j + 2] = j + 2
                    self.b[j] =  2.0*V_x[j // n,j % n]
            if on_boundary(j,size - self.h,n,self.h):
                dele(self.row,self.column,self.data,j)
                self.row[13*j],self.column[13*j],self.row[13*j + 1] = j,j,j
                self.data[13*j + 1] = - 1 
                self.data[13*j ] = 1
                if fabs(ht(j % n)) == size - self.h :
                    if ht(j % n) > 0:
                        self.column[13*j + 1] = j - 2
                        self.b[j] =  - 2*V_x[j // n ,j %  n - 1]
                    else: 
                        self.column[13*j + 1] = j + 2
                        self.b[j] =     2*V_x[j // n ,j % n + 1]
                elif fabs(ht(j % n)) != fabs(ht(j // n)):
                    if (ht(j // n)) > 0:
                        self.column[13*j + 1] = j - 2*n
                        self.b[j]  =  2*V_y[j // n - 1 , j % n]
                    else: 
                        self.column[13*j + 1] = j + 2*n
                        self.b[j] =  - 2*V_y[j // n + 1,j % n]                          
    def setBoundaryOuter(self):
        #Setting Boundary condition for the infinity region (At the distancce L / 2 from the cylinder)
        ht = lambda i: ((self.grid - 1)/2 - i)*self.h
        for j in range(self.grid):
              self.b[j] =  20.0*ht(0)
              self.b[j + self.grid] = 20*ht(1)
              self.b[self.grid**2 - self.grid + j] = 20.0*ht(self.grid - 1)
              self.b[self.grid**2 - 2*self.grid + j] = 20.0*ht(self.grid - 2)  
              self.b[j*self.grid] = 20.0*ht(j)
              self.b[j*self.grid + 1] = 20.0*ht(j)
              self.b[self.grid*(j + 1) - 1] = 20.0*ht(j)
              self.b[self.grid*(j + 1) - 2] = 20.0*ht(j)
              assign(self.row,self.column,self.data,20,j,j + self.grid,self.grid**2 - 2*self.grid + j,self.grid**2 - self.grid + j)
              assign(self.row,self.column,self.data,20,j*self.grid,j*self.grid + 1,self.grid*(j + 1) - 1,self.grid*(j + 1) - 2) 
    def getVelocity(self,bsize):
        V_y = np.zeros([bsize])
        V_x = np.zeros([bsize])
        n = self.grid
        k = 0
        for j in range(self.grid**2):
            if on_boundary(j,bsize*self.h/8,self.grid,self.h):
                V_y[k] = (self.Z[j // n + 1, j % n ] - self.Z[j // n - 1, j % n ])/2
                V_x[k] = (self.Z[j // n, j % n + 1] - self.Z[j // n, j % n  - 1])/2
                k += 1
        return V_x,V_y
    def update(self):
         A = scipy.sparse.coo_matrix((self.data,(self.row,self.column)),shape = ([self.grid**2,self.grid**2]))
         x = scipy.sparse.linalg.spsolve(A,self.b)
         self.Z = np.zeros([self.grid,self.grid])
         for i in range(self.grid**2):
             self.Z[i // self.grid,i % self.grid] = x[i]