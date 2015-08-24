import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from  matplotlib.pyplot import contour
from mpl_toolkits.mplot3d.axes3d import Axes3D
import scipy.sparse.linalg
from math import sqrt,log,fabs
from math import log
from numpy import meshgrid
from scipy import weave 
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
def on_boundary(i,bwize,n,h):
    if (((abs(height(i % n,n,h)) == bwize) and (abs(height(i // n,n,h))<= bwize))
            or (abs(height(i // n,n,h)) == bwize and abs(height(i % n,n,h)) < bwize)):
def height(i,n,h):
    return ((n - 1.)/2. - i)*h
def visualise(Z,n,h):
    #Visualisization of the streamlines
    ht = lambda i: ((n - 1)/2 - i)*h
    y_axis = np.zeros([n])
    x_axis = np.zeros([n])
    for i in range(n):
        y_axis[i] = ht(i)
        x_axis[i] = i*h
    xx,yy = meshgrid(x_axis,y_axis)
    fig = plt.figure(figsize = (10,10))
    ay = fig.add_subplot(1,1,1)
    circ = plt.Circle(((n-1)*h*0.5, 0.0), radius=2.0,color = 'k')
    ay.add_patch(circ)
    ay.contour(xx,yy,Z,51)
def mult_assign(col,dat,val,i,ic):
    col[i] = ic
    dat[i] = val
def iteroverRe(n,b,V,U,h):
    #Iteration in order to take into Reynolds number influence
     for i in range(2, n - 2):
       for j in range(2, n - 2):
           if abs(height(i // n)>st_range and  abs(height(i % n)))>st_range:
               b[i*n + j] = Re*h**2(V[i,j]*(U[i + 1,j] + U[i - 1,j] + U[i,j + 1] + U[i, j - 1]))
               b[i*n + j] = b[i*n + j]  -  Re*(U[i, j]*(V[i + 1,j] + V[i - 1,j] + V[i,j + 1]))*h**2 
def analstokes(a,b): # value of the analytical solution in the stokes range for point with location "loc"  
    Re = 0.05    
    r = sqrt(a**2 +b**2)
    if (r !=  0): si =a / r
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
    def setBoundaryInner(self,ssize,V_y,V_x,ran,h_in):
        #Setting Neuman boundary condition on the rectangle with side - 2*size
        ## (round) - is used as a substition for the most basic interpolation
        n = self.grid        
        ht = lambda (i): ((self.grid - 1)/2 - i)*self.h
        convy = lambda (y): int(round((ht(y) +  ran / 2.)/ h_in))    
        convx = lambda (x): int(round((ran / 2. - ht(x))/ h_in))
        y_st = int((self.L / 2 - ssize)/ self.h)
        y_end = int((self.L / 2 + ssize) / self.h) + 1
        x_st = int((self.L / 2 - ssize) / self.h)
        x_end = int((self.L /2 + ssize) / self.h) + 1
        for ii in range(y_st,y_end): # inclusion of the last point
            for jj in range(x_st,x_end):
              j = ii*n + jj
              if on_boundary(j,ssize,n,self.h):
                  if (fabs(ht(j % n)) == ssize) and fabs(ht(j % n)) == 35:
                      dele(self.row,self.column,self.data,j)
                      self.column[13*j],self.row[13*j],self.row[13*j + 1],self.row[13*j + 2]= j,j,j,j
                      self.data[13*j] = 3 
                      self.data[13*j + 1] = - 4
                      self.data[13*j + 2] = 1
                      self.column[13*j + 1] = j - n
                      self.column[13*j + 2] = j - 2*n
                      self.b[j] =   - 2.0*V_x[convy(ii),convx(jj)]*self.h
                  if (fabs(ht(j // n)) == ssize) and (fabs(ht(j % n))) != ssize:
                      dele(self.row,self.column,self.data,j)
                      self.column[13*j],self.row[13*j],self.row[13*j + 1],self.row[13*j + 2]= j,j,j,j
                      self.data[13*j] = 3 
                      self.data[13*j + 1] = - 4
                      self.data[13*j + 2] = 1
                      self.column[13*j + 1] = j + 1
                      self.column[13*j + 2] = j + 2
                      self.b[j] =   2.0*V_y[convy(ii),convx(jj)]*self.h #dont change!!!
              if on_boundary(j,ssize - self.h,n,self.h):
                      if fabs(ht(j % n)) == ssize - self.h:
                          dele(self.row,self.column,self.data,j)
                          self.row[13*j],self.column[13*j],self.row[13*j + 1] = j,j,j
                          self.data[13*j + 1] = - 1 
                          self.data[13*j ] = 1
                          if ht(j % n) > 0:
                              self.column[13*j + 1] = j - 2
                              self.b[j] =    - 2.*V_y[convy(ii), convx(jj - 1)]*self.h 
                          else: 
                              self.column[13*j + 1] = j + 2
                              self.b[j] =    2.*V_y[convy(ii),convx(jj + 1)]*self.h
                      elif fabs(ht(j % n)) != fabs(ht(j // n)):
                          dele(self.row,self.column,self.data,j)
                          self.row[13*j],self.column[13*j],self.row[13*j + 1] = j,j,j
                          self.data[13*j + 1] = - 1 
                          self.data[13*j ] = 1
                          if (ht(j // n)) > 0:
                              self.column[13*j + 1] = j - 2*n
                              self.b[j]  =  -2.*V_x[convy(ii - 1),convx(jj)]*self.h
                          else: 
                              self.column[13*j + 1] = j + 2*n
                              self.b[j] =   2.*V_x[convy(ii + 1),convx(jj)]*self.h
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
    def getVelocity(self,ii,jj,ran):
        ht = lambda i: ((self.grid - 1)/2 - i)*self.h
        #ii_old = ii
        #jj_old = jj
        #ii_st = (ii  - ran  / 2)
        #jj_st = (jj - ran / 2)
        kor =  self.h*round((ii - ran / 2)/self.h)
        kor2 = self.h*round((jj - ran / 2)/self.h)
        ii = int(round((self.L / 2. - kor)/ self.h))
        jj = int(round((self.L / 2. + kor2)/ self.h))
        n = self.grid   
        #V_x = (analstokes(ii_st + 0.04,jj_st) - analstokes(ii_st - 0.04,jj_st))/ 2
        #V_y = -(analstokes(ii_st,jj_st + 0.04) - analstokes(ii_st,jj_st - 0.04)) / 2
        V_x1 =  -(self.Z[ii  + 1, jj] - self.Z[ii - 1,jj])/(2*self.grid) 
        V_y1 = -(self.Z[ii,jj + 1] - self.Z[ii,jj - 1])/(2*self.grid)       
        return V_x1/(0.06*25),V_y1/(0.06*25)
    def update(self):
         A = scipy.sparse.coo_matrix((self.data,(self.row,self.column)),shape = ([self.grid**2,self.grid**2]))
         x = scipy.sparse.linalg.spsolve(A,self.b)
         self.Z = np.zeros([self.grid,self.grid])
         ht = lambda (i): ((self.grid - 1)/2 - i)*self.h
         for i in range(self.grid**2):
             self.Z[i // self.grid,i % self.grid] = x[i]
