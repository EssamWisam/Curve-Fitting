import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit
import numpy as np
from numpy import *
#from math import *
#This should be made into a function "Nonlinear Regression" that receives two lists and a string function and returns a list with the constants.
#def func(x, a, b): 
  #  return ff(x,a,b) 
  #theFunction = input("enter func\n")
#ff= lambda x,a,b: eval(theFunction)

#xdata = []
#ydata = []
#n = int(input("Enter number of elements : ")) 
#print('Enter each point as space seperated x & y')

#for i in range(0, n): 
#    x,y = map(float, input().split())
#    xdata.append(x)
#    ydata.append(y)
#popt, pcov = curve_fit(func, xdata, ydata)
#print('\n', '[a b] for agmad fitting =    ', popt,'\n')
##Graphing the fit
#plt.figure(figsize=(6, 4))
#plt.scatter(xdata, ydata, label='Data')
#plt.plot(xdata, ff(xdata, popt[0], popt[1]), label='Agmad Fit')
#plt.legend(loc='best')
#plt.show() 

#This is an initial version for a function that applies the general regression formula
import sympy# We shouldn't import everything
from sympy.abc import x,y
import numpy as np
from numpy import log
from sympy import symbols
# Entering the functions
from numpy import arange, meshgrid, sqrt




Function=[]
SympF = []
F = []
n = int(input("Please enter the number of terms in the general regression formula (include both RHS and LHS) : "))
for i in range(0, n):
    Function.append(input("Insert your functions : \n"))
    SympF.append(sympy.sympify(Function[i])) #The first function is that on the RHS (y)
    F.append(sympy.lambdify([x, y], SympF[i]))

# Entering Points
xdata = []
ydata = []

m = int(input("Enter number of elements : "))
print("Enter each point as space seperated x & y")
for i in range(0, m):
    x, y = map(float, input().split())
    xdata.append(x)
    ydata.append(y)
r = int(input("Round the results to how many decimals ? : "))

FL = [] 
for i in range(0, n):
    Z = [] #temporary list 
    for j in range(0, m): 
        Z.append(F[i](xdata[j], ydata[j])) 
    FL.append(Z) #FL contains a sublist for each function, this sublist is the result from plugging each xdata and ydata into the function

    #Solving the general matrix for the coefficients
if 3 == n:
    a = np.array([[np.sum(np.multiply(FL[1],FL[1])),np.sum(np.multiply(FL[1],FL[2]))],
                    [np.sum(np.multiply(FL[1],FL[2])),np.sum(np.multiply(FL[2],FL[2]))]])
    b = np.array([np.sum(np.multiply(FL[0],FL[1])),np.sum(np.multiply(FL[0],FL[2]))])
    Sol = np.linalg.solve(a, b)
    print("According to linear regression the Best fit is : "+Function[0]+"="+str(round(Sol[0],r))+Function[1]+'+'+str(round(Sol[1],r))+Function[2])
    #Graph:
    from sympy import symbols
    from sympy.plotting import plot, plot3d, PlotGrid
    x, y = symbols('x, y')
    p1 = plot(Sol[0]*SympF[1]+ Sol[1]*SympF[2], (x, xdata[0], xdata[m-1])) #Plots F(y) vs X
    
if 4 == n:
    a = np.array([[np.sum(np.multiply(FL[1],FL[1])),np.sum(np.multiply(FL[1],FL[2])),np.sum(np.multiply(FL[1],FL[3]))],
                    [np.sum(np.multiply(FL[1],FL[2])),np.sum(np.multiply(FL[2],FL[2])),np.sum(np.multiply(FL[2],FL[3]))],
                    [np.sum(np.multiply(FL[1],FL[3])),np.sum(np.multiply(FL[2],FL[3])),np.sum(np.multiply(FL[3],FL[3]))]])
    b = np.array([np.sum(np.multiply(FL[0],FL[1])),np.sum(np.multiply(FL[0],FL[2])),np.sum(np.multiply(FL[0],FL[3]))])
    Sol = np.linalg.solve(a, b)
    print("According to linear regression the Best fit is : "+Function[0]+"="+str(round(Sol[0],r))+Function[1]+'+'+str(round(Sol[1],r))+Function[2]+'+'+str(round(Sol[2],r))+Function[3]) #The best fit.
    from sympy import symbols
    from sympy.plotting import plot, plot3d, PlotGrid
    x, y = symbols('x, y')
    p1 = plot(Sol[0]*SympF[1]+ Sol[1]*SympF[2]+Sol[2]*SympF[3], (x, xdata[0], xdata[m-1])) #Plots F(y) vs X
