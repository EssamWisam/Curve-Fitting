import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np
from math import *

def func(x, a, b):
    return ff(x,a,b)

theFunction = input("enter func\n")
ff= lambda x,a,b: eval(theFunction)

xdata = []
ydata = []
n = int(input("Enter number of elements : ")) 
print('Enter each point as space seperated x & y')

for i in range(0, n): 
    x,y = map(float, input().split())
    xdata.append(x)
    ydata.append(y)

popt, pcov = curve_fit(func, xdata, ydata)
print('\n', '[a b] for agmad fitting =    ', popt,'\n')