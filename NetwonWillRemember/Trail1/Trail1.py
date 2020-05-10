import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np #temporary
from numpy import *
import sympy
from sympy.abc import x,y
from sympy import symbols
from sympy.plotting import plot


def Linearized_Regression(xdata, ydata, Function,r):
    SympF = [] #Contains the functions in Sympy form
    F = [] #contains the function in classic form
    x, y = symbols('x, y')
    for i in range(0,len(Function)): #converting the string functions into a suitable form
        SympF.append(sympy.sympify(Function[i])) #The first function is that on the RHS (y)
        F.append(sympy.lambdify([x, y], SympF[i]))

    FL = [] 
    for i in range(0,len(F)): #this for loop goes through each function
        Z = [] #temporary list 
        for j in range(0, len(xdata)):  #this one makes a new list by plugging each xdata, ydata for each function
            Z.append(F[i](xdata[j], ydata[j])) 
        FL.append(Z) #FL contains a sublist for each function, this sublist is the result from plugging each xdata and ydata into the function

    if 3 == len(F):
        a = np.array([[np.sum(np.multiply(FL[1],FL[1])),np.sum(np.multiply(FL[1],FL[2]))],#Setting up the matrices
                        [np.sum(np.multiply(FL[1],FL[2])),np.sum(np.multiply(FL[2],FL[2]))]])
        b = np.array([np.sum(np.multiply(FL[0],FL[1])),np.sum(np.multiply(FL[0],FL[2]))])
        Sol = np.linalg.solve(a, b)# solving the general equation
        Solution="According to linear regression the Best fit is : "+Function[0]+"="+str(round(Sol[0],r))+'('+Function[1]+')'+'+'+str(round(Sol[1],r))+'('+Function[2]+')'
        #Graph
        #x, y = symbols('x, y')
       # p1 = plot(Sol[0]*SympF[1]+ Sol[1]*SympF[2], (x, xdata[0], xdata[len(xdata)-1])) 
        return Sol[0], Sol[1], 0, Solution
    if 4 == len(F): #doing the same for 3x3 matrices
        a = np.array([[np.sum(np.multiply(FL[1],FL[1])),np.sum(np.multiply(FL[1],FL[2])),np.sum(np.multiply(FL[1],FL[3]))],
                        [np.sum(np.multiply(FL[1],FL[2])),np.sum(np.multiply(FL[2],FL[2])),np.sum(np.multiply(FL[2],FL[3]))],
                        [np.sum(np.multiply(FL[1],FL[3])),np.sum(np.multiply(FL[2],FL[3])),np.sum(np.multiply(FL[3],FL[3]))]])
        b = np.array([np.sum(np.multiply(FL[0],FL[1])),np.sum(np.multiply(FL[0],FL[2])),np.sum(np.multiply(FL[0],FL[3]))])
        Sol = np.linalg.solve(a, b)
        Solution="According to linear regression the Best fit is : "+Function[0]+"="+str(round(Sol[0],r))+'('+Function[1]+')'+'+'+str(round(Sol[1],r))+'('+Function[2]+')'+'+'+str(round(Sol[2],r))+'('+Function[3]+')'
        #x, y = symbols('x, y')
        #p1 = plot(Sol[0]*SympF[1]+ Sol[1]*SympF[2]+Sol[2]*SympF[3], (x, xdata[0], xdata[len(xdata)-1])) #Y vs x
        return Sol[0], Sol[1], Sol[2], Solution


def Nonlinear_Regression(xdata,ydata,NonlinearFunction): #takes x,y lists and a nonlinear function string
  F= lambda x,a,b,c: eval(NonlinearFunction)
  Constants, Covariance = curve_fit(NonlinearFunction, xdata, ydata) #we don't need to show the covariance matrix
  return Constants[0],Constants[1],Constants[2] #this contains a,b,c if the former function doesn't have 'c' then it takes it as one (the initial guess)
  
  
def Nonlinear_Plot(xdata,ydata,NonlinearFunction,a,b,c): #Unstable, keeps giving an error here for some reason.
    F= lambda x,a,b,c: eval(NonlinearFunction)
    plt.figure(figsize=(6, 4))
    plt.scatter(xdata, ydata, label='Data')
    plt.plot(xdata, NonlinearFunction(xdata, a, b,c), label='Agmad Fit')
    plt.legend(loc='best')
    plt.show()



def main():

    #Entering data Points
    xdata = []
    ydata = []
    n = int(input("Type in the no. of points : ")) 
    print('Insert each point as space seperated x & y')
    for i in range(0, n):
       x,y = map(float, input().split()) # splits a given input 'a b' , maps 'a' in x and 'b' in y 
       xdata.append(x) #filling the x,y lists
       ydata.append(y)
    r = int(input("Round the results to how many decimals ? :\n "))
    #Picking a choice:
    Choice=input("""Type \"N\" to curve fit using nonlinear regression, \"L\" for linear regression, \"P\" for popular linear regression forms and 
\"S\" for surface fitting through linear regression : """)
    if(Choice=='N'):
          NonlinearFunction = input("Type in the Nonlinear Function : \n")
          A,B,C=Nonlinear_Regression(xdata,ydata,F)
          if (C==1): #The initial guess is as it is; the given function doesn't involve in c
              print('\n', '[a b] for the best fit= ', '['+str(round(A,r)) +'   '+str(round(B,r))+ ']' ,'\n')
          else:
              print('\n', '[a b c] for the best fit= ', '['+str(round(A,r)) +'   '+str(round(B,r))+'   '+str(round(C,r))+ ']' ,'\n')
          Nonlinear_Plot(xdata, ydata, F,A,B,C)
    elif(Choice=='L'):
        n = int(input("Type in the no. of functions in your linearized form Ex. Sin(x/y)=a*(x^2)+b*tan(x)+c/x involves 4 functions : "))
        Function=[]
        for i in range(0, n):
            Function.append(input("Insert your functions Y, X1, X2... : \n"))
        a,b,c,Solution=Linearized_Regression(xdata, ydata, Function,r)
        print(Solution);

main()


 





