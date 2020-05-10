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
        n=len(F)
    a = np.empty((n-1,n-1))
    b = np.empty((n-1,1))
    for i in range(1,n):
        for j in range(1,n):
            a[i-1][j-1] = np.sum(np.multiply(FL[i],FL[j]))
        b[i-1][0]= np.sum(np.multiply(FL[0],FL[i]))
    Sol=np.round(np.transpose(np.linalg.solve(a, b)),r) #receiving the list, making it horizontal then rounding each element
    Solution = []
    for sublist in Sol: #Flattening the list ( from [[ ]] to [ ])
        for item in sublist:
            Solution.append(item)
    RegressionError = []
    for i in range(0,len(ydata)):
        Val = FL[0][i]
        for j in range(1,n):
            Val -= (FL[j][i]*Solution[j-1])
        RegressionError.append(Val**2)
    Sr = round(np.sum(RegressionError),r) #Sr = Sum((Yi-Y(regression))^2) = Sum((Yi-Const1*X1i-Const2*X2i-......)^2)
    StringSol = [str(c) for c in Solution ] #converting each constant to a string
    LHS=Function.pop(0) #removing the LHS
    rhs=[' * '.join(x) for x in zip(StringSol, Function)]#multiplying the constants and the functions element wise
    RHS=(" + ".join(str(x) for x in rhs))

    return LHS,RHS,StringSol,Sr #LHS and RHS are just what's actually needed


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
    TrueError=[]
    ydata_Avg = (np.sum(ydata))/len(ydata)
    for y in ydata:
        TrueError.append((y-ydata_Avg)**2)
    St = round(np.sum(TrueError),r)
    #Picking a choice:
    Choice=input("""Type \"N\" to curve fit using nonlinear regression, \"L\" for linear regression, \"P\" for popular linear regression forms and 
\"S\" for surface fitting through linear regression : """)
    if(Choice=='N' or Choice=='n'):
          NonlinearFunction = input("Type in the Nonlinear Function : \n")
          A,B,C=Nonlinear_Regression(xdata,ydata,F)
          if (C==1): #The initial guess is as it is; the given function doesn't involve in c
              print('\n', '[a b] for the best fit= ', '['+str(round(A,r)) +'   '+str(round(B,r))+ ']' ,'\n')
          else:
              print('\n', '[a b c] for the best fit= ', '['+str(round(A,r)) +'   '+str(round(B,r))+'   '+str(round(C,r))+ ']' ,'\n')
          Nonlinear_Plot(xdata, ydata, F,A,B,C)
    elif(Choice=='L' or Choice=='l'):
        n = int(input("Type in the no. of functions in your linearized form Ex. Sin(x/y)=a*(x^2)+b*tan(x)+c/x involves 4 functions : "))
        Function=[]
        for i in range(0, n):
            Function.append(input("Insert your functions Y, X1, X2... : \n"))
        LHS,RHS,Constants,Sr=Linearized_Regression(xdata, ydata, Function,r)
        print(LHS,'=',RHS);
        print("Regression Error(Sr)= ",Sr);
        print("True Error(St)= ", St);
        corrolation_coff=round(sqrt((St-Sr)/St),r)
        print("Corrolation Coffecient(r)= ", corrolation_coff);


main()








