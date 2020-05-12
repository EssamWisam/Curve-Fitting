import warnings

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import numpy as np  # temporary
from numpy import *
import sympy
from sympy.abc import x, y
from sympy import symbols
from sympy.plotting import plot
from mpl_toolkits.mplot3d import Axes3D
from sympy.plotting import plot3d


def Plot_3D_RHS(xdata, ydata, zdata, RHS):
    x, y = symbols('x y')
    rhs = sympy.sympify(RHS)
    RHS = sympy.lambdify([x, y], rhs)
    plot3d(rhs, (x, -5 + min(xdata), 5 + max(xdata)), (y, -5 + min(ydata), 5 + max(ydata)))
    # 3D graphing is pretty straight forward using Sympy, in contrast to how it's only done through parametric
    # equations in MPL, however it does not do scatter plots...
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(xdata, ydata, zdata, c = 'b', marker='o')
    # ax.set_xlabel('X-axis')
    # ax.set_ylabel('Y-axis')
    # ax.set_zlabel('Z-axis')
    # plt.show()


def Plot_2D_RHS(xdata, ydata, RHS, LHS):
    x, y = symbols('x y')
    RHS = sympy.sympify(RHS)
    RHS = sympy.lambdify([x], RHS)
    LHS = sympy.sympify(LHS)
    LHS = sympy.lambdify([x, y], LHS)
    Ydata = []
    for i in xdata:  # the scatter plot will involve (f(y),x) since we're plotting the linearized RHS
        Ydata.append(LHS(xdata, ydata))
    plt.figure(figsize=(6, 4))
    plt.scatter(xdata, Ydata[0], label='Data')
    xdata = np.linspace(min(xdata), max(xdata), 10000)
    plt.plot(xdata, RHS(xdata), label='Fit')
    plt.legend(loc='best')
    plt.show()
    # A=plot(F,show=False, xlim=(min(xdata)-4,max(xdata)+4) , ylim=(min(ydata)-4,max(ydata)+4))
    # A.show()


def Linearized_Regression(xdata, ydata, Function, r):
    SympF = []  # Contains the functions in Sympy form
    F = []  # contains the function in classic form
    x, y = symbols('x, y')
    for i in range(0, len(Function)):  # converting the string functions into a suitable form
        SympF.append(sympy.sympify(Function[i]))  # The first function is that on the RHS (y)
        F.append(sympy.lambdify([x, y], SympF[i]))
    FL = []
    for i in range(0, len(F)):  # this for loop goes through each function
        Z = []  # temporary list
        for j in range(0, len(xdata)):  # this one makes a new list by plugging each xdata, ydata for each function
            Z.append(F[i](xdata[j], ydata[j]))
        FL.append(
            Z)  # FL contains a sublist     for each function, this sublist is the result from plugging each xdata and ydata into the function
        n = len(F)
    a = np.empty((n - 1, n - 1))
    b = np.empty((n - 1, 1))
    for i in range(1, n):
        for j in range(1, n):
            a[i - 1][j - 1] = np.sum(np.multiply(FL[i], FL[j]))
        b[i - 1][0] = np.sum(np.multiply(FL[0], FL[i]))
    Sol = np.round(np.transpose(np.linalg.solve(a, b)),
                   r)  # receiving the list, making it horizontal then rounding each element
    Solution = []
    for sublist in Sol:  # Flattening the list ( from [[ ]] to [ ])
        for item in sublist:
            Solution.append(item)
    RegressionError = []
    for i in range(0, len(ydata)):
        Val = FL[0][i]  # the first sublist inside FL corresponds to the outputs from the function in the LHS
        for j in range(1, n):
            Val -= (FL[j][i] * Solution[
                j - 1])  # subtract each of the entries in the 1st sublist from each of the functions*constant in the RHS.
        RegressionError.append(Val ** 2)  # square at each time.
    Sr = round(np.sum(RegressionError), r)  # Sr = Sum((Yi-Y(regression))^2) = Sum((Yi-Const1*X1i-Const2*X2i-......)^2)
    StringSol = [str(c) for c in Solution]  # converting each constant to a string
    LHS = Function.pop(0)  # removing the LHS
    RHS = [' * '.join(x) for x in zip(StringSol, Function)]  # multiplying the constants and the functions element wise
    RHS = (" + ".join(str(x) for x in RHS))

    return LHS, RHS, StringSol, Sr  # LHS and RHS are just what's actually needed


def Surface_Fit_Beta(xdata, ydata, zdata, Function, r):
    SympF = []  # Contains the functions in Sympy form
    F = []  # contains the function in classic form
    x, y, z = symbols('x, y, z')
    for i in range(0, len(Function)):  # converting the string functions into a suitable form
        SympF.append(sympy.sympify(Function[i]))  # The first function is that on the RHS (y)
        F.append(sympy.lambdify([x, y, z], SympF[i]))

    FL = []
    for i in range(0, len(F)):  # this for loop goes through each function
        Z = []  # temporary list
        for j in range(0, len(xdata)):  # this one makes a new list by plugging each xdata, ydata for each function
            Z.append(F[i](xdata[j], ydata[j], zdata[j]))
        FL.append(
            Z)  # FL contains a sublist for each function, this sublist is the result from plugging each xdata,ydata,zdata into the function
        n = len(F)
    a = np.empty((n - 1, n - 1))
    b = np.empty((n - 1, 1))
    for i in range(1, n):
        for j in range(1, n):
            a[i - 1][j - 1] = np.sum(np.multiply(FL[i], FL[j]))
        b[i - 1][0] = np.sum(np.multiply(FL[0], FL[i]))
    Sol = np.round(np.transpose(np.linalg.solve(a, b)),
                   r)  # receiving the list, making it horizontal then rounding each element
    Solution = []
    for sublist in Sol:  # Flattening the list ( from [[ ]] to [ ])
        for item in sublist:
            Solution.append(item)
    StringSol = [str(c) for c in Solution]  # converting each constant to a string
    LHS = Function.pop(0)  # removing the LHS
    rhs = [' * '.join(x) for x in zip(StringSol, Function)]  # multiplying the constants and the functions element wise
    RHS = (" + ".join(str(x) for x in rhs))

    return LHS, RHS, StringSol  # LHS and RHS are just what's actually needed


def Nonlinear_Regression(xdata, ydata, NonlinearFunction):  # takes x,y lists and a nonlinear function string
    F = lambda x, a, b, c: eval(NonlinearFunction)
    Constants, Covariance = curve_fit(F, xdata, ydata)  # we don't need to show the covariance matrix
    return Constants[0], Constants[1], Constants[
        2]  # this contains a,b,c if the former function doesn't have 'c' then it takes it as one (the initial guess)


def Nonlinear_Plot(xdata, ydata, NonlinearFunction, a, b, c):  # Unstable, keeps giving an error here for some reason.
    F = lambda x, a, b, c: eval(NonlinearFunction)
    plt.figure(figsize=(6, 4))
    plt.scatter(xdata, ydata, label='Data')
    xdata = np.linspace(min(xdata), max(xdata), 10000)
    plt.plot(xdata, F(xdata, a, b, c), label='Agmad Fit')
    plt.legend(loc='best')
    plt.show()


def Input_2D():
    xdata = []
    ydata = []
    n = int(input("Type in the no. of points : "))
    print('Insert each point as space seperated x & y')
    for i in range(0, n):
        x, y = map(float, input().split())  # splits a given input 'a b' , maps 'a' in x and 'b' in y
        xdata.append(float(x))  # filling the x,y lists
        ydata.append(float(y))
    return np.array(xdata), np.array(ydata)


def Input_3D():
    xdata = []
    ydata = []
    zdata = []
    n = int(input("Type in the no. of points : "))
    print('Insert each point as space seperated x & y & z')
    for i in range(0, n):
        x, y, z = map(float, input().split())  # splits a given input 'a b' , maps 'a' in x and 'b' in y
        xdata.append(float(x))  # filling the x,y lists
        ydata.append(float(y))
        zdata.append(float(z))
    return np.array(xdata), np.array(ydata), np.array(zdata)


def TrueError(ydata, r):
    TrueError = []
    ydata_Avg = (np.sum(ydata)) / len(ydata)
    for y in ydata:
        TrueError.append((y - ydata_Avg) ** 2)
    St = round(np.sum(TrueError), r)
    return St


def PopularForms(xdata, ydata, r):
    formulas = {"linear": ["y", "1", "x"],
                "quadratic": ["y", "1","x","x**2"],
                "cubic": ["y", "1", "x", "x**2", "x**3"],
                "exponential": ["ln(y)", "1", "x"],
                "logarithmic": ["y", "1", "ln(x)"],
                "reciprocal": ["1/y", "1", "x"],
                "power": ["log(y)", "1", "log(x)"]}
    forms = ["linear", "quadratic", "cubic", "exponential", "logarithmic", "reciprocal", "power"]
    LHS = [] #holds the lhs for all possible forms
    RHS = [] #holds the lhs for all possible forms
    Str_Sol = [] #holds the constants for all possible forms
    reg_errors = [] #holds the constants for all possible forms
    str_equation = "" #holds a string representing the best-fitted equation chosen from the upove formulas
    for n in formulas:
        lhs, rhs, str_sol, sr = Linearized_Regression(xdata, ydata, formulas.get(n), r)
        LHS.append(lhs)
        RHS.append(rhs)
        Str_Sol.append(str_sol)
        reg_errors.append(sr)
    ind = reg_errors.index(min(reg_errors))
    if (ind == 3):
        #y=ae^(bx), Str_Sol[3] = ["ln(a)","b"]
        str_equation = "y = " + str(round(exp(float(Str_Sol[ind][0])),r)) + " * e^(" + Str_Sol[ind][1] + " * x)"
    elif (ind == 5):
        #y=1/(a+bx), RHS[5] = "a+bx"
        str_equation = "y = 1/("+RHS[ind]+")"
    elif (ind == 6):
        # y=a*x^(b),  Str_Sol[6] = ["ln(a)","b"]
        str_equation = "y = " + str(round(exp(float(Str_Sol[ind][0])),r)) + " * x^(" + Str_Sol[ind][1] + ")"
    else:
        #The linearized form is the original form itself
        str_equation = LHS[ind] + " = " + RHS[ind]
    return str_equation,forms[ind],reg_errors[ind]

def main():
    repeat = True
    while (repeat):
            # Picking a choice:
            Choice = input(
                "Type \"N\" to curve fit using nonlinear regression, \"L\" for linear regression, \"P\" for popular linear regression forms and \"S\" for surface fitting through linear regression : ")
            Choice=Choice.upper()
            r = int(input("Round the results to how many decimals ? :\n "))
            if (Choice == 'N'):
                xdata, ydata = Input_2D()
                NonlinearFunction = input("Type in the Nonlinear Function : \n")
                A, B, C = Nonlinear_Regression(xdata, ydata, NonlinearFunction)
                if (C == 1):  # The initial guess is as it is; the given function doesn't involve in c
                    print('\n', '[a b] for the best fit= ', '[' + str(round(A, r)) + '   ' + str(round(B, r)) + ']',
                          '\n')
                else:
                    print('\n', '[a b c] for the best fit= ',
                          '[' + str(round(A, r)) + '   ' + str(round(B, r)) + '   ' + str(round(C, r)) + ']', '\n')
                Nonlinear_Plot(xdata, ydata, NonlinearFunction, A, B, C)
            elif (Choice == 'L'):
                xdata, ydata = Input_2D()
                n = int(input(
                    "Type in the no. of functions in your linearized form Ex. Sin(x/y)=a*(x^2)+b*tan(x)+c/x involves 4 functions : "))
                Function = []
                for i in range(0, n):
                    Function.append(input("Insert your functions Y, X1, X2... : \n"))
                LHS, RHS, Constants, Sr = Linearized_Regression(xdata, ydata, Function, r)
                print(LHS, '=', RHS);
                print("Regression Error(Sr)= ", Sr);
                St = TrueError(ydata, r);
                print("True Error(St)= ", St);
                corrolation_coeff = round(sqrt((abs(St - Sr)) / St), r)
                print("Corrolation Coeffecient(r)= ", corrolation_coeff);
                Plot_2D_RHS(xdata, ydata, RHS, LHS)
            elif (Choice == 'S'):
                xdata, ydata, zdata = Input_3D()
                n = int(input(
                    """Type in the no. of functions, Ex. the paraboloid : Z= f(x, y) = A * x^2 +B* x*y + C*y^2 + D*x + E*y + H corresponds to seven functions  z,x^2,x*y,y^2,x,y,1 : \n"""))
                Function = []
                for i in range(0, n):
                    Function.append(input("Insert your functions : \n"))
                LHS, RHS, Constants = Surface_Fit_Beta(xdata, ydata, zdata, Function, r);
                print(LHS, '=', RHS);
                Plot_3D_RHS(xdata, ydata, zdata, RHS)
            elif (Choice == 'P'):
                xdata, ydata = Input_2D()
                str_equation, formula_Family, Sr = PopularForms(xdata, ydata, r)
                print("your function is in the family of " + formula_Family + " functions")
                print(str_equation)
                print("Regression Error(Sr)= ", Sr);
                St = TrueError(ydata, r);
                print("True Error(St)= ", St);
                corrolation_coeff = round(sqrt((abs(St - Sr)) / St), r)
                print("Corrolation Coeffecient(r)= ", corrolation_coeff);

            else:
                print("You entered an invalid symbol")
            again = input("Do you want another try? type yes or no")
            if (again.lower() == "yes"):
                repeat = True
            else:
                repeat = False
main()

