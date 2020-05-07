from sympy import symbols, diff, sin, cos, Function, exp, sympify
from sympy import *
from sympy.abc import x, y, z

f=sympify(input("Please Enter your first function : \n"))
g=sympify(input("Please Enter your second function :\n "))
fx=diff(f,x)
fy=diff(f,y)
gx=diff(g,x)
gy=diff(g,y)

xi = lambdify([x, y, z], x-(gy*f-fy*g)/(fx*gy-fy*gx))
yi = lambdify([x, y, z], y-(-gx*f+fx*g)/(fx*gy-fy*gx))

a=float(input("initial guess_1 \n"))
b=float(input("initial guess_2 \n"))



for x in range(6):
    print(str(a)+" "+str(b))
    a=xi(a,b,0)
    b=yi(a,b,0)


#f=sympify("x^3+y")
#print(f.subs(x,3))


