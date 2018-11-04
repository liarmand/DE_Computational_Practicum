import numpy as np
from matplotlib import pyplot as plt
import math

'''
The function from my variant №7
x: value of variable x
y: value of variable y
'''
def funct(x, y):
    return -x + (y * (2 * x + 1)) / x


'''
Euler's Method
Variables x0, y0, xf (in moodle's document X) -- from the document with requirements
n: number of steps
Return: arrays of x and y variables
'''
def euler(x0, y0, xf, n):
    deltax = (xf - x0) / (n - 1)
    x = np.linspace(x0, xf, n)
    y = np.zeros([n])
    y[0] = y0
    for i in range(1, n):
        y[i] = deltax * funct(x[i - 1], y[i - 1]) + y[i - 1]
    return x, y


'''
Improved Euler's Method
Variables x0, y0, xf (in moodle's document X), n -- from the document with requirements
n: number of steps
Return: arrays of x and y variables
'''
def imp_euler(x0, y0, xf, n):
    deltax = (xf - x0) / (n - 1)
    x = np.linspace(x0, xf, n)
    y = np.zeros([n])
    y[0] = y0
    for i in range(1, n):
        m1 = funct(x[i - 1], y[i - 1])
        m2 = funct(x[i], y[i - 1] + deltax * m1)
        y[i] = y[i - 1] + (deltax / 2) * (m1 + m2)
    return x, y


'''
Runge-Kutta Method
Variables x0, y0, xf (in moodle's document X) -- from the document with requirements
n: number of steps
Return: arrays of x and y variables
'''
def runge_kutta(x0, y0, xf, n):
    deltax = (xf - x0) / (n - 1)
    x = np.linspace(x0, xf, n)
    y = np.zeros([n])
    y[0] = y0
    for i in range(1, n):
        del1 = deltax * funct(x[i - 1], y[i - 1])
        del2 = deltax * funct(x[i - 1] + 0.5 * deltax, y[i - 1] + 0.5 * del1)
        del3 = deltax * funct(x[i - 1] + 0.5 * deltax, y[i - 1] + 0.5 * del2)
        del4 = deltax * funct(x[i - 1] + deltax, y[i - 1] + del3)
        y[i] = y[i - 1] + 1 / 6 * (del1 + 2 * del2 + 2 * del3 + del4)

    return x, y


'''
Method to find approximation errors in special method
Variables x0, y0, xf (in moodle's document X) -- from the document with requirements
nmin -- minimal number of steps
nmax -- maximal number of steps
Return: array of y variables
'''
def errors(x0, y0, xf, nmin, nmax, f):
    y = []
    for i in range(nmin, nmax + 1):
        x1, y1 = exact(x0, y0, xf, i)
        x2, y2 = f(x0, y0, xf, i)
        y.append(abs(y1[-1] - y2[-1]))
    return y


'''
Method represents exact solution
Variables x0, y0, xf (in moodle's document X) -- from the document with requirements
n: number of steps
Return: arrays of x and y variables
'''
def exact(x0, y0, xf, n):
    x = np.linspace(x0, xf, n)
    y = np.zeros([n])
    c = (y0 / x0 - 0.5) / math.exp(2 * x0)
    y[0] = y0
    for i in range(1, n):
        y[i] = x[i] * (c * math.exp(2 * x[i]) + 0.5)
    return x, y


'''
Method to draw the plot of the function
x: array of x variables
y: array of y variables
c: color of the line
l: label or name of the method
'''
def draw(x, y, c, l):
    plt.plot(x, y, c, label=l)


'''
Assigning values from my variant №7 to x0, y0, xf(X)
Assigning values for minimal and maximal number of steps
'''

x0 = float(input('x0 = '))
y0 = float(input('y0 = '))
xf = float(input('xf = '))
n = int(input('№ of steps = '))
nmin = int(input('min № of steps = '))
nmax = int(input('max № of steps = '))


'''
Finding x and y variables values and drawing the plot for Euler's method
'''
x, y = euler(x0, y0, xf, n)
draw(x, y, "b-", "Euler's method")


'''
Finding x and y variables values and drawing the plot for Exact Solution
'''
x, y = exact(x0, y0, xf, n)
draw(x, y, "r-", "Exact solution")


'''
Finding x and y variables values and drawing the plot for Improved Euler's method"
'''
x, y = imp_euler(x0, y0, xf, n)
draw(x, y, 'g-', "Improved Euler's method")


'''
Finding x and y variables values and drawing the plot for Runge-Kutta method
'''
x, y = runge_kutta(x0, y0, xf, n)
draw(x, y, 'y-', "Runge-Kutta method")


'''
Showing plots
'''
plt.xlabel("Value of x")
plt.ylabel("Value of y")
plt.title("Approximate Solution with Forward Euler’s Method")
plt.legend(loc = 'upper left')
plt.show()


'''
Finding x and y variables values and drawing the plot of approximation errors for Euler's method
'''
x = range(nmin, nmax + 1)
y = errors(x0, y0, xf, nmin, nmax, euler)
draw(x, y, 'r-', "Euler's method")


'''
Finding x and y variables values and drawing the plot of approximation errors for Improved Euler's method
'''
y = errors(x0, y0, xf, nmin, nmax, imp_euler)
draw(x, y, 'g-', "Improved Euler's method")


'''
Finding x and y variables values and drawing the plot of approximation errors for Runge-Kutta method
'''
y = errors(x0, y0, xf, nmin, nmax, runge_kutta)
draw(x, y, 'b-', "Runge-Kutta method")


'''
Showing plots
'''
plt.xlabel("Value of x")
plt.ylabel("Value of y")
plt.title("Approximate Solution with Forward Euler’s Method")
plt.legend(loc='upper left')
plt.show()
