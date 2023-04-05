#write code for hill climbing algorithm

import random

def f(x):
    return x**2

def hill_climbing(x0, f, n):
    x = x0
    x_list = [x]
    for i in range(n):
        x_new = x + random.uniform(-1, 1)
        if f(x_new) > f(x):
            x = x_new
            x_list.append(x)
    return x, x_list

x0 = 0
n = 100
x, x_list = hill_climbing(x0, f, n)
print(x)
print(x_list)