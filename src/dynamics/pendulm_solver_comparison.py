from __future__ import annotations
import numpy as np

g = 9.81
l = 1.0
c = 0.0

def pendulum_ode(t, y):
    theta, omega = y
    return np.array([omega, -(g / l) * np.sin(theta) - c * omega])

def energy(y):
    theta, omega = y
    return 0.5 * (l ** 2) * (omega ** 2) + g * l * (1 - np.cos(theta))

def euler(y0, h, t_end):
    y = y0.copy()
    t = 0.0
    ys = [y0.copy()]
    while t < t_end:
        y = y + h * pendulum_ode(t, y)
        t = t + h
        ys.append(y.copy())
    return np.array(ys)



y0 = np.array([0.1, 0.0])          
traj = euler(y0, h=0.001, t_end=4.0)   

print("形状:", traj.shape)            
print("初始能量:", energy(traj[0]))
print("末尾能量:", energy(traj[-1]))