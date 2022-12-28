#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 23 10:50:18 2022

El robotista. Control de velocidad de un motor de corriente directa.

@author: malon
"""
import matplotlib.pyplot as plt
import math

# simulation variables
ti = 0.
tf = 2.
ts = 0.001
t  = ti

# variables del motor
K = 145.47 # DC motor gain
tau = 0.087 # System time constant
w = 0. # Initial motor speed.
wr = 1000.0

# Solution vectors
W = [w] # Speed vector
T = [t] # Time vector
U = [0.] # Voltage vector
Wr = [wr / 1000.0] # desired speed vector

# Variables de PID
e = 0. # Error
ie = 0. # integral error
de = 0. # derivative error
prev_e = 0. # previous error

# Target parameters
Ts = 0.2 # stabilizing time
Pos = 0.05 # max overshoot in percentage

# System parameters second order
z = abs(math.log(Pos)) / math.sqrt((math.log(Pos))**2 + math.pi**2)
wn = 4 / (z * Ts)

# Controller gains calculated from desired parameters
kp = (2 * z * wn * tau - 1) / K
ki = (tau * wn **2) / K
kd = 0.

while t < tf:
    # error
    e = wr - w
    
    # error derivative calculation
    de = (e - prev_e) / ts
    prev_e = e
    
    # Integral calculation
    ie += e * ts
    
    # PID calculation
    Vin = (kp * e) + (ki * ie) + (kd * de)
    
    # Motor dynamics
    dw = (Vin * K - w) / tau
    
    # Integrate to obtain speed
    w += dw * ts
    
    # Next time step
    t += ts
    
    # Save vectors to graph
    W.append(w / 1000.) # Convert to RPM x 1000
    U.append(Vin)
    Wr.append(wr / 1000.) # Convert to RPM x 1000
    T.append(t)
    
# Plot results
plt.plot(T, W, linewidth = 2)
plt.title('PD response of motor')
plt.xlabel('Time (s)')
plt.ylabel('Speed ( RPM x 1000)')
plt.grid()
plt.show()


