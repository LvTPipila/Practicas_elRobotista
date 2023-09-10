
"""
Created on Sat Dec 17 13:55:01 2022

In this file, we cover the blog on position control of dc motor from elrobotista.

"""

import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint

class DC_Motor(object):

    def __init__(self, R=1., L=1., J=1., kc=1., kt=1., kf=1., x0=None):
        assert(R > 0. and L > 0. and J > 0. and kc > 0. and kt > 0. and kf > 0.)
        self.params = {
            'R': R,     # Armature resistance
            'L': L,     # Armature inductance
            'J': J,     # Rotor's inertia
            'kc': kc,   # Back emf constant
            'kt': kt,   # Torque constant
            'kf': kf,   # Friction coefficient
        }
        print('Motor parameters-> {}'.format(self.params))
        self.vin = 0.

        """ Define the equations in state space. """
        self.A = np.array([[-R / L, -kc / L, 0.],
                           [kt / J, -kf / J, 0.],
                           [0.,      1.,     0.]])

        self.B = np.array([[1. / L],
                           [0.],
                           [0.]])

    def dynamics(self, x, t, *args):
        try:
            self.A.reshape(3, 3)
            self.B.reshape(3, 1)
            dx = self.A.dot(x.reshape(3, 1)) + self.vin * self.B  
        except Exception:
            raise ValueError('System dynamics dimension mismatch at t {}'.format(t))
        return dx.flatten()


def compute_gains(Ts, Pos, R = 0., L=0., J=0., kc=0., kt=0., kf=0.):
    assert(Ts > 0. and Pos > 0. and R > 0. and L > 0. and J > 0. and kc > 0. and kt > 0. and kf > 0.)
    motor_k = kt / (R * kf + kt * kc)
    motor_tau = R * J / (R * kf + kt * kc)
    print('First order approximation parameters -> K: {}, Tau: {}'.format(motor_k, motor_tau))
    z = abs(math.log(Pos)) / math.sqrt((math.log(Pos)) ** 2 + math.pi ** 2)
    wn = 4. / (z * Ts)
    if -z * wn >= 0.:
        print('System is unstable!!!')
    print('System response -> Natural freq: {}, Damping coefficient: {}'.format(wn, z))
    print('Design parameters-> Max Overshoot: {}%, Settling time: {}s'.format(Pos * 100, Ts))
    k1 = (2 * z * wn * motor_tau - 1.) / motor_k # Zero order coefficient.
    k2 = (motor_tau * wn ** 2) / motor_k  # first order coefficient.
    print('Computed controller gains -> k1: {}, k2: {}'.format(k1, k2))
    return k1, k2


def simulation(x, t, *params):
    motor, kp, kd, setpoint = params
    motor.vin = kp * (setpoint - x[2]) + kd * (-x[1])
    return motor.dynamics(x, t)


if __name__ == '__main__':
    """ Setting up conditions for controller gain computations. """
    p_os, t_s = 0.05, 0.1   # Maximum overshoot and settlign time
    motor = DC_Motor(R = 0.83, J = 2.37e-3, L = 2.31e-3, kc = 0.128, kt = 0.128, kf = 1.697e-3)
    """ Compute the gains of the controller. """
    kd, kp = compute_gains(t_s, p_os, **motor.params)
    
    """ Setting up conditions for simulation. """
    tf, ts = 0.3, 1.0e-3    # Total simulation time and sample time.
    t = np.arange(0, tf + ts, ts)   # Vector time for simulation.
    x0 = [0.0, 0.0, 0.0]    # Initial conditions of current, speed, and position.
    setpoint = 7.0  # Desired position.

    """ Simulate system and plot results. """
    y = odeint(simulation, x0, t, args=(motor, kp, kd, setpoint))
    for i, name in enumerate(['$\i\,[\mathrm{amps}]$', '$\omega\,[\mathrm{rad/s}]$', 
                             '$theta\, [\mathrm{rad}$']):
        ax = plt.subplot(3, 1, i + 1)
        ax.plot(t, y[:, i], linewidth=3.0)
        ax.set_ylabel(name)
        ax.set_xlim([t[0], t[-1]])
    plt.xlabel('$Time\,\mathrm{[s]}$')
    plt.show()

