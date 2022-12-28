

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from motors import DC_Motor
from controller import PID_controller, compute_gains_current_controller

def simulate(dynamics, x0, t0, tf, ts = 10e-3, simulation=None, args=()):
    """
    Runs the simulation for the specified amount of time and given initial conditions.
    This function also saves the data returned by the simulation function specified in the object
    contructor. The additional data to be saved must be returned as a set of tuples.
    The data will be stored in a numpy array in the order that the simulation function given by
    the user returns it. This is useful to store simulation data such as internal signals as 
    controller error, controller signals, time-varying parameters, etc.
    """
    assert(dynamics)
    assert(ts > 0.)
    assert(tf > t0 + ts)
    assert(x0)

    datalog = None
    unpacked_data = None
    ldata = None
    time = np.arange(t0, tf + ts, ts)
    x = np.array(x0)
    solution = np.array(x0)
    for _t in time:
        if simulation is not None:
            ldata = simulation(x, _t, args = args)
        if ldata is not None:
            # If the simulation returned data, unpack the tuple and store it in a temp
            # variable to be processed.
            unpacked_data = _unpack_tuple(ldata)
            # If the simulatin returned data and it has been unpacked, store in the 
            # datalog array.
            datalog = _logdata(datalog, unpacked_data)
        if _t != time[-1]:
            # Integrator runs one step ahead of siulation.
            # Notice: _t + ts is one timestep ahead.
            x = odeint(dynamics, x, [_t, _t + ts])[1]
            solution = np.vstack((solution, x))
    return (time, solution, datalog) if datalog is not None else (time, solution)


def _unpack_tuple(tdata):
    tmp = np.array([])
    for data in tdata:
        tmp = np.hstack((tmp, data))
    return tmp


def _logdata(datalog, data):
    if datalog is None:
        datalog = data
    else:
        datalog = np.vstack((datalog, data))
    return datalog


def plot_results(t, y):
    for i, name in enumerate(['$\i\,[\mathrm{amps}]$', '$\omega\,[\mathrm{rad/s}]$', \
                              '$\\theta\,[\mathrm{rad}]$']):
        ax = plt.subplot(3, 1, i+1)
        ax.plot(t, y[:, i])
        ax.set_ylabel(name)
        ax.set_xlim([t[0], t[-1]])
        plt.grid()
    plt.xlabel('$Time\,\mathrm{[s]}$')


def plot_voltage(t, v):
    f, ax = plt.subplots()
    ax.plot(t, v)
    ax.set_xlim(t[0], t[-1])
    ax.set_ylabel('$V_{in}\,[\mathrm{Volts}]$')


def run_current_control(m, x0, tf, ts):

    def pid_simulation(x, t, args = ()):
        # Motor and controller are inside optional args.
        motor, ctrl = args

        # Motor input voltage is the output of the controller calculated with setp method of motor.
        # The term x[1] * motor.params['kc'] is the back-emf cancelation of the motor.
        motor.vin = ctrl.step(x[0]) + x[1] * motor.params['kc']

        # Returning motor voltage as a tuple allows us to obtain in main fcn.
        return (motor.vin,)

    i_des = 1.0 # Desired current: 1 Amp.

    # Calculate controller gains according to the relations obtain from analysis.
    kp, ki = compute_gains_current_controller(50e-3, R = m.params['R'], L = m.params['L'])

    # Construct a PI controller with calculated gains.
    ctrl = PID_controller(kp = kp, ki = ki, ts = ts)

    # Set desired current to controller.
    ctrl.target = i_des

    # Simulation is executed where system dynamics is calculated by the dynamics method of DC+motor
    # class.
    t, sol, vin = simulate(m.dynamics, x0, 0., tf, ts, simulation = pid_simulation,
                           args = (m, ctrl))
    plot_results(t, sol)
    plt.show()


if __name__ == '__main__':
    tf, ts = 0.3, 1.0e-3
    xInit = [0.0, 0.0, 0.0]
    m = DC_Motor(R=0.83, L=2.31e-3, J=2.37e-4, kc=0.128, kt=0.128, kf=0.001697)
    run_current_control(m, xInit, tf, ts)


