import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def lr_circuit(t, y):
    r = 0.83
    l = 2.31e-3
    vin = 1
    A = -r/l
    B = 1./l
    return A * y + B * vin


def main():
    # Parameters for simulation
    Vin = 12
    R = 0.83
    L = 2.31e-3

    # call the function of LR circuit
    t0 = 0.
    tf = 0.05
    ts = 1e-4
    t_span = np.arange(t0, tf + ts, ts)
    sol = solve_ivp(lr_circuit, [0., 0.1], [0.], t_eval=t_span)

    plt.plot(sol.t, sol.y[0])
    plt.xlabel('Time')
    plt.ylabel('Response')
    plt.title('Step reponse 1. order')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    main()

