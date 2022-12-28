import numpy as np

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
        self.load = 0.
        """ Define the equations in state space. """
        self.A = np.array([[-R / L, -kc / L, 0.],
                           [kt / J, -kf / J, 0.],
                           [0.,      1.,     0.]])

        self.B = np.array([[1. / L, 0.],
                           [0., 1. / J],
                           [0., 0.]])

    def dynamics(self, x, t, *args):
        try:
            self.A.reshape(3, 3)
            self.B.reshape(3, 2)
            dx = self.A.dot(x.reshape(3, 1)) +\
                    self.B.dot(np.array([self.vin, self.load]).reshape(2,1))
        except Exception:
            raise ValueError('System dynamics dimension mismatch at t {}'.format(t))
        return dx.flatten()
