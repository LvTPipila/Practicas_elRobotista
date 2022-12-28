#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 06 20:58:01 2022

Defines the class PID_Controller and the function to compute the controller's gains.

@author: malon
"""

import numpy as np

class PID_controller(object):
    
    def __init__(self, kp=0, ki=0, kd=0., target=0., I_MAX=None, I_MIN=None, ts=1e-3):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.target = target
        self.error = 0.
        self.feedback = 0.
        self.output = 0.
        self.ts = ts
        self._ie = 0.
        self._de = 0.
        self._preverr = 0.
        self.I_MIN = I_MIN
        self.I_MAX = I_MAX

    def reset(self):
        self._ie = 0.

    def step(self,y):
        self.error = self.target - y
        self._de = (self.error - self._preverr) / self.ts
        self._ie += self.error * self.ts
        if self.I_MAX is not None:
            self._ie = np.max((self._ie, self.I_MAX))
        if self.I_MIN is not None:
            self._ie = np.min((self._ie, self.I_MIN))
        self._preverr = self.error
        self.output = self.error * self.kp + self._ie * self.ki + self._de * self.kd
        return self.output

def compute_gains_current_controller(Ts, R = 0., L = 0.):
    assert(Ts > 0. and R > 0. and L > 0.)
    kp = 4 * L / Ts
    ki = 4 * R / Ts
    return kp, ki


