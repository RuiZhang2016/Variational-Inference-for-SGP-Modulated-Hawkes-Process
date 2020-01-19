import numpy as np


class my_hawkes:
    def __init__(self, baseline, kernel_fun):
        self.kernel_fun = kernel_fun
        self.baseline = baseline

    def intensity(self, events, t):
        t_diff = t - events
        t_diff = t_diff[t_diff > 0]
        intensity_value = self.baseline + sum(self.kernel_fun(t_diff))
        return intensity_value

    def loglikelihood(self, events, T):
        interp_x = np.linspace(1e-6, T, 2048)
        interp_y = self.kernel_fun(interp_x)
        measure = interp_x[1]-interp_x[0]
        integral = self.baseline*(T-1e-6)
        for e in events:
            interp_y_e = interp_y[interp_x <= (T-e)]
            integral += np.sum(interp_y_e[:-1]+interp_y_e[1:])*measure*0.5
        if self.baseline == 0:
            log = sum(np.log([self.intensity(events, e) for e in events[1:]]))
        else:
            log = sum(np.log([self.intensity(events, e) for e in events]))
        ll = log - integral
        return ll