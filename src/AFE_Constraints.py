import numpy as np
import os


class Constraints(object):
    def __init__(self, dat):
        self.dat = dat

        self.func_r = self.smoothing_function(self.dat.fea[0])

    def np_operate(self, func, var):
        with np.errstate(all='raise'):
            try:
                new_fea = []
                samples = len(var[0]) if type(var) == tuple else len(var)
                for i in range(samples):
                    fea = func((var[0][i], var[1][i])) if type(var) == tuple else func(var[i])
                    # Check the eligibility
                    if (np.max(np.abs(fea[:])) < 1e-11) & (np.max(np.abs(fea[:])) > 1e11):
                        self.dat.printf("[" + str(self.dat.local_time()) + "]" + " Warning: Exceeding range",
                                        filename=os.path.join(self.dat.dir, str(self.dat.dim) + 'D_log.txt'))
                        return None
                    if np.max(np.abs(fea[:] - fea[0])) < 1e-8:
                        self.dat.printf("[" + str(self.dat.local_time()) + "]" + " Warning: Identical values",
                                        filename=os.path.join(self.dat.dir, str(self.dat.dim) + 'D_log.txt'))
                        return None
                    new_fea.append(fea)
                # If the new feature is valid
                return new_fea
            except FloatingPointError as e:
                self.dat.printf("[" + str(self.dat.local_time()) + "]" + " Warning: " + str(e),
                                filename=os.path.join(self.dat.dir, str(self.dat.dim) + 'D_log.txt'))
                return None

    def insert_constants(self, fea):
        clf = self.dat.clf
        c_fea = np.array([f.sum() for f in fea]).reshape(-1, 1)
        clf.fit(c_fea, self.dat.target)
        weights = [clf.coef_[0], clf.intercept_]
        func = lambda x: weights[0] * x + weights[1] / len(x)
        fea = self.np_operate(func, fea)
        if fea is not None:
            self.dat.constants.append(weights)
        return fea

    def smoothing_function(self, var, r_in=3, r_out=5):
        func = lambda r: (2 * r ** 2 - 3 * r_in ** 2 + + r_out ** 2) * (r_out ** 2 - r ** 2) ** 2 * (
                r_out ** 2 - r_in ** 2) ** (-3)
        func_r = self.np_operate(func, var)
        return func_r

    def multiply_smoothing_function(self, fea):
        mul = lambda x: (x[0] * x[1])
        var = (fea, self.func_r)
        var_smooth = self.np_operate(mul, var)
        if var_smooth is None:
            var_smooth = fea
        return var_smooth
