import numpy as np

class FunctionProvider:
    def __init__(self):
        pass

    def f(self, x1, x2):
        return (1250 - 0.1 * x1 - 0.03 * x2) * x1 + (1500 - 0.04 * x1 - 0.1 * x2) * x2 - (500000 + 700 * x1 + 850 * x2)

    def grad(self, x1, x2):
        df_dx1 = 1250 - 0.2 * x1 - 0.07 * x2 - 700
        df_dx2 = 1500 - 0.2 * x2 - 0.07 * x1 - 850
        return np.array([df_dx1, df_dx2])

    def hessian(self, x1, x2):
        return np.array([
            [-0.2, -0.07],
            [-0.07, -0.2]
        ])


