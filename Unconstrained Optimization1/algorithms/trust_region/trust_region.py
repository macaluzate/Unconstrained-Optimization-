import numpy as np
import matplotlib.pyplot as plt
from algorithms.trust_region import dogleg_method
from algorithms import functions

norm_p = dogleg_method.norm_p
f_provider = functions.FunctionProvider()


def approximate_model_generator(x, b_matrix):
    return lambda p: f_provider.f(*x) + np.dot(f_provider.grad(*x), p) + 0.5 * np.dot(np.dot(p, b_matrix), p)


# Parámetros globales
RHO_TOLERANCE = 1 / .000001
DELTA_HAT = 1.5
ETA = 1 / 5
EQUALITY_TOLERANCE = .000001


def trust_region(x, function_provider, subproblem_solver=dogleg_method.dog_leg, max_steps=2000, plot=False):
    global f_provider
    f_provider = function_provider
    delta = DELTA_HAT / 2
    step = 0
    rho = RHO_TOLERANCE
    initial_x = x
    plot_y = []

    while step < max_steps and abs(rho) <= RHO_TOLERANCE:
        b_matrix = f_provider.hessian(*x)
        p = subproblem_solver(x, b_matrix, delta)
        approximate_model = approximate_model_generator(x, b_matrix)
        delta_f = f_provider.f(*x) - f_provider.f(*(x + p))  # Diferencia en el valor de la función
        delta_m = approximate_model(np.zeros(np.shape(x)[0])) - approximate_model(
            p)  # Diferencia en el valor del modelo aproximado

        if abs(delta_m) < 1e-8:
            delta_m = 1e-8

        rho = delta_f / delta_m

        if rho < .25:
            delta = .25 * delta
        else:
            if rho > .75 and abs(norm_p(p) - delta) < EQUALITY_TOLERANCE:
                delta = min(2 * delta, DELTA_HAT)

        if rho > ETA:
            x = x + p

        step += 1
        plot_y.append(rho)

    if plot:
        plt.title(f'Rho on each step for point ({initial_x[0]}, {initial_x[1]}) with max_steps={max_steps}')
        plt.ylabel('Rho')
        plt.xlabel('step')
        plt.plot(plot_y)
        plt.show()

    print('number of steps: ', step)
    return x


# Valores iniciales
x0 = np.array([0, 0])

# Eel método de región de confianza con 2000 iteraciones
print("Running with 2000 iterations")
trust_region(x0, dogleg_method.f_provider, subproblem_solver=dogleg_method.dog_leg, max_steps=2000, plot=True)

# el método de región de confianza con 3000 iteraciones
print("Running with 3000 iterations")
trust_region(x0, dogleg_method.f_provider, subproblem_solver=dogleg_method.dog_leg, max_steps=3000, plot=True)

