import numpy as np
from algorithms import functions



def norm_p(vector):
    return np.linalg.norm(vector)

f_provider = functions.FunctionProvider()


def dog_leg(x, b_matrix, delta):
    g_matrix = f_provider.grad(*x)
    gT_b_g = np.dot(np.dot(g_matrix, b_matrix), g_matrix)
    gT_g = np.dot(g_matrix, g_matrix)
    grad_path_best_alpha = gT_g / (gT_b_g + 1e-8)  # Para evitar divisiones por cero
    p_u = -grad_path_best_alpha * g_matrix
    p_b = -np.dot(np.linalg.inv(b_matrix + 1e-8 * np.eye(len(b_matrix))), g_matrix)  # Evitar singularidades
    tau = delta / norm_p(p_u)
    if tau <= 1:
        return tau * p_u
    elif tau <= 2:
        return p_u + (tau - 1) * (p_b - p_u)
    else:
        p_b_norm = norm_p(p_b)
        if p_b_norm <= delta:
            return p_b
        else:
            return p_b * delta / p_b_norm