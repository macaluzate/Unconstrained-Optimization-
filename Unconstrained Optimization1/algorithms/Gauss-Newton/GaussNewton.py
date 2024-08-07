import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Definir la función objetivo
def f(x1, x2):
    return (1250 - 0.1 * x1 - 0.03 * x2) * x1 + (1500 - 0.04 * x1 - 0.1 * x2) * x2 - (500000 + 700 * x1 + 850 * x2)


# Crear una malla de puntos
x1 = np.linspace(-1000, 1000, 400)
x2 = np.linspace(-1000, 1000, 400)
x1, x2 = np.meshgrid(x1, x2)
z = f(x1, x2)


# Encontrar el máximo usando el método de Gauss-Newton (pero con minimización de la función original)
def gradient(x):
    x1, x2 = x
    dfdx1 = 550 - 0.2 * x1 - 0.07 * x2
    dfdx2 = 650 - 0.07 * x1 - 0.2 * x2
    return np.array([dfdx1, dfdx2])


def jacobian(x):
    x1, x2 = x
    d2fdx1x1 = -0.2
    d2fdx1x2 = -0.07
    d2fdx2x2 = -0.2
    return np.array([[d2fdx1x1, d2fdx1x2], [d2fdx1x2, d2fdx2x2]])


def gauss_newton(x0, tol=1e-6, max_iter=100):
    x = x0
    for _ in range(max_iter):
        grad = gradient(x)
        J = jacobian(x)
        JtJ = J.T @ J
        Jt_grad = J.T @ grad

        try:
            delta = np.linalg.solve(JtJ, Jt_grad)
        except np.linalg.LinAlgError:
            print("Problema al resolver el sistema lineal.")
            return x

        x = x - delta  # Minimización

        if np.linalg.norm(delta) < tol:
            break

    return x


# Valor inicial
x0 = np.array([0, 0])

# Encontrar el máximo (pero en realidad se está minimizando)
max_x = gauss_newton(x0)
max_value = f(max_x[0], max_x[1])

# Crear la figura y los ejes 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Superficie de la función
ax.plot_surface(x1, x2, z, cmap='viridis', alpha=0.8)

# Marcar el máximo encontrado (pero en realidad es un mínimo en términos de la función original)
ax.scatter(max_x[0], max_x[1], max_value, color='r', s=100, label='Máximo Encontrado')

# Etiquetas y título
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('f(x1, x2)')
ax.set_title('Superficie de la Función Objetivo y Máximo Encontrado')

# Descomentar para agregar leyenda si necesario
# ax.legend(loc='upper right')  # Cambiar ubicación o eliminar si es innecesario

plt.show()

print(f"Valores óptimos encontrados para el máximo: x1 = {max_x[0]}, x2 = {max_x[1]}")
print(f"Valor máximo de la función: {max_value}")
