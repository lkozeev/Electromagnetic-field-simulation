import numpy as np
import matplotlib.pyplot as plt
from numba import jit

@jit(nopython=True)
def simulate_trajectory(E, B, q, m, dt, steps, pos, v, optimize_step):
    positions = np.zeros((steps // optimize_step, 3))
    velocities = np.zeros((steps // optimize_step, 3))

    for i in range(steps):
        if i % optimize_step == 0:
            positions[i // optimize_step] = pos
            velocities[i // optimize_step] = v

        f1 = q * (E + np.cross(v, B))
        a1 = f1 / m
        v2_ = v + a1 * dt
        f2 = q * (E + np.cross(v2_, B))
        a2 = f2 / m
        v2 = v + 0.5 * (a1 + a2) * dt
        pos += 0.5 * (v + v2) * dt

        v = v2

    return positions

# Константы
E = 1 * np.array([1, 0, 0], dtype=float)  # Электрическое поле
B = 1 * np.array([0, 0, 2], dtype=float)  # Магнитное поле
q = -1.0  # Заряд электрона
m = 1.0  # Масса электрона
dt = 1e-5  # Шаг времени
steps = 10**6  # Количество шагов
optimize_step = 100  # Оптимизируется кол-во записей в массив в n раз

pos = np.array([0, 0, 0], dtype=float)  # Позиция частицы
v = np.array([-1, 0, 0], dtype=float)  # Скорость частицы

# Симуляция траектории
positions = simulate_trajectory(E, B, q, m, dt, steps, pos, v, optimize_step)

# 3D визуализация траектории
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Установка масштаба графика
ax.set_xlim(-100, 100)  # Границы по оси x
ax.set_ylim(-100, 100)  # Границы по оси y
ax.set_zlim(-100, 100)  # Границы по оси z

ax.plot(positions[:, 0], positions[:, 1], positions[:, 2])
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.set_zlabel('z')
plt.show()