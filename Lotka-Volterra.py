import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parametry
alpha = 1.1
beta = 0.4
delta = 0.1
gamma = 0.4

# ODE systém
def lotka_volterra(state, t):
    x, y = state
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y
    return [dxdt, dydt]

# Počáteční hodnoty
x0 = 10
y0 = 5
t = np.linspace(0, 50, 1000)

# Numerické řešení
sol = odeint(lotka_volterra, [x0, y0], t)

# Graf
plt.plot(t, sol[:, 0], label='Kořist (x)')
plt.plot(t, sol[:, 1], label='Lovec (y)')
plt.xlabel('čas')
plt.ylabel('populace')
plt.legend()
plt.title('Model Lotka–Volterra')
plt.grid(True)
plt.show()






