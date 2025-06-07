import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Parametry modelu
Pi = 0        # porodnost
beta = 0.0095 # infekčnost
delta = 0.0001 # přirozená smrt
zeta = 0.0001  # reanimace
alpha = 0.005  # zničení zombie

# Modelové rovnice
def zombie_model(y, t, Pi, beta, delta, zeta, alpha):
    S, Z, R = y
    dSdt = Pi - beta*S*Z - delta*S
    dZdt = beta*S*Z + zeta*R - alpha*S*Z
    dRdt = delta*S + alpha*S*Z - zeta*R
    return [dSdt, dZdt, dRdt]

# Počáteční stavy
S0 = 500
Z0 = 1
R0 = 0
y0 = [S0, Z0, R0]

# Časová osa
t = np.linspace(0, 30, 1000)

# Integrace
results = odeint(zombie_model, y0, t, args=(Pi, beta, delta, zeta, alpha))
S, Z, R = results.T

# Graf
plt.figure(figsize=(10,6))
plt.plot(t, S, label='Živí lidé')
plt.plot(t, Z, label='Zombie')
plt.plot(t, R, label='Mrtví')
plt.xlabel('Čas')
plt.ylabel('Populace')
plt.legend()
plt.title('Zombie apokalypsa – základní model SZR')
plt.grid()
plt.show()
