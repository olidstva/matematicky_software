# Nové parametry
epsilon = 0.1
eta = 0.1
mu = 0.3

def extended_model(state, t):
    x, y, z = state
    dxdt = alpha * x - beta * x * y
    dydt = delta * x * y - gamma * y - epsilon * y * z
    dzdt = eta * y * z - mu * z
    return [dxdt, dydt, dzdt]

# Počáteční hodnoty
x0 = 10
y0 = 5
z0 = 2
t = np.linspace(0, 50, 1000)

# Řešení
sol_ext = odeint(extended_model, [x0, y0, z0], t)

# Graf
plt.plot(t, sol_ext[:, 0], label='Kořist (x)')
plt.plot(t, sol_ext[:, 1], label='Lovec (y)')
plt.plot(t, sol_ext[:, 2], label='Superpredátor (z)')
plt.xlabel('čas')
plt.ylabel('populace')
plt.legend()
plt.title('Rozšířený model Lotka–Volterra se 3 druhy')
plt.grid(True)
plt.show()
