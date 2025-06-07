import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

def sir_model(y, t, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I
    dIdt = beta * S * I - gamma * I
    dRdt = gamma * I
    return [dSdt, dIdt, dRdt]

def run_simulation(R0, gamma=1/10, days=160, I0=0.01, N=1):
    beta = R0 * gamma
    S0 = N - I0
    R0_ = 0.0
    t = np.linspace(0, days, days)
    y0 = [S0, I0, R0_]
    sol = odeint(sir_model, y0, t, args=(beta, gamma))
    S, I, R = sol.T
    return t, S, I, R

def plot_result(t, S, I, R, disease_name):
    plt.figure(figsize=(10, 6))
    plt.plot(t, S, label='S (vnímaví)')
    plt.plot(t, I, label='I (infikovaní)')
    plt.plot(t, R, label='R (zotavení)')
    plt.xlabel('Čas (dny)')
    plt.ylabel('Podíl populace')
    plt.title(f'SIR model pro {disease_name}')
    plt.legend()
    plt.grid()
    plt.show()


diseases = {
    "Chřipka": 1.3,
    "COVID-19": 2.5,
    "Spalničky": 15.0,
    "Malárie": 1.2,
    "SARS": 3.0
}

for name, R0 in diseases.items():
    t, S, I, R = run_simulation(R0)
    plot_result(t, S, I, R, name)
    peak_day = t[np.argmax(I)]
    duration = np.where(I < 1e-4)[0]
    duration_days = t[duration[0]] if len(duration) > 0 else t[-1]
    final_infected = R[-1]
    print(f"Nemoc: {name}")
    print(f"▶ Vrchol epidemie: {peak_day:.1f} dní")
    print(f"▶ Trvání epidemie: {duration_days:.1f} dní")
    print(f"▶ Onemocní: {final_infected*100:.1f} % populace")
    print(f"▶ Zůstane zdravých: {(1 - final_infected)*100:.1f} %")
    print("------------")
