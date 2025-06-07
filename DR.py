import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time


def differential_solver(expr_str, x0, y0, x_end, step=0.01):
    # Symbolické řešení
    x, y = sp.symbols('x y')
    f_sym = sp.sympify(expr_str)
    
    # 1. Symbolické řešení
    start_sym = time.perf_counter()
    try:
        sol = sp.dsolve(sp.Eq(y.diff(x), f_sym), y, ics={y.subs(x, x0): y0})
        y_sym_func = sp.lambdify(x, sol.rhs, modules='numpy')
        time_sym = time.perf_counter() - start_sym
    except Exception as e:
        sol = None
        y_sym_func = None
        time_sym = None
        print("Symbolické řešení se nezdařilo:", e)

    # 2. Numerické řešení metodou Euler
    def euler_method(f, x0, y0, x_end, h):
        steps = int((x_end - x0) / h) + 1
        xs = np.linspace(x0, x_end, steps)
        ys = np.zeros(steps)
        ys[0] = y0
        for i in range(1, steps):
            ys[i] = ys[i - 1] + h * f(xs[i - 1], ys[i - 1])
        return xs, ys

    f_lambda = sp.lambdify((x, y), f_sym, modules='numpy')

    start_euler = time.perf_counter()
    xs_euler, ys_euler = euler_method(f_lambda, x0, y0, x_end, step)
    time_euler = time.perf_counter() - start_euler

    # 3. Numerické řešení pomocí solve_ivp
    def scipy_func(x, y): return f_lambda(x, y)

    start_ivp = time.perf_counter()
    sol_ivp = solve_ivp(scipy_func, [x0, x_end], [y0], method='RK45', t_eval=np.linspace(x0, x_end, 500))
    time_ivp = time.perf_counter() - start_ivp

    # Výstup + graf
    plt.figure(figsize=(10, 6))
    if y_sym_func:
        plt.plot(sol_ivp.t, y_sym_func(sol_ivp.t), label='Symbolické řešení', linestyle='--')
    plt.plot(xs_euler, ys_euler, label='Eulerova metoda')
    plt.plot(sol_ivp.t, sol_ivp.y[0], label='solve_ivp (RK45)')
    plt.legend()
    plt.title('Řešení diferenciální rovnice')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid(True)
    plt.show()

    # Benchmark
    print("⏱ Benchmark:")
    if time_sym is not None:
        print(f"Symbolické řešení: {time_sym:.6f} sekund")
    print(f"Eulerova metoda:     {time_euler:.6f} sekund")
    print(f"Solve_ivp (RK45):    {time_ivp:.6f} sekund")

    return {
        "symbolic_solution": str(sol) if sol else None,
        "euler_points": (xs_euler, ys_euler),
        "ivp_solution": sol_ivp,
        "times": {
            "symbolic": time_sym,
            "euler": time_euler,
            "ivp": time_ivp
        }
    }
