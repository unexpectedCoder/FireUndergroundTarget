from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

from my_types import FlightComplex
import data_reader
import solver


def main():
    try:
        # Task 1
        probs, res = task1()
        print(f"Resulting probabilities: {probs}")
        print(res)
        # Task 2
        task2()
    except ValueError:
        exit(-1)
    else:
        print("All done...")
    return 0


def task1():
    print("\n\tTASK #1")
    # Init
    n_complex = int(input("Amount of flight complexes: "))
    if n_complex < 1:
        print("Error: amount of complexes must be > 1!")
        raise ValueError
    paths = ['init/event_' + f"{i + 1}.txt" for i in range(n_complex)]
    # Solution
    complexes = []
    for i, path in enumerate(paths):
        events = data_reader.read_events(path)
        complexes.append(FlightComplex(i + 1, events))
    # Results
    probs = np.array(solver.delivery_probs(complexes))
    return probs, f"Flight complex number {probs.argmax() + 1} has max probability P = {round(probs.max(), 5)}"


def task2():
    print("\n\tTASK #2")
    # Init
    data = data_reader.read_csv(data_reader.txt2csv('init/data.txt', 'init/data.csv'))[0]
    hit_points, p_mean = solver.stochastic_modelling(data)
    # Visualization
    make_plot(hit_points, p_mean)

    # Clarify
    print("Set new x boundaries to clarify:")
    data['r0'] = input(" - left r: ")
    data['rn'] = input(" - right r: ")
    data['n_steps'] = input("Set amount of hit point's radius breaks: ")
    data['n_tests'] = input("Set amount of stochastic tests: ")
    pwr = int(input("Approximation polynomial power (in range of integer [1; 5]): "))
    if pwr < 1:
        pwr = 1
    elif pwr > 5:
        pwr = 5
    # Approximation
    hit_points, p_mean = solver.stochastic_modelling(data)
    polynom = np.polyfit(hit_points, p_mean, pwr)
    p_approx = np.polyval(polynom, hit_points)
    extr_max = minimize(approx_func, hit_points.mean(), args=(polynom[::-1]))
    print(f"Extremum:"
          f"\n - fun: {-extr_max.fun}"
          f"\n - x: {extr_max.x}")
    # Visualization
    make_plot(hit_points, p_mean, p_approx)

    with open('results/clarify.txt', 'w') as file:
        file.write(f"Adjusted results\n"
                   f" - max probability: {-extr_max.fun}\n"
                   f" - hit point's r-coordinate: {extr_max.x}")


def approx_func(x: float, coeffs: np.ndarray) -> float:
    ans = 0
    for i, k in enumerate(coeffs):
        ans += k * x**i
    return -ans


def make_plot(x: np.ndarray, y: np.ndarray, y_approx: np.ndarray = None, title: str = None):
    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(x, y, 'b.', label='Средняя вероятность')
    ax.plot(x, y, color='gray', lw=1, ls=':')
    if y_approx is not None:
        ax.plot(x, y_approx, color='green', lw=2, label='Аппроксимация')

    if title:
        ax.set_title(title)
    ax.set_xlabel('$r$, км')
    ax.set_ylabel('$P$')
    ax.legend()
    ax.grid()

    plt.show()


if __name__ == '__main__':
    main()
else:
    print("Fatal Error: no entering point!")
    exit(-1)
