from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt

from my_types import FlightComplex
import data_init
import solver


def main():
    try:
        # Task 1
        probs, res = task1()
        print(f"Resulting probabilities: {probs}")
        print(res)
        # Task 2
        extr_max = task2()
        print(f"Extremum:"
              f"\n - fun: {-extr_max.fun}"
              f"\n - x: {extr_max.x}")
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
        events = data_init.read_events(path)
        complexes.append(FlightComplex(i + 1, events))
    # Results
    probs = np.array(solver.delivery_probs(complexes))
    return probs, f"Flight complex number {probs.argmax() + 1} has max probability P = {round(probs.max(), 5)}"


def task2():
    print("\n\tTASK #2")
    # Init
    data = data_init.read_csv(data_init.txt2csv('init/data.txt', 'init/data.csv'))[0]
    # Run
    hit_points, p_mean = solver.stochastic_modelling(data)
    # Visualization
    plot(hit_points, p_mean)
    # Clarify
    solver.clarify(data)
    pwr = solver.set_polynom_power()
    # Run
    hit_points, p_mean = solver.stochastic_modelling(data)
    # Approximation
    p_approx, extr_max = solver.approximate(hit_points, p_mean, pwr)
    # Visualization
    plot(hit_points, p_mean, p_approx)

    with open('results/clarify.txt', 'w') as file:
        file.write(f"Adjusted results\n"
                   f" - max probability: {-extr_max.fun}\n"
                   f" - hit point's r-coordinate: {extr_max.x}")

    return extr_max


def plot(x: np.ndarray, y: np.ndarray, y_approx: np.ndarray = None, title: str = None):
    """
    Makes and show plots for results of stochastic modelling.
    :param x: x-argument for plots
    :param y: y-argument for plots
    :param y_approx: approximated y-arguments for plots
    :param title: title all of the plots
    """
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
