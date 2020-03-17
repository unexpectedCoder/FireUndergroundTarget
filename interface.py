import numpy as np
import matplotlib.pyplot as plt

from my_types import FlightComplex
import data_init
import solver


def task1() -> tuple:
    """
    Solves the problem of choosing of the most efficient flight complex.
    :return: (probabilities for each flight complex, max probability)
    """
    print("\n*** TASK #1: choosing of the most efficient flight complex ***")
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
    print(f"Resulting probabilities: {probs}")
    print(f"Flight complex number {probs.argmax() + 1} has max probability P = {round(probs.max(), 5)}")

    return probs, probs.max()


def task2() -> tuple:
    """
    Solves the problem of the target defeat.
    :return: (hit point coordinate r, max target defeat probability)
    """
    print("\n*** TASK #2: calculating the target's defeat probability ***")
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

    # Results
    print(f"Max probability:"
          f"\n - max P: {extr_max.fun}"
          f"\n - r*: {extr_max.x}")
    with open('results/clarify.txt', 'w') as file:
        file.write(f"Max probability (adjusted result):"
                   f"\n - max P: {extr_max.fun}\n"
                   f"\n - r*: {extr_max.x}")

    return extr_max.x, extr_max.fun


def task3():
    """Solves the problem of the analyze input data influence."""
    print("\n*** TASK 3: analyze D1, D2 and sigma influence ***")
    data = data_init.read_csv('init/data.csv')[0]
    solver.clarify(data)
    share = int(input("Set variation percent: ")) / 100
    n = int(input("Set amount of variation steps: ")) + 1

    if n < 2 or share < 0 or share > 1:
        print(f"Error {ValueError}: n must be > 1 or variation percent must be in [0; 1]!")
        raise ValueError()

    D1 = np.linspace(float(data['D1']), float(data['D1']) * (1 + share), n)
    D2 = np.linspace(float(data['D2']), float(data['D2']) * (1 + share), n)
    sigma = np.linspace(float(data['sigma']), float(data['sigma']) * (1 + share), n)

    r_res, p_res, res_max = solver.analyze(data, D1, D2, sigma)

    print(f"Max probability for point (D1, D2, sigma) =\n"
          f"({res_max['D1']}, {res_max['D2']}, {res_max['sigma']}) is {res_max['prob']}")
    with open('results/analyze.txt', 'w') as file:
        file.write(f"Max probability for point (D1, D2, sigma) = "
                   f"({res_max['D1']}, {res_max['D2']}, {res_max['sigma']}) is {res_max['prob']}")

    return res_max


def get_n_fc(p: float) -> tuple:
    """
    Calculates necessary number of flight complexes for target task achievement.
    :param p: single flight complex probability
    :return: (number of flight complexes, required probability)
    """
    p_req = float(input(" - set required probability of target defeat [0; 1): "))
    return solver.get_needed_n(p, p_req), p_req


# Visualization
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


def color_plot(x: np.ndarray, y: np.ndarray):
    pass
