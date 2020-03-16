import numpy as np
import matplotlib.pyplot as plt

from my_types import FlightComplex
import data_init
import solver


def main():
    try:
        # Task 1
        probs, p_max = task1()
        print(f"Resulting probabilities: {probs}")
        print(f"Flight complex number {probs.argmax() + 1} has max probability P = {round(p_max, 5)}")
        # Optional
        if input("Do you want to know how many flight complexes are needed for success payload "
                 "delivering? (+/-): ") == '+':
            n_fc, p_req = amount_complexes(p_max)
            print(f"Answer: {n_fc} flight complex(es) with single probability "
                  f"{round(p_max, 5)}\ncan delivery payload to target with required probability {p_req}")

        # Task 2
        extr_max = task2()
        print(f"Max probability:"
              f"\n - P_max: {-extr_max.fun}"
              f"\n - r_max: {extr_max.x}")

        # Other calculations
        full_prob = -extr_max.fun * p_max
        print(f"\nFull probability for best flight complex: P = {full_prob}")
        with open('results/results.txt', 'a') as file:
            file.write(f"\nFull probability for best flight complex: P = {full_prob}")
            if input("Do you want to know how many such flight complexes are needed for success "
                     "target defeat? (+/-): ") == '+':
                n_fc, p_req = amount_complexes(full_prob)
                print(f"Answer: {n_fc} flight complex(es) with single probability "
                      f"{round(full_prob, 5)}\ncan defeat target with required probability {p_req}")
                file.write(f"\n{n_fc} flight complex(es) with single probability "
                           f"{round(full_prob, 5)} can defeat target with required probability {p_req}")
    except ValueError:
        print(f"Main Error {ValueError}!")
        exit(-1)
    else:
        print("All done...")
    return 0


def task1():
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
    return probs, probs.max()


def task2():
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

    with open('results/clarify.txt', 'w') as file:
        file.write(f"Max probability (adjusted result):\n"
                   f" - P_max: {-extr_max.fun}\n"
                   f" - r_max: {extr_max.x}")

    return extr_max


def amount_complexes(p: float) -> tuple:
    """
    Calculates necessary number of flight complexes for target task achievement.
    :param p: single flight complex probability
    :return: (number of flight complexes, required probability)
    """
    p_req = float(input(" - set required probability of target defeat [0; 1): "))
    return solver.get_needed_n(p, p_req), p_req


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
