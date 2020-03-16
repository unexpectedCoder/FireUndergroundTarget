from typing import Union, List
from math import floor
from scipy import stats
from scipy.optimize import minimize
import numpy as np

from my_types import FlightComplex


# For task #1
def delivery_probs(complexes: Union[FlightComplex, List[FlightComplex]]):
    """
    Calculates the probability of successful payload delivery for each flight complex.
    :param complexes: list of flight complexes
    :return: probability of successful payload delivery for each flight complex
    """
    # If not a list of complexes
    if isinstance(complexes, FlightComplex):
        return delivery_prob(complexes)
    # Run
    probs = [1 for _ in range(len(complexes))]
    for i, complex in enumerate(complexes):
        t = 0
        for event in complex.events:
            if event.distrib == 'norm' or event.distrib == 'uniform':
                probs[i] *= event.get_prob(t_bias=t)
            else:
                probs[i] *= event.get_prob()
            t += event.tau if event.tau < 100 else 0
    return probs


def delivery_prob(complex: FlightComplex):
    """
    Calculates the probability of successful payload delivery for flight complex.
    :param complex: flight complex
    :return: the probability of successful payload delivery
    """
    prob = 1
    for event in complex.events:
        prob *= event.get_prob()
    return prob


def get_needed_n(p1: float, p_req: float) -> int:
    """
    Returns needed number of flight complexes for to complete a task with a required probability p_req.
    :param p1: success probability for the one flight complex
    :param p_req: required probability
    :return: number of flight complexes for to complete a task
    """
    if p1 < 0 or p1 >= 1:
        raise ValueError()
    if p_req < 0 or p_req >= 1:
        raise ValueError()
    return floor(np.log(1 - p_req) / np.log(1 - p1)) + 1


# For task #2
def stochastic_modelling(data: dict, to_files: bool = True):
    check_data(data)
    # Values preparing
    R, D1, D2, sigma, r0, rn, n_steps, N = prepare_values(data)     # N - amount of stochastic tests
    # Run
    with open('results/results.txt', 'w') as res_file, open('results/plot.txt', 'w') as plot_file:
        res_file.write("Stochastic modelling results")
        plot_file.write("x\ty")

        r = np.linspace(r0, rn, n_steps + 1)
        phi = stats.uniform(scale=2 * np.pi)
        x_fc_dist = [stats.norm(loc=ri, scale=sigma) for ri in r]
        y_fc_dist = stats.norm(scale=sigma)

        hit_points, probs = [], []
        for i, ri in enumerate(r):
            temp_probs = []
            for _ in range(N):
                # Coordinates of..
                x_fc, y_fc = x_fc_dist[i].rvs(), y_fc_dist.rvs()    # ..flight complex
                x_targ, y_targ = target_coords(R, phi.rvs())        # ..target
                d = np.sqrt((x_fc - x_targ)**2 + (y_fc - y_targ)**2)
                # Mean probability for current step
                temp_probs.append(defeat_law(d, D1, D2))
            mean_prob = np.array(temp_probs).mean()

            # Results
            hit_points.append(ri)
            probs.append(mean_prob)
            if to_files:
                res_file.write(f"\nStep {i + 1}: p = {mean_prob}")
                plot_file.write(f"\n{ri}\t{mean_prob}")

        return np.array(hit_points), np.array(probs)


def check_data(data: dict):
    """
    Checks that input data for stochastic modelling function has needed keys.
    :param data: input data for stochastic modelling
    """
    keys = ['R', 'D1', 'D2', 'sigma', 'r0', 'rn', 'n_steps', 'n_tests']
    for key in keys:
        if key not in data.keys():
            print(f"Error {ValueError}: something wrong with input data for stochastic modelling!")
            raise ValueError()


def prepare_values(data: dict) -> tuple:
    """
    Prepares values from input data for stochastic modelling.
    :param data: stochastic modelling input data
    :return: tuple of numeric data
    """
    return (float(data['R']), float(data['D1']), float(data['D2']), float(data['sigma']),
            float(data['r0']), float(data['rn']), int(data['n_steps']), int(data['n_tests']))


def defeat_law(d: float, D1: float, D2: float) -> float:
    """
    Target defeat law G(d).
    :param d: current distance from the hit point to the target
    :param D1: diameter of the zone of unconditional target defeat
    :param D2: diameter of the conditional target defeat zone
    :return: probability of the target defeat
    """
    if D1 < d <= D2:
        return (D2 - d) / (D2 - D1)
    if d <= D1:
        return 1.0
    return 0.0


def target_coords(R: float, phi: float) -> tuple:
    """
    Calculates target moving on circle coordinates.
    :param R: circle's radius
    :param phi: horizontal angle
    :return: (x, y) coordinates
    """
    return R * np.cos(phi), R * np.sin(phi)


def clarify(data):
    """
    Clarifies the range of data for stochastic modelling using user's input.
    :param data: current data
    """
    print("Set new x boundaries to clarify:")
    data['r0'] = input(" - left r: ")
    data['rn'] = input(" - right r: ")
    data['n_steps'] = input("Set amount of hit point's radius breaks: ")
    data['n_tests'] = input("Set amount of stochastic tests: ")


def approximate(x: np.ndarray, y: np.ndarray, pwr: int) -> tuple:
    """
    Approximates input data.
    :param x: data x-axes
    :param y: data y-axes
    :param pwr: polynomial power
    :return: approximation polynomial coefficients, extremum (max) of approximation function
    """
    polynom = np.polyfit(x, y, pwr)
    p_approx = np.polyval(polynom, x)
    extr_max = minimize(approx_func, x.mean(), args=(polynom[::-1], True))
    return p_approx, extr_max


def set_polynom_power() -> int:
    """
    Sets polynomial power using user's input.
    :return: polynomial power
    """
    pmin, pmax = 1, 5
    pwr = int(input(f"Approximation polynomial power (in range of integer [{pmin}; {pmax}]): "))
    if pwr < pmin:
        return pmin
    if pwr > pmax:
        return pmax
    return pwr


def approx_func(x: float, coeffs: np.ndarray, neg: bool = False) -> float:
    """
    Approximation function (polynomial).
    :param x: current argument
    :param coeffs: polynomial coefficients array (first element for the lowest 'x' power)
    :param neg: mul the result by -1 (True) or not (False)
    :return: approximation value
    """
    ans = 0
    for i, k in enumerate(coeffs):
        ans += k * x**i
    if neg:
        return -ans
    return ans
