"""
Helper types.
"""
from scipy import stats


class EventA:
    """
    Describes an event object for modeling world.
    """
    distribs = {'norm': ['t', 'mean', 'std'],
                'uniform': ['t', 'a', 'b'],
                'expon': ['tau', 'lambda']}

    def __init__(self, id: str, distrib: str, params: tuple):
        """
        :param id: event's private name
        :param distrib: distribution's name
        :param params: distribution's parameters
        """
        # Check
        if distrib not in self.distribs.keys():
            print(f"Error: no such distribution ({distrib}) in class 'EventA'!")
            raise ValueError
        if len(params) != len(self.distribs[distrib]):
            raise IndexError
        # Init
        self.id, self.distrib = id, distrib
        self.tau = None
        self.m, self.sigma, self.lam, self.a, self.b = None, None, None, None, None
        if distrib == 'norm':
            self.tau, self.m, self.sigma = params[0], params[1], params[2]
        elif distrib == 'uniform':
            self.tau, self.a, self.b = params[0], params[1], params[2]
        elif distrib == 'expon':
            self.tau, self.lam = params[0], params[1]
            if self.lam == 0:
                print(f"Error {ValueError}: lambda parameter cannot be 0!")
                raise ValueError

    def __str__(self):
        out = f"Event '{self.id}' has distribution '{self.distrib}' with "
        if self.distrib == 'norm':
            out += f"t = {self.tau}, mean m = {self.m} and standard deviation sigma = {self.sigma}."
        elif self.distrib == 'uniform':
            out += f"t = {self.tau}, left a = {self.a} and right b = {self.b} boundaries."
        elif self.distrib == 'expon':
            out += f"t = {self.tau} and intensity lambda = {self.lam}."
        return out

    def get_prob(self, t_bias: float = 0.0, fail_prob: bool = True):
        """
        Calculates the probability of event's occurrence.
        :param t_bias: the time's bias if time scale conversion is needed
        :param fail_prob: calculate probability of failure (True) of success (False) - for exponential distribution
        :return: probability value
        """
        if self.distrib == 'norm':
            return stats.norm.cdf(self.tau + t_bias, loc=self.m, scale=self.sigma)
        if self.distrib == 'uniform':
            if fail_prob:
                return stats.uniform.sf(self.tau + t_bias, loc=self.a, scale=self.b - self.a)
            return stats.uniform.cdf(self.tau + t_bias, loc=self.a, scale=self.b - self.a)
        if self.distrib == 'expon':
            if fail_prob:
                return stats.expon.sf(self.tau, scale=1/self.lam)
            return stats.expon.cdf(self.tau, scale=1/self.lam)


class FlightComplex:
    """
    Describes flight complex object containing events.
    """
    def __init__(self, id: int, events: tuple):
        """
        :param id: flight complex private name
        :param events: tuple of events
        """
        self.id = id
        self.events = events

    def __str__(self):
        out = f"Flight complex '{self.id}' has events:"
        for i, ev in enumerate(self.events):
            out += f"\n {i + 1}) {ev}"
        return out
