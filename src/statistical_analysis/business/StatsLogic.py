from math import sqrt
from statistics import mean, stdev


def calculate_cohens_s(a, b):
    return (mean(a) - mean(b)) / (sqrt((stdev(a) ** 2 + stdev(b) ** 2) / 2))
