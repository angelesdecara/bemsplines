import numpy as np
import scipy.io
import math
from scipy import signal
import scipy.stats

class ERRORS:
    def __init__(self, actual, predicted):
        self.actual = actual
        self.predicted = predicted

    def calculate_smape(self) -> float:
        smape = 100 * np.mean(np.abs(self.predicted - self.actual) / (np.abs(self.predicted) + np.abs(self.actual)/2 ))
        return smape

    def calculate_rmse(self) -> float:
        rmse = np.square(np.subtract(self.actual, self.predicted)).mean()
        return math.sqrt(rmse)

    def calculate_cross_correlation(self) -> float:
        cross_correlation = signal.correlate(self.actual, self.predicted)
        return cross_correlation

    def calculate_correlation(self) -> float:
        pearson, pval = scipy.stats.pearsonr(self.actual, self.predicted)
        return pearson

    def combined_errors(self):
        return self.calculate_smape(), self.calculate_rmse(), self.calculate_cross_correlation()
