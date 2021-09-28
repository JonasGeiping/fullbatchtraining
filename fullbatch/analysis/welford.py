""" Implement stable online calcuation of mean / std."""


class WelfordAccumulation():
    """Stable algorithm for online accumulation of mean and standard deviation of a vector.
    Assume the input is a 1-dim torch vector.

    M2 denotes the sum-of-squares difference to the current mean.
    This class also outputs the average Euclidean norm of the input vector.

    For more details see wikipedia:
    https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's%20online%20algorithm
    """

    def __init__(self):
        self.count = 0
        self.mean = 0
        self.M2 = 0

        self.norm_estimate = 0  # let's just do this as well
        self.squared_norm_estimate = 0

    def __call__(self, vector):
        self.count += 1
        current_delta = vector - self.mean
        self.mean += current_delta / self.count
        corrected_delta = vector - self.mean
        self.M2 += current_delta * corrected_delta

        self.norm_estimate += vector.pow(2).sum().sqrt()
        self.squared_norm_estimate += vector.pow(2).sum()

    def finalize(self):
        mean = self.mean
        sample_variance = self.M2 / (self.count - 1)
        sample_std = sample_variance.sqrt()
        euclidean_norm = self.norm_estimate / self.count
        squared_norm = self.squared_norm_estimate / self.count
        return mean, sample_variance, sample_std, euclidean_norm, squared_norm
