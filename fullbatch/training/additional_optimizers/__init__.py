"""Non-standard optimizers."""
from .fista import FISTA, FISTALineSearch, SGDLineSearch
from .lars import LARS
from .sgd_agc import SGD_AGC
from .scheduler import GradualWarmupScheduler
from .lbfgs import LBFGS
from .sam import SAM
from .sgd_linesearch import RestartingLineSearch, NonMonotoneLinesearch, WolfeGradientDescent
from .adaptive_clipping import AdaptiveGradientClipping

__all__ = ['FISTA', 'FISTALineSearch', 'SGDLineSearch', 'LARS', 'SGD_AGC', 'GradualWarmupScheduler', 'LBFGS', 'SAM',
           'RestartingLineSearch', 'NonMonotoneLinesearch', 'WolfeGradientDescent', 'AdaptiveGradientClipping']
