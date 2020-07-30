from glob import glob
import numpy
from typing import List
from matplotlib import pyplot

pyplot.style.use('dark_background')

def get_array_from_pattern(pattern):

    array = []

    for f in glob("../{}".format(pattern)):
        array.append(
            numpy.genfromtxt(
                f,
                dtype=numpy.uint8,
                delimiter=";",
                skip_header=1
            )
        )

    return numpy.concatenate(array)

def print_time_of_each_prediction(start: float, end: float, x_size: int, y_size: int):
    print("Time of each prediction: {:.3f} us".format(
        (end-start)*1e6/(x_size*y_size))
          )

def print_score(score_test: float, score_train: float):
    print("Score test: {:.3f}/100".format(100*score_test))
    print("Score train: {:.3f}/100".format(100*score_train))


def plot_results(x_axis: List[int],
                 score_test: List[float],
                 score_train: List[float],
                 x_label: str = '???',
                 y_label: str = 'score'):

    pyplot.plot(x_axis, score_train, 'r-', label='Train score')
    pyplot.plot(x_axis, score_test, 'c-', label='Test score')
    pyplot.xlabel(x_label)
    pyplot.ylabel(y_label)
    pyplot.legend(loc="upper right")
    pyplot.grid()

    pyplot.show()
