from glob import glob
import numpy
from matplotlib import pyplot
import time
from sklearn.model_selection import cross_val_score

pyplot.style.use('bmh')
# pyplot.style.use('dark_background')

pyplot.rc('font', size=17)


def get_array_from_pattern(pattern):

    array = []

    for f in glob("./{}".format(pattern)):
        array.append(
            numpy.genfromtxt(
                f,
                dtype=numpy.uint8,
                delimiter=";",
                skip_header=1
            )
        )

    return numpy.concatenate(array)


def get_x_y_passes(array_passes: numpy.ndarray, version=0):

    if version >= 1:
        return (array_passes[:, [0, 1, 2, 3, 4, 5, 6, 7]], array_passes[:, 8])
    return (array_passes[:, [1, 2, 3, 4, 5, 6, 7, 8]], array_passes[:, 0])


def get_x_y_shoots(array_shoot: numpy.ndarray, version=0):
    if version >= 1:
        return (array_shoot[:, [0, 1, 2]], array_shoot[:, 3])
    return (array_shoot[:, [1, 2, 3]], array_shoot[:, 0])


def print_time_of_each_prediction(start: float, end: float, x_size: int, y_size: int):
    print("Total time: {:.3f} ms".format((end-start)*1e3))


def print_score(score_test: float, score_train: float):
    print("Score test: {:.3f}/100".format(100*score_test))
    print("Score train: {:.3f}/100".format(100*score_train))


def plot_results(x_axis,
                 score_test,
                 score_train,
                 time_taken,
                 x_label: str = '???'):

    fig, host = pyplot.subplots()

    par1 = host.twinx()

    p_time, = host.plot(x_axis, time_taken, 'b-', label='Prediction time (µs)')
    p_score_train, = par1.plot(x_axis, score_train, 'r-', label='Train score')
    p_test_score, = par1.plot(x_axis, score_test, 'r--', label='Test score')

    pyplot.xlim(0, x_axis.max())

    par1.set_ylim(0, 1)

    lns = [p_score_train, p_test_score, p_time]
    pyplot.legend(handles=lns,
                  loc="upper left")

    host.set_ylabel("Average prediction time (µs)")
    par1.set_ylabel("R^2 score")

    host.set_xlabel(x_label)

    par1.grid(visible=False)
    host.grid(visible=False)
    host.grid(visible=True, axis='x', which='both')

    host.yaxis.label.set_color(p_time.get_color())
    par1.yaxis.label.set_color(p_score_train.get_color())

    fig.tight_layout()

    pyplot.show()


def find_prediction_time(model, x_size, predictions=10000):
    X_in = numpy.random.randint(
        low=0, high=250, size=(predictions, x_size), dtype=numpy.uint8
    )

    start: float = time.time()

    model.predict(X_in)

    end: float = time.time()

    print("Time of each prediction: {:.3f} us".format((end-start)*1e6/predictions))


def knn_feature_accuracy(model, X, y):

    scores = numpy.zeros(shape=(X.shape[1]))

    for i in range(X.shape[1]):
        X_alt = X[:, i].reshape(-1, 1)
        scores[i] = cross_val_score(model, X_alt, y, cv=10).mean()

    if numpy.min(scores) < 0:
        scores -= numpy.min(scores)

    scores /= numpy.sum(scores)

    # print(scores)

    return scores
