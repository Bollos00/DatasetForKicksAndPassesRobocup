
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
import joblib
import time
import analise_auxiliar
from random import randint
from typing import List


def customized_weights_linear(distances_weights: numpy.ndarray) -> numpy.ndarray:

    MAXIMUM_DISTANCE: numpy.float64 = numpy.float64(100)
    # create a new array weights with the same dimension distances and fill
    # the array with 0 element.
    for i, var in enumerate(distances_weights):  # for each prediction:
        if numpy.size(var) == 0:
            continue

        for j, _var in enumerate(distances_weights[i]):  # apply the weight function for each distance

            if (_var >= MAXIMUM_DISTANCE):
                distances_weights[i][j] = 0.0001
                continue
            distances_weights[i][j] = 1 - _var/MAXIMUM_DISTANCE

    return distances_weights


array_chute: numpy.ndarray = numpy.concatenate([
    analise_auxiliar.get_array_from_pattern("ROBOCUP-2021-VIRTUAL/DIVISION-B/ER_FORCE/ATA/*Shoot.csv"),
    # analise_auxiliar.get_array_from_pattern("ROBOCUP-2021-VIRTUAL/DIVISION-B/KIKS/ATA/*Shoot.csv"),
    analise_auxiliar.get_array_from_pattern("ROBOCUP-2021-VIRTUAL/DIVISION-B/RoboFEI/ATA/*Shoot.csv"),
    analise_auxiliar.get_array_from_pattern("ROBOCUP-2021-VIRTUAL/DIVISION-B/TIGERs_Mannheim/ATA/*Shoot.csv")
])

X, y = analise_auxiliar.get_x_y_shoots(array_chute, 1.01)

# knn_out: KNeighborsRegressor = KNeighborsRegressor(n_neighbors=15,
#                                                    weights=customized_weights_linear,
#                                                    n_jobs=1).fit(X, y)

# joblib.dump(knn_out, "models/avaliacao_chute_knn_with_weights.sav")

x_axis: numpy.ndarray = numpy.fromiter(range(1, 100, 1), dtype=numpy.uint16)
score_train: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
score_test: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)

start: float = time.time()

for j, i in enumerate(x_axis):

    [X_train, X_test, y_train, y_test] = train_test_split(
        X, y, test_size=.2, random_state=randint(0, 1000)
        )

    model: KNeighborsRegressor = KNeighborsRegressor(
        n_neighbors=10,
        weights=customized_weights_linear,
        algorithm='auto',
        leaf_size=30,
        p=4,
        metric='minkowski',
        metric_params=None,
        n_jobs=1
        ).fit(X_train, y_train)

    score_test[j] = model.score(X_test, y_test)
    score_train[j] = model.score(X_train, y_train)

end: float = time.time()

analise_auxiliar.print_time_of_each_prediction(start, end, numpy.size(x_axis), numpy.size(y))
analise_auxiliar.print_score(numpy.mean(score_test), numpy.mean(score_train))

analise_auxiliar.plot_results(x_axis, score_test, score_train)
