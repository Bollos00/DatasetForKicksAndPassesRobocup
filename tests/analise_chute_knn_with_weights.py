
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance
import time
import analise_auxiliar
from random import randint
from typing import List
from weight import *

array_chute: numpy.ndarray = numpy.concatenate([
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/ER_FORCE/ATA/*Shoot.csv"),
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/KIKS/ATA/*Shoot.csv"),
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/RoboFEI/ATA/*Shoot.csv"),
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/TIGERs_Mannheim/ATA/*Shoot.csv")
])

X, y = analise_auxiliar.get_x_y_shoots(array_chute, 1.12)

x_axis: numpy.ndarray = numpy.linspace(start=1, stop=100, num=10, dtype=numpy.int16)
score_train: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
score_test: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
time_taken: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)

cofs = None
Weights.set_weight_func(WeightParabolic.weight)

start: float = time.time()

for j, i in enumerate(x_axis):

    WeightParabolic.set_maximum_distance(75)
    kmax = 10
    for k in range(kmax):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, random_state=randint(0, 1000)
        )

        model: KNeighborsRegressor = KNeighborsRegressor(
            n_neighbors=20,
            weights=Weights.weights,
            n_jobs=1
        ).fit(X_train, y_train)

        importances = permutation_importance(
            model, X_train, y_train, random_state=randint(0, 1000)
        )

        if cofs is None:
            cofs = importances.importances_mean
        else:
            cofs += importances.importances_mean

        time_a = time.time()
        # score_test[j] += model.score(X_test, y_test)
        # score_train[j] += model.score(X_train, y_train)
        model.predict(X)
        time_b = time.time()
        
        time_taken[j] += (time_b - time_a)*1e6/(X.shape[0])

    score_test[j] /= kmax
    score_train[j] /= kmax
    time_taken[j] /= kmax

    importances = permutation_importance(
        model, X_train, y_train, random_state=randint(0, 1000)
    )

    if cofs is None:
        cofs = importances.importances_mean
    else:
        cofs += importances.importances_mean

end: float = time.time()

analise_auxiliar.print_time_of_each_prediction(start, end, numpy.size(x_axis), numpy.size(y))
analise_auxiliar.print_score(numpy.mean(score_test), numpy.mean(score_train))

numpy.set_printoptions(precision=0)
print(f'Coeficientes: {100*cofs/cofs.sum()}')

analise_auxiliar.plot_results(x_axis, score_test, score_train, time_taken,
                              x_label="neighbors $k$")

print(f'Time taken average: {numpy.average(time_taken)}')
