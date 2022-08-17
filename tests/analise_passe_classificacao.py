
import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import time
import analise_auxiliar
from random import randint
from matplotlib import pyplot

array_passe: numpy.ndarray = numpy.concatenate([
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/ER_FORCE/ATA/*Pass.csv"),
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/KIKS/ATA/*Pass.csv"),
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/RoboCin/ATA/*Pass.csv"),
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/RoboFEI/ATA/*Pass.csv"),
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/TIGERs_Mannheim/ATA/*Pass.csv")
])

X, y = analise_auxiliar.get_x_y_passes(array_passe, 1.12)

x_axis: numpy.ndarray = numpy.fromiter(range(1, 100, 5), dtype=numpy.uint16)
score_train: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
score_test: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
time_taken: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)

cofs = None

start: float = time.time()

for j, i in enumerate(x_axis):

    kmax = 20
    for k in range(kmax):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2
        )

        model = KNeighborsClassifier(
            n_neighbors=i,
            n_jobs=1,
        ).fit(X_train, y_train)

        time_a = time.time()
        score_test[j] += model.score(X_test, y_test)
        score_train[j] += model.score(X_train, y_train)
        # model.predict(X)
        time_b = time.time()
        
        time_taken[j] += (time_b - time_a)*1e6/(X.shape[0])
        
    score_test[j] /= kmax
    score_train[j] /= kmax
    time_taken[j] /= kmax

    if score_test[j] < 0:
        score_test[j] = 0

    print(f'{j+1}/{x_axis.shape[0]}')


end: float = time.time()

analise_auxiliar.print_time_of_each_prediction(start, end, numpy.size(x_axis), numpy.size(y))
analise_auxiliar.print_score(numpy.mean(score_test), numpy.mean(score_train))


numpy.set_printoptions(precision=0)

analise_auxiliar.plot_results(x_axis, score_test, score_train, time_taken,
                              x_label="Training dataset size")
