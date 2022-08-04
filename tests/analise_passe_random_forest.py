import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import time
from random import randint
import analise_auxiliar

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

x_axis: numpy.ndarray = numpy.fromiter(range(1, 15, 1), dtype=numpy.uint16)
score_train: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
score_test: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
time_taken: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)

start: float = time.time()

cofs = None

for j, i in enumerate(x_axis):

    kmax = 50
    for k in range(kmax):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, random_state=randint(0, 1000)
        )

        model: RandomForestRegressor = RandomForestRegressor(
            n_estimators=60,
            criterion='squared_error',
            max_depth=i,
            bootstrap=True,
            n_jobs=None,
            random_state=randint(0, 1000),
            max_samples=0.5
        ).fit(X_train, y_train)

        if cofs is None:
            cofs = model.feature_importances_
        else:
            cofs += model.feature_importances_

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
print(f'Coeficientes: {100*cofs/cofs.sum()}')

print(f'Time taken average: {numpy.average(time_taken)}')

analise_auxiliar.plot_results(x_axis, score_test, score_train, time_taken,
                              x_label="maximum tree depth")
