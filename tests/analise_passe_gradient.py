
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
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

x_axis: numpy.ndarray = numpy.linspace(start=1e-3, stop=70e-3, num=10, dtype=numpy.float64)
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

        model: GradientBoostingRegressor = GradientBoostingRegressor(
            loss='squared_error',
            learning_rate=50e-3,
            n_estimators=40,
            max_depth=4,
            random_state=randint(0, 1000),
        ).fit(X_train, y_train)

        if cofs is None:
            cofs = model.feature_importances_
        else:
            cofs += model.feature_importances_

        time_a = time.time()
        # score_test[j] += model.score(X_test, y_test)
        # score_train[j] += model.score(X_train, y_train)
        model.predict(X)
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
analise_auxiliar.print_score(numpy.average(score_test), numpy.average(score_train))

numpy.set_printoptions(precision=0)
print(f'Coeficientes: {100*cofs/cofs.sum()}')

print(f'Time taken average: {numpy.average(time_taken)}')

analise_auxiliar.plot_results(x_axis, score_test, score_train, time_taken,
                              x_label="learning rate")
