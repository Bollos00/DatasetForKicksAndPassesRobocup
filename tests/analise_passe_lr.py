import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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

x_axis: numpy.ndarray = numpy.fromiter(range(0, 500, 1), dtype=numpy.uint16)
score_train: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
score_test: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)

cofs = None

start: float = time.time()
for j, i in enumerate(x_axis):

    [X_train, X_test, y_train, y_test] = train_test_split(X,
                                                          y,
                                                          test_size=.2,
                                                          random_state=randint(0, 1000))

    model: LinearRegression = LinearRegression(
        fit_intercept=True,
        copy_X=True, n_jobs=None, positive=False
    ).fit(X_train, y_train)

    # print(model.coef_)

    if cofs is None:
        cofs = model.coef_
    else:
        cofs += model.coef_

    score_test[j] = model.score(X_test, y_test)
    score_train[j] = model.score(X_train, y_train)

    analise_auxiliar.find_prediction_time(model, X.shape[1])
    exit(0)

end: float = time.time()

analise_auxiliar.print_time_of_each_prediction(start, end, numpy.size(x_axis), numpy.size(y))
analise_auxiliar.print_score(numpy.mean(score_test), numpy.mean(score_train))

cofs /= x_axis.shape[0]

print(f'Coeficientes: {cofs}')

print(f'Importância dos coeficientes: {abs(cofs)*100/sum(abs(cofs))}')

# analise_auxiliar.plot_results(x_axis, score_test, score_train)
