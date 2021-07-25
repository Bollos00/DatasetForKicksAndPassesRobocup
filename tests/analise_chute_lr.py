
import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
from matplotlib import pyplot
import joblib
import time
from random import randint
import analise_auxiliar

# array_chute: numpy.ndarray = numpy.concatenate([
#     analise_auxiliar.get_array_from_pattern("LARC-2020-VIRTUAL/RoboCin/ATA/*Chute.csv"),
#     analise_auxiliar.get_array_from_pattern("LARC-2020-VIRTUAL/RoboFEI/ATA/*Chute.csv"),
#     analise_auxiliar.get_array_from_pattern("LARC-2020-VIRTUAL/Maracatronics/ATA/*Chute.csv")
# ])

# array_chute: numpy.ndarray = analise_auxiliar.get_array_from_pattern("LARC-2020-VIRTUAL/ALL/*Chute.csv")

array_chute: numpy.ndarray = numpy.concatenate([
    analise_auxiliar.get_array_from_pattern("ROBOCUP-2021-VIRTUAL/DIVISION-B/ER_FORCE/ATA/*Shoot.csv"),
    # analise_auxiliar.get_array_from_pattern("ROBOCUP-2021-VIRTUAL/DIVISION-B/KIKS/ATA/*Shoot.csv"),
    analise_auxiliar.get_array_from_pattern("ROBOCUP-2021-VIRTUAL/DIVISION-B/RoboFEI/ATA/*Shoot.csv"),
    analise_auxiliar.get_array_from_pattern("ROBOCUP-2021-VIRTUAL/DIVISION-B/TIGERs_Mannheim/ATA/*Shoot.csv")
])

X, y = analise_auxiliar.get_x_y_shoots(array_chute, 1.01)

# model_out: ElasticNet = ElasticNet(
#     alpha=0,
#     l1_ratio=1,
#     fit_intercept=True,
#     normalize=False,
#     precompute=False,
#     max_iter=1000,
#     copy_X=True,
#     tol=0.0001,
#     warm_start=False,
#     positive=False,
#     random_state=28,
#     selection='cyclic'
# ).fit(X, y)
#
# joblib.dump(model_out, "models/avaliacao_chute_lr.sav")

x_axis: numpy.ndarray = numpy.fromiter(range(0, 200, 1), dtype=numpy.uint16)
score_train: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
score_test: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)

start: float = time.time()
for j, i in enumerate(x_axis):

    [X_train, X_test, y_train, y_test] = train_test_split(X,
                                                          y,
                                                          test_size=.2,
                                                          random_state=randint(0, 1000))

    model: ElasticNet = ElasticNet(
        alpha=.01,
        l1_ratio=.5,
        fit_intercept=True,
        normalize=False,
        precompute=False,
        max_iter=1000,
        copy_X=True,
        tol=1e-3,
        warm_start=False,
        positive=False,
        random_state=randint(0, 1000),
        selection='cyclic'
    ).fit(X_train, y_train)

    print(model.coef_)

    score_test[j] = model.score(X_test, y_test)
    score_train[j] = model.score(X_train, y_train)

end: float = time.time()

analise_auxiliar.print_time_of_each_prediction(start, end, numpy.size(x_axis), numpy.size(y))
analise_auxiliar.print_score(numpy.mean(score_test), numpy.mean(score_train))

analise_auxiliar.plot_results(x_axis, score_test, score_train)
