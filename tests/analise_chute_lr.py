
import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler
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
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/ER_FORCE/ATA/*Shoot.csv"),
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/KIKS/ATA/*Shoot.csv"),
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/RoboFEI/ATA/*Shoot.csv"),
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/TIGERs_Mannheim/ATA/*Shoot.csv")
])

X, y = analise_auxiliar.get_x_y_shoots(array_chute, 1.21)

x_axis: numpy.ndarray = numpy.fromiter(range(0, 500, 1), dtype=numpy.uint16)
score_train: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
score_test: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
time_taken: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)

cofs = None

start: float = time.time()
for j, i in enumerate(x_axis):

    [X_train, X_test, y_train, y_test] = train_test_split(
        X, y, test_size=.2, random_state=randint(0, 1000)
    )

    ss = StandardScaler()
    X_train_s = ss.fit_transform(X_train)
    X_test_s = ss.fit_transform(X_test)

    model: LinearRegression = LinearRegression(
        fit_intercept=True,
        copy_X=True, n_jobs=None, positive=False
    ).fit(X_train_s, y_train)


    if cofs is None:
        cofs = model.coef_
    else:
        cofs += model.coef_

    time_a = time.time()
    score_test[j] += model.score(X_test_s, y_test)
    score_train[j] += model.score(X_train_s, y_train)
    # model.predict(X)
    time_b = time.time()
    
    time_taken[j] = (time_b - time_a)*1e6/(X.shape[0])

    # analise_auxiliar.find_prediction_time(model, X.shape[1])
    # exit(0)

end: float = time.time()

cofs /= x_axis.shape[0]
analise_auxiliar.print_time_of_each_prediction(start,
                                               end,
                                               numpy.size(x_axis),
                                               numpy.size(y))
analise_auxiliar.print_score(numpy.mean(score_test), numpy.mean(score_train))

print(f'Coeficientes: {cofs}')
importance = numpy.abs(cofs)
importance /= numpy.sum(importance)
print(f'Importance: {importance}')


# analise_auxiliar.plot_results(x_axis, score_test, score_train)
print(f'Time taken average: {numpy.average(time_taken)}')
# for a in time_taken:
#     print(a)