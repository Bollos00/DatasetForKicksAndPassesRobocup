
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import time
from random import randint
import analise_auxiliar

# array_chute: numpy.ndarray = numpy.concatenate([
#     analise_auxiliar.get_array_from_pattern("ROBOCUP-2019/ER_FORCE/ATA/*Chute.csv"),
#     analise_auxiliar.get_array_from_pattern("ROBOCUP-2019/ZJUNlict/ATA/*Chute.csv")
# ])

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

X, y = analise_auxiliar.get_x_y_shoots(array_chute, 1.01)

x_axis: numpy.ndarray = numpy.linspace(start=30e-3, stop=70e-3, num=10, dtype=numpy.float64)
score_train: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
score_test: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
time_taken: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)

cofs = None

start: float = time.time()

for j, i in enumerate(x_axis):

    kmax = 100
    for k in range(kmax):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=.2, random_state=randint(0, 1000)
        )

        model: GradientBoostingRegressor = GradientBoostingRegressor(
            loss='squared_error',
            learning_rate=50e-3,
            n_estimators=30,
            max_depth=3,
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

    print(f'{j}/{x_axis.shape[0]}')

end: float = time.time()

analise_auxiliar.print_time_of_each_prediction(start, end, numpy.size(x_axis), numpy.size(y))
analise_auxiliar.print_score(numpy.mean(score_test), numpy.mean(score_train))

numpy.set_printoptions(precision=0)
print(f'Coeficientes: {100*cofs/cofs.sum()}')

print(f'Time taken average: {numpy.average(time_taken)}')

analise_auxiliar.plot_results(x_axis, score_test, score_train, time_taken,
                              x_label="learning rate")
