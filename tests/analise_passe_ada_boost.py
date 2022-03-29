
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
import time
import analise_auxiliar
from random import randint
from sklearn.tree import DecisionTreeRegressor

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

x_axis: numpy.ndarray = numpy.fromiter(range(1, 100, 1), dtype=numpy.uint16)
score_train: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
score_test: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
time_taken: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)

cofs = None

start: float = time.time()

for j, i in enumerate(x_axis):

    [X_train, X_test, y_train, y_test] = train_test_split(
        X, y, test_size=.2, random_state=7
    )

    tree_aux: DecisionTreeRegressor = DecisionTreeRegressor(
        criterion='squared_error',
        splitter='best',
        max_depth=3,
        min_samples_split=1*1e-3,
        min_samples_leaf=40*1e-3,
        min_weight_fraction_leaf=0,
        max_features='auto',
        random_state=1,
        max_leaf_nodes=None,
        min_impurity_decrease=0,
        ccp_alpha=0
    )

    model: AdaBoostRegressor = AdaBoostRegressor(
        base_estimator=tree_aux,
        n_estimators=i,
        learning_rate=50e-3,
        loss='square',
        random_state=10
    ).fit(X_train, y_train)

    if cofs is None:
        cofs = model.feature_importances_
    else:
        cofs += model.feature_importances_

    time_a = time.time()
    score_test[j] = model.score(X_test, y_test)
    score_train[j] = model.score(X_train, y_train)
    time_b = time.time()

    time_taken[j] = (time_b - time_a)*1e6/(X_train.shape[0] + X_test.shape[0])

    # analise_auxiliar.find_prediction_time(model, X.shape[1])
    # exit(0)

end: float = time.time()

analise_auxiliar.print_time_of_each_prediction(start, end, numpy.size(x_axis), numpy.size(y))
analise_auxiliar.print_score(numpy.mean(score_test), numpy.mean(score_train))

numpy.set_printoptions(precision=0)
print(f'Coeficientes: {100*cofs/cofs.sum()}')

analise_auxiliar.plot_results(x_axis, score_test, score_train, time_taken,
                              x_label="n_estimators")
