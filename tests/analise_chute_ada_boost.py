
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
import joblib
import time
import analise_auxiliar
from random import randint
from sklearn.tree import DecisionTreeRegressor

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

X, y = analise_auxiliar.get_x_y_shoots(array_chute, 1.01)

x_axis: numpy.ndarray = numpy.fromiter(range(0, 500, 1), dtype=numpy.uint16)
score_train: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
score_test: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)

cofs = None

start: float = time.time()

for j, i in enumerate(x_axis):

    [X_train, X_test, y_train, y_test] = train_test_split(
        X, y, test_size=.2, random_state=randint(0, 1000)
    )

    tree_aux: DecisionTreeRegressor = DecisionTreeRegressor(
        criterion='squared_error',
        splitter='best',
        max_depth=2,
        min_samples_split=1*1e-3,
        min_samples_leaf=100*1e-3,
        min_weight_fraction_leaf=0,
        max_features='auto',
        random_state=randint(0, 1000),
        max_leaf_nodes=None,
        min_impurity_decrease=0,
        ccp_alpha=0
    )
    model: AdaBoostRegressor = AdaBoostRegressor(
        base_estimator=tree_aux,
        n_estimators=30,
        learning_rate=100*1e-3,
        loss='square',
        random_state=randint(0, 1000)
    ).fit(X_train, y_train)

    if cofs is None:
        cofs = model.feature_importances_
    else:
        cofs += model.feature_importances_

    score_test[j] = model.score(X_test, y_test)
    score_train[j] = model.score(X_train, y_train)

    # analise_auxiliar.find_prediction_time(model, X.shape[1])
    # exit(0)

end: float = time.time()

analise_auxiliar.print_time_of_each_prediction(start, end, numpy.size(x_axis), numpy.size(y))
analise_auxiliar.print_score(numpy.mean(score_test), numpy.mean(score_train))

numpy.set_printoptions(precision=0)
print(f'Coeficientes: {100*cofs/cofs.sum()}')

# analise_auxiliar.plot_results(x_axis, score_test, score_train)
