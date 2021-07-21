
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
import joblib
import time
import analise_auxiliar
from random import randint
from sklearn.tree import DecisionTreeRegressor

array_passe: numpy.ndarray = analise_auxiliar.get_array_from_pattern("LARC-2020-VIRTUAL/ALL/*Passe.csv")

X, y = analise_auxiliar.get_x_y_passes(array_passe, 1.02)

# tree_aux_out: DecisionTreeRegressor = DecisionTreeRegressor(
#     criterion='mse',
#     splitter='best',
#     max_depth=3,
#     min_samples_split=100*1e-3,
#     min_samples_leaf=100*1e-3,
#     min_weight_fraction_leaf=100*1e-3,
#     max_features='auto',
#     random_state=2,
#     max_leaf_nodes=5,
#     min_impurity_decrease=0,
#     min_impurity_split=None,
#     presort='deprecated',
#     ccp_alpha=0
# )
# model_out: AdaBoostRegressor = AdaBoostRegressor(
#     base_estimator=tree_aux_out,
#     n_estimators=40,
#     learning_rate=100*1e-3,
#     loss='linear',
#     random_state=2
# ).fit(X, y)
#
# joblib.dump(model_out, "models/avaliacao_passe_ada_boost.sav")

x_axis: numpy.ndarray = numpy.fromiter(range(1, 1000, 10), dtype=numpy.uint16)
score_train: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
score_test: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)

start: float = time.time()

for j, i in enumerate(x_axis):

    r = randint(0, 999)
    [X_train, X_test, y_train, y_test] = train_test_split(
        X, y, test_size=.2, random_state=r*2
        )

    tree_aux: DecisionTreeRegressor = DecisionTreeRegressor(
        criterion='mse',
        splitter='best',
        max_depth=3,
        min_samples_split=400*1e-3,
        min_samples_leaf=100*1e-3,
        min_weight_fraction_leaf=250*1e-3,
        max_features='auto',
        random_state=r*5,
        max_leaf_nodes=5,
        min_impurity_decrease=0,
        min_impurity_split=None,
        ccp_alpha=0
    )
    model: AdaBoostRegressor = AdaBoostRegressor(
        base_estimator=tree_aux,
        n_estimators=20,
        learning_rate=500*1e-3,
        loss='linear',
        random_state=r*10
    ).fit(X_train, y_train)

    # print(model.feature_importances_, '\t', i)

    score_test[j] = model.score(X_test, y_test)
    score_train[j] = model.score(X_train, y_train)

end: float = time.time()

analise_auxiliar.print_time_of_each_prediction(start, end, numpy.size(x_axis), numpy.size(y))
analise_auxiliar.print_score(numpy.mean(score_test), numpy.mean(score_train))

analise_auxiliar.plot_results(x_axis, score_test, score_train)
