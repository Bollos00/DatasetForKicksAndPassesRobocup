
import numpy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import joblib
import time
from random import randint
import analise_auxiliar

array_passe: numpy.ndarray = analise_auxiliar.get_array_from_pattern("LARC-2020-VIRTUAL/ALL/*Passe.csv")

X, y = analise_auxiliar.get_x_y_passes(array_passe, 1.02)

tree_out: DecisionTreeRegressor = DecisionTreeRegressor(
    criterion='mse',
    splitter='best',
    max_depth=4,
    min_samples_split=33*1e-3,
    min_samples_leaf=80*1e-3,
    min_weight_fraction_leaf=87e-3,
    max_features='auto',
    random_state=38,
    max_leaf_nodes=6,
    min_impurity_decrease=0,
    min_impurity_split=None,
    presort='deprecated',
    ccp_alpha=40
).fit(X, y)

joblib.dump(tree_out, "models/avaliacao_passe_tree.sav")

x_axis: numpy.ndarray = numpy.fromiter(range(1, 1000, 10), dtype=numpy.uint16)
score_train: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
score_test: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)

start: float = time.time()

for j, i in enumerate(x_axis):

    [X_train, X_test, y_train, y_test] = train_test_split(
        X, y, test_size=.2, random_state=i
        )

    model: DecisionTreeRegressor = DecisionTreeRegressor(
        criterion='mse',
        splitter='best',
        max_depth=4,
        min_samples_split=33*1e-3,
        min_samples_leaf=80*1e-3,
        min_weight_fraction_leaf=87e-3,
        max_features='auto',
        random_state=i,
        max_leaf_nodes=6,
        min_impurity_decrease=0,
        min_impurity_split=None,
        presort='deprecated',
        ccp_alpha=40
    ).fit(X_train, y_train)

    score_test[j] = model.score(X_test, y_test)
    score_train[j] = model.score(X_train, y_train)

end: float = time.time()

analise_auxiliar.print_time_of_each_prediction(start, end, numpy.size(x_axis), numpy.size(y))
analise_auxiliar.print_score(numpy.mean(score_test), numpy.mean(score_train))

analise_auxiliar.plot_results(x_axis, score_test, score_train)
