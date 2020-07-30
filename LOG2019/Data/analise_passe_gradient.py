
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import time
import analise_auxiliar
from typing import List

array_passe: numpy.ndarray = analise_auxiliar.get_array_from_pattern("ALL/*Passe.csv")

y: numpy.ndarray = array_passe[:, 0]
X: numpy.ndarray = array_passe[:, [1, 2, 3,  4, 4, 6, 7, 8]]

gradient_out: GradientBoostingRegressor = GradientBoostingRegressor(
    loss='ls',
    learning_rate=.4,
    n_estimators=5,
    subsample=.79,
    criterion='friedman_mse',
    min_samples_split=.01,
    min_samples_leaf=.07,
    min_weight_fraction_leaf=0,
    max_depth=10,
    min_impurity_decrease=0.0,
    min_impurity_split=None,
    init=None,
    random_state=5*18,
    max_features='auto',
    alpha=0.9,
    verbose=0,
    max_leaf_nodes=None,
    warm_start=False,
    presort='deprecated',
    validation_fraction=0.1,
    n_iter_no_change=None,
    tol=1e-4,
    ccp_alpha=0.0
    ).fit(X, y)

joblib.dump(gradient_out, "models/avaliacao_passe_gradient.sav")

x_axis: List[int] = list(range(1, 200))
score_train: List[float] = []
score_test: List[float] = []

start: float = time.time()
for i in x_axis:

    [X_train, X_test, y_train, y_test] = train_test_split(
        X, y, test_size=.2, random_state=i
        )

    gradient: GradientBoostingRegressor = GradientBoostingRegressor(
        loss='ls',
        learning_rate=.4,
        n_estimators=5,
        subsample=.79,
        criterion='friedman_mse',
        min_samples_split=.01,
        min_samples_leaf=.07,
        min_weight_fraction_leaf=0,
        max_depth=10,
        min_impurity_decrease=0.0,
        min_impurity_split=None,
        init=None,
        random_state=5*i,
        max_features='auto',
        alpha=0.9,
        verbose=0,
        max_leaf_nodes=None,
        warm_start=False,
        presort='deprecated',
        validation_fraction=0.1,
        n_iter_no_change=None,
        tol=1e-4,
        ccp_alpha=0.0
    ).fit(X_train, y_train)

    score_test.append(gradient.score(X_test, y_test))
    score_train.append(gradient.score(X_train, y_train))

end: float = time.time()

analise_auxiliar.print_time_of_each_prediction(start, end, numpy.size(x_axis), numpy.size(y))
analise_auxiliar.print_score(numpy.mean(score_test), numpy.mean(score_train))

analise_auxiliar.plot_results(x_axis, score_test, score_train)
