
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib import pyplot
import joblib
import time
import analise_auxiliar
from typing import List

pyplot.style.use('dark_background')

array_chute: numpy.ndarray = analise_auxiliar.getArrayFromPattern("ALL/*Chute.csv")

y: numpy.ndarray = array_chute[:, 0]
X: numpy.ndarray = array_chute[:, [1, 2, 3]]

gradient_out: GradientBoostingRegressor = GradientBoostingRegressor(
    loss='ls',
    learning_rate=50e-3,
    n_estimators=40,
    subsample=1.0,
    criterion='friedman_mse',
    min_samples_split=0.01,
    min_samples_leaf=2e-3,
    min_weight_fraction_leaf=2e-3,
    max_depth=4,
    min_impurity_decrease=20e3,
    min_impurity_split=3e3,
    init=None,
    random_state=127*3,
    max_features='auto',
    alpha=.9,
    verbose=0,
    max_leaf_nodes=12,
    warm_start=False,
    presort='deprecated',
    validation_fraction=.1,
    n_iter_no_change=None,
    tol=1e-4,
    ccp_alpha=0.0
    ).fit(X, y)

joblib.dump(gradient_out, "models/avaliacao_chute_gradient.sav")

x_axis: List[int] = list(range(1, 100, 1))
score_train: List[float] = []
score_test: List[float] = []

start: float = time.time()
for i in x_axis:

    [X_train, X_test, y_train, y_test] = train_test_split(
        X, y, test_size=.2, random_state=i*5
        )

    gradient: GradientBoostingRegressor = GradientBoostingRegressor(
        loss='ls',
        learning_rate=50e-3,
        n_estimators=40,
        subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=0.01,
        min_samples_leaf=2e-3,
        min_weight_fraction_leaf=2e-3,
        max_depth=4,
        min_impurity_decrease=20e3,
        min_impurity_split=3e3,
        init=None,
        random_state=i*3,
        max_features='auto',
        alpha=.9,
        verbose=0,
        max_leaf_nodes=12,
        warm_start=False,
        presort='deprecated',
        validation_fraction=.1,
        n_iter_no_change=None,
        tol=1e-4,
        ccp_alpha=0.0
    ).fit(X_train, y_train)

    score_test.append(gradient.score(X_test, y_test))
    score_train.append(gradient.score(X_train, y_train))

end: float = time.time()

print('\n')
print("Score test: ", numpy.mean(score_test))
print("Score train: ", numpy.mean(score_train))
print("Time of operation: {} ms".format(
    (end-start)*1e3/(numpy.size(x_axis)*numpy.size(y)))
      )

pyplot.plot(x_axis, score_test, 'c-', label='Test score')
pyplot.plot(x_axis, score_train, 'r-', label='Train score')
pyplot.xlabel('???')
pyplot.ylabel('score')
pyplot.legend(loc="upper right")
pyplot.grid()

pyplot.show()
