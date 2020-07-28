
import numpy
from sklearn.model_selection import train_test_split
from sklearn.forest import RandomForestRegressor
from matplotlib import pyplot
import joblib
import time
import analise_auxiliar
from typing import List, NewType

pyplot.style.use('dark_background')

array_chute: numpy.ndarray = analise_auxiliar.getArrayFromPattern("ALL/*Chute.csv")

y: numpy.ndarray = array_chute[:, 0]
X: numpy.ndarray = array_chute[:, [1, 2, 3]]

forest_out: RandomForestRegressor = RandomForestRegressor(
    n_estimators=10,
    max_depth=6,
    min_samples_split=.05,
    min_samples_leaf=.02,
    min_weight_fraction_leaf=.03,
    max_features='auto',
    max_leaf_nodes=20,
    min_impurity_decrease=40,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=28**2,
    verbose=0,
    warm_start=False,
    ccp_alpha=0,
    max_samples=None
    ).fit(X, y)

joblib.dump(forest_out, "models/avaliacao_chute_forest.sav")

x_axis: List[int] = list(range(1, 100, 1))
score_train: List[float] = []
score_test: List[float] = []

start: float = time.time()
for i in x_axis:

    [X_train, X_test, y_train, y_test] = train_test_split(
        X, y, test_size=.2, random_state=i
        )

    forest: RandomForestRegressor = RandomForestRegressor(
        n_estimators=10,
        max_depth=6,
        min_samples_split=.05,
        min_samples_leaf=.02,
        min_weight_fraction_leaf=.03,
        max_features='auto',
        max_leaf_nodes=20,
        min_impurity_decrease=40,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=i**2,
        verbose=0,
        warm_start=False,
        ccp_alpha=0,
        max_samples=None
    ).fit(X_train, y_train)

    score_test.append(forest.score(X_test, y_test))
    score_train.append(forest.score(X_train, y_train))

end: float = time.time()

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
