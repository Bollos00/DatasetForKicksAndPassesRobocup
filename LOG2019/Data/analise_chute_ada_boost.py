
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostRegressor
from matplotlib import pyplot
import joblib
import time
import analise_auxiliar
from typing import List

pyplot.style.use('dark_background')

array_chute: numpy.ndarray = analise_auxiliar.getArrayFromPattern("ALL/*Chute.csv")

y: numpy.ndarray = array_chute[:, 0]
X: numpy.ndarray = array_chute[:, [1, 2, 3]]

ada_boost_out: AdaBoostRegressor = AdaBoostRegressor(
    base_estimator=None,
    n_estimators=5,
    learning_rate=85e-3,
    loss='linear',
    random_state=22
    ).fit(X, y)

joblib.dump(ada_boost_out, "models/avaliacao_chute_ada_boost.sav")

x_axis: List[int] = list(range(1, 100, 1))
score_train: List[float] = []
score_test: List[float] = []

start: float = time.time()

for i in x_axis:

    [X_train, X_test, y_train, y_test] = train_test_split(
        X, y, test_size=.2, random_state=i
        )

    ada_boost: AdaBoostRegressor = AdaBoostRegressor(
        base_estimator=None,
        n_estimators=5,
        learning_rate=85e-3,
        loss='linear',
        random_state=i
    ).fit(X_train, y_train)

    # ada_boost: AdaBoostRegressor = AdaBoostRegressor(
    #     base_estimator=None,
    #     n_estimators=10,
    #     learning_rate=2e-3,
    #     loss='linear',
    #     random_state=i
    # ).fit(X_train, y_train)

    score_test.append(ada_boost.score(X_test, y_test))
    score_train.append(ada_boost.score(X_train, y_train))

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
