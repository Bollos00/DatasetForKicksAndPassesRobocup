import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import time
import analise_auxiliar
from typing import List

array_passe: numpy.ndarray = analise_auxiliar.get_array_from_pattern("ALL/*Passe.csv")

y: numpy.ndarray = array_passe[:, 0]
X: numpy.ndarray = array_passe[:, [1, 2, 3,  4, 4, 6, 7, 8]]

forest_out: RandomForestRegressor = RandomForestRegressor(
    n_estimators=33,
    max_depth=3,
    min_samples_split=5,
    min_samples_leaf=1,
    min_weight_fraction_leaf=0,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=5*36,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None
    ).fit(X, y)

joblib.dump(forest_out, "models/avaliacao_passe_forest.sav")

x_axis: List[int] = list(range(1, 200))
score_train: List[float] = []
score_test: List[float] = []

start: float = time.time()
for i in x_axis:

    [X_train, X_test, y_train, y_test] = train_test_split(
        X, y, test_size=.2, random_state=i
        )

    forest: RandomForestRegressor = RandomForestRegressor(
        n_estimators=33,
        max_depth=3,
        min_samples_split=5,
        min_samples_leaf=1,
        min_weight_fraction_leaf=0,
        max_features='auto',
        max_leaf_nodes=None,
        min_impurity_decrease=0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=5*i,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None
    ).fit(X_train, y_train)

    score_test.append(forest.score(X_test, y_test))
    score_train.append(forest.score(X_train, y_train))

end: float = time.time()

analise_auxiliar.print_time_of_each_prediction(start, end, numpy.size(x_axis), numpy.size(y))
analise_auxiliar.print_score(numpy.mean(score_test), numpy.mean(score_train))

analise_auxiliar.plot_results(x_axis, score_test, score_train)
