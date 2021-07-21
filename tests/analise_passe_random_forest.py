import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import time
from random import randint
import analise_auxiliar

array_passe: numpy.ndarray = analise_auxiliar.get_array_from_pattern("LARC-2020-VIRTUAL/ALL/*Passe.csv")

X, y = analise_auxiliar.get_x_y_passes(array_passe, 1.02)

model_out: RandomForestRegressor = RandomForestRegressor(
    n_estimators=20,
    max_depth=5,
    min_samples_split=30*1e-3,
    min_samples_leaf=1,
    min_weight_fraction_leaf=50*1e-3,
    max_features='auto',
    max_leaf_nodes=None,
    min_impurity_decrease=0,
    min_impurity_split=None,
    bootstrap=True,
    oob_score=False,
    n_jobs=None,
    random_state=5*8,
    verbose=0,
    warm_start=False,
    ccp_alpha=0.0,
    max_samples=None
).fit(X, y)

joblib.dump(model_out, "models/avaliacao_passe_forest.sav")

x_axis: numpy.ndarray = numpy.fromiter(range(1, 500, 10), dtype=numpy.uint16)
score_train: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
score_test: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)

start: float = time.time()


for j, i in enumerate(x_axis):

    r = randint(0, 999)

    [X_train, X_test, y_train, y_test] = train_test_split(
        X, y, test_size=.2, random_state=r*3
        )

    model: RandomForestRegressor = RandomForestRegressor(
        n_estimators=25,
        max_depth=3,
        min_samples_split=50*1e-3,
        min_samples_leaf=1,
        min_weight_fraction_leaf=100*1e-3,
        max_features='auto',
        max_leaf_nodes=None,
        min_impurity_decrease=0,
        min_impurity_split=None,
        bootstrap=True,
        oob_score=False,
        n_jobs=None,
        random_state=r*8,
        verbose=0,
        warm_start=False,
        ccp_alpha=0.0,
        max_samples=None
    ).fit(X_train, y_train)

    # print(model.feature_importances_, '\t', i)

    score_test[j] = model.score(X_test, y_test)
    score_train[j] = model.score(X_train, y_train)

end: float = time.time()

analise_auxiliar.print_time_of_each_prediction(start, end, numpy.size(x_axis), numpy.size(y))
analise_auxiliar.print_score(numpy.mean(score_test), numpy.mean(score_train))

analise_auxiliar.plot_results(x_axis, score_test, score_train)
