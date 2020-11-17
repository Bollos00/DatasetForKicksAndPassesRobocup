
import numpy
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import time
import analise_auxiliar

array_chute: numpy.ndarray = numpy.concatenate([
    analise_auxiliar.get_array_from_pattern("LOG2019/ER_FORCE/ATA/*Chute.csv"),
    analise_auxiliar.get_array_from_pattern("LOG2019/ZJUNlict/ATA/*Chute.csv")
])

y: numpy.ndarray = array_chute[:, 0]
X: numpy.ndarray = array_chute[:, [1, 2, 3]]

model_out: GradientBoostingRegressor = GradientBoostingRegressor(
    loss='ls',
    learning_rate=100*1e-3,
    n_estimators=200,
    subsample=1.0,
    criterion='friedman_mse',
    min_samples_split=250*1e-3,
    min_samples_leaf=200*1e-3,
    min_weight_fraction_leaf=150*1e-3,
    max_depth=4,
    min_impurity_decrease=1e3,
    min_impurity_split=None,
    init=None,
    random_state=7*20,
    max_features='auto',
    alpha=.9,
    verbose=0,
    max_leaf_nodes=3,
    warm_start=False,
    presort='deprecated',
    validation_fraction=.1,
    n_iter_no_change=None,
    tol=1e-4,
    ccp_alpha=0.0
).fit(X, y)

joblib.dump(model_out, "models/avaliacao_chute_gradient.sav")

x_axis: numpy.ndarray = numpy.fromiter(range(1, 50, 1), dtype=numpy.uint16)
score_train: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
score_test: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)

start: float = time.time()

for j, i in enumerate(x_axis):

    [X_train, X_test, y_train, y_test] = train_test_split(
        X, y, test_size=.2, random_state=i*5
        )

    model: GradientBoostingRegressor = GradientBoostingRegressor(
        loss='ls',
        learning_rate=100*1e-3,
        n_estimators=200,
        subsample=1.0,
        criterion='friedman_mse',
        min_samples_split=250*1e-3,
        min_samples_leaf=200*1e-3,
        min_weight_fraction_leaf=150*1e-3,
        max_depth=4,
        min_impurity_decrease=1e3,
        min_impurity_split=None,
        init=None,
        random_state=7*20,
        max_features='auto',
        alpha=.9,
        verbose=0,
        max_leaf_nodes=3,
        warm_start=False,
        presort='deprecated',
        validation_fraction=.1,
        n_iter_no_change=None,
        tol=1e-4,
        ccp_alpha=0.0
    ).fit(X_train, y_train)
    print(model.feature_importances_, '\t', i)
    score_test[j] = model.score(X_test, y_test)
    score_train[j] = model.score(X_train, y_train)

end: float = time.time()

analise_auxiliar.print_time_of_each_prediction(start, end, numpy.size(x_axis), numpy.size(y))
analise_auxiliar.print_score(numpy.mean(score_test), numpy.mean(score_train))

analise_auxiliar.plot_results(x_axis, score_test, score_train)
