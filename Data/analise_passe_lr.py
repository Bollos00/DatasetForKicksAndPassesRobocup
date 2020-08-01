import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import joblib
import time
import analise_auxiliar

array_passe: numpy.ndarray = analise_auxiliar.get_array_from_pattern("ALL/*Passe.csv")

y: numpy.ndarray = array_passe[:, 0]
X: numpy.ndarray = array_passe[:, [1, 2, 3,  4, 4, 6, 7, 8]]

model_out: ElasticNet = ElasticNet(
    alpha=80,
    l1_ratio=1,
    fit_intercept=True,
    normalize=False,
    precompute=False,
    max_iter=1000,
    copy_X=True,
    tol=0.0001,
    warm_start=False,
    positive=False,
    random_state=28,
    selection='cyclic'
).fit(X, y)

print(model_out.coef_)
print(model_out.intercept_)

limit_factor_plus = (250 - model_out.intercept_) / (
    numpy.sum(numpy.fromiter((k for k in model_out.coef_ if k > 0), dtype=numpy.float64))*250
    )

limit_factor_minus = (model_out.intercept_) / numpy.abs(
    numpy.sum(numpy.fromiter((k for k in model_out.coef_ if k < 0), dtype=numpy.float64))*250
    )

limit = numpy.min([limit_factor_plus, limit_factor_minus])
print(limit)
model_out.coef_ = model_out.coef_*limit

print(model_out.coef_)
print(model_out.intercept_)

joblib.dump(model_out, "models/avaliacao_passe_lr.sav")

quit()

# for n in range(0,8):
#     z = numpy.polyfit(X[:, n], y, 2)
#     p = numpy.poly1d(z)
#     pyplot.plot(X[:, n], p(X[:, n]), 'go')

#     pyplot.plot(X[:, n], y, 'ro')

#     pyplot.show()

x_axis: numpy.ndarray = numpy.fromiter(range(0, 100, 1), dtype=numpy.uint16)
score_train: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
score_test: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)

start: float = time.time()

for j, i in enumerate(x_axis):

    [X_train, X_test, y_train, y_test] = train_test_split(X,
                                                          y,
                                                          test_size=.2,
                                                          random_state=i)

    model: ElasticNet = ElasticNet(
        alpha=80,
        l1_ratio=1,
        fit_intercept=True,
        normalize=False,
        precompute=False,
        max_iter=1000,
        copy_X=True,
        tol=0.0001,
        warm_start=False,
        positive=False,
        random_state=i,
        selection='cyclic'
    ).fit(X_train, y_train)

    score_test[j] = model.score(X_test, y_test)
    score_train[j] = model.score(X_train, y_train)


end: float = time.time()

analise_auxiliar.print_time_of_each_prediction(start, end, numpy.size(x_axis), numpy.size(y))
analise_auxiliar.print_score(numpy.mean(score_test), numpy.mean(score_train))

analise_auxiliar.plot_results(x_axis, score_test, score_train)
