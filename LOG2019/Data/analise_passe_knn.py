import numpy
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
import joblib
import time
import analise_auxiliar
from typing import List

array_passe: numpy.ndarray = analise_auxiliar.get_array_from_pattern("ALL/*Passe.csv")

y: numpy.ndarray = array_passe[:, 0]
X: numpy.ndarray = array_passe[:, [1, 2, 3,  4, 4, 6, 7, 8]]

knn_out: KNeighborsRegressor = KNeighborsRegressor(n_neighbors=30,
                                                   weights='uniform',
                                                   n_jobs=1).fit(X, y)

joblib.dump(knn_out, "models/avaliacao_passe_knn.sav")

x_axis: List[int] = list(range(1, 200))
score_train: List[float] = []
score_test: List[float] = []

start: float = time.time()

for i in x_axis:

    [X_train, X_test, y_train, y_test] = train_test_split(X,
                                                          y,
                                                          test_size=.2,
                                                          random_state=i)

    knn: KNeighborsRegressor = KNeighborsRegressor(n_neighbors=30,
                                                   weights='uniform').fit(X_train, y_train)
    # knn: RadiusNeighborsRegressor = RadiusNeighborsRegressor(radius=500, weights=customized_weights).fit(X_train, y_train)

    score_test.append(knn.score(X_test, y_test))
    score_train.append(knn.score(X_train, y_train))


end: float = time.time()

analise_auxiliar.print_time_of_each_prediction(start, end, numpy.size(x_axis), numpy.size(y))
analise_auxiliar.print_score(numpy.mean(score_test), numpy.mean(score_train))

analise_auxiliar.plot_results(x_axis, score_test, score_train)
