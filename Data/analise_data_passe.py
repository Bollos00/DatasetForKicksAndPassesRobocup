import numpy
import analise_auxiliar
import analise_data_auxiliar
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

array_passe: numpy.ndarray = analise_auxiliar.get_array_from_pattern("LOG2019/ZJUNlict/ATA/*Passe.csv")

X: numpy.ndarray = array_passe[:, [1, 2, 3, 4, 5, 6, 7, 8]]
y: numpy.ndarray = array_passe[:, 0]
# print(X.shape)
# print(X)
# print("=================")
#
# prep: preprocessing.PolynomialFeatures = preprocessing.PolynomialFeatures(
#     degree=2,
#     interaction_only=False,
#     include_bias=False,
#     order='C'
# )
# # prep: preprocessing.QuantileTransformer = preprocessing.QuantileTransformer(
# #     n_quantiles=1000,
# #     output_distribution='uniform',
# #     ignore_implicit_zeros=False,
# #     subsample=100000,
# #     random_state=0,
# #     copy=True
# # )
#
# new_X = prep.fit_transform(X)
# print("New dimension: {0}".format(new_X.shape[1]))
# for i in range(new_X.shape[1]):
#     analise_data_auxiliar.plot_data_analise(new_X[:, i],
#                                             y,
#                                             numpy.max(new_X[:, i]))
#
# x_axis: numpy.ndarray = numpy.fromiter(range(0, 200, 1), dtype=numpy.uint16)
# score_train: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
# score_test: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
# score_train_p: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
# score_test_p: numpy.ndarray = numpy.full(x_axis.shape, 0, dtype=numpy.float64)
#
# for j, i in enumerate(x_axis):
#     [X_train, X_test, y_train, y_test] = train_test_split(X,
#                                                           y,
#                                                           test_size=.2,
#                                                           random_state=i)
#     model: LinearRegression = LinearRegression().fit(X_train, y_train)
#     score_test[j] = model.score(X_test, y_test)
#     score_train[j] = model.score(X_train, y_train)
#
#     [X_train_p, X_test_p, y_train_p, y_test_p] = train_test_split(new_X,
#                                                                   y,
#                                                                   test_size=.2,
#                                                                   random_state=i)
#     model: LinearRegression = LinearRegression().fit(X_train_p, y_train_p)
#     score_test_p[j] = model.score(X_test_p, y_test_p)
#     score_train_p[j] = model.score(X_train_p, y_train_p)
#
# print("Without preprocessing: ")
# analise_auxiliar.print_score(numpy.mean(score_test), numpy.mean(score_train))
# print('\n', "With preprocessing")
# analise_auxiliar.print_score(numpy.mean(score_test_p), numpy.mean(score_train_p))
#
# quit()

angulo_livre_passe: numpy.ndarray = X[:, 0]
distancia_passe: numpy.ndarray = X[:, 1]
liberdade_marcacao_receptor: numpy.ndarray = X[:, 2]
angulo_redirect: numpy.ndarray = X[:, 3]
angulo_livre_chute_receptor: numpy.ndarray = X[:, 4]
distancia_receptor_gol: numpy.ndarray = X[:, 5]
liberdade_marcacao_passador: numpy.ndarray = X[:, 6]
delta_xis: numpy.ndarray = X[:, 7]

analise_data_auxiliar.plot_data_analise(angulo_livre_passe, y, x_label="Ângulo livre passe")
analise_data_auxiliar.plot_data_analise(distancia_passe, y, x_label="Distância passe")
analise_data_auxiliar.plot_data_analise(liberdade_marcacao_receptor, y, x_label="Liberdade marcação receptor")
analise_data_auxiliar.plot_data_analise(angulo_redirect, y, x_label="Ângulo redirect")
analise_data_auxiliar.plot_data_analise(angulo_livre_chute_receptor, y, x_label="Ângulo livre chute receptor")
analise_data_auxiliar.plot_data_analise(distancia_receptor_gol, y, x_label="Distância receptor gol")
analise_data_auxiliar.plot_data_analise(liberdade_marcacao_passador, y, x_label="Liberdade marcação passador")
analise_data_auxiliar.plot_data_analise(delta_xis, y, x_label="Delta xis")
