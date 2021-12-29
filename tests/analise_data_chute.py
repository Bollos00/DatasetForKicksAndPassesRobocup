import numpy
import analise_auxiliar
import analise_data_auxiliar
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


array_chute: numpy.ndarray = numpy.concatenate([
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/ER_FORCE/ATA/*Shoot.csv"),
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/KIKS/ATA/*Shoot.csv"),
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/RoboFEI/ATA/*Shoot.csv"),
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/TIGERs_Mannheim/ATA/*Shoot.csv"),
    # analise_auxiliar.get_array_from_pattern(
    #     "ROBOCUP-2021-VIRTUAL/DIVISION-B/RoboCin/ATA/*Shoot.csv")
])

X, y = analise_auxiliar.get_x_y_shoots(array_chute, 1.12)

# array_chute: numpy.ndarray = numpy.concatenate([
#     analise_auxiliar.get_array_from_pattern("ROBOCUP-2019/ER_FORCE/ATA/*Chute.csv"),
#     analise_auxiliar.get_array_from_pattern("ROBOCUP-2019/ZJUNlict/ATA/*Chute.csv")
# ])
# X, y = analise_auxiliar.get_x_y_shoots(array_chute)


# print(X.shape)
# print(X)
# print("=================")
# poly: PolynomialFeatures = PolynomialFeatures(
#     degree=3,
#     interaction_only=False,
#     include_bias=False,
#     order='C'
# )
# new_X = poly.fit_transform(X)
# print("New dimension: {0}".format(new_X.shape[1]))
# # for i in range(new_X.shape[1]):
# #     analise_data_auxiliar.plot_data_analise(new_X[:, i],
# #                                             y,
# #                                             numpy.max(new_X[:, i])/5)
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

angulo_livre_caminho: numpy.ndarray = X[:, 0]  # X is 1 and Y is 0
distancia_bola: numpy.ndarray = X[:, 1]
liberdade_marcacao: numpy.ndarray = X[:, 2]

print(angulo_livre_caminho.shape)

y = y * .98

degree = 1
analise_data_auxiliar.plot_data_analise(
    angulo_livre_caminho, y,
    x_label="Ângulo livre do caminho da bola até o gol adversário", poly_degree=degree)
analise_data_auxiliar.plot_data_analise(
    distancia_bola, y,
    x_label="Distância da bola ao gol adversário", poly_degree=degree)
analise_data_auxiliar.plot_data_analise(
    liberdade_marcacao, y,
    x_label="Marcação do adversário", poly_degree=degree)
