import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import time
from random import randint
import analise_auxiliar
import autosklearn.regression


array_passe: numpy.ndarray = numpy.concatenate([
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/ER_FORCE/ATA/*Pass.csv"),
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/KIKS/ATA/*Pass.csv"),
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/RoboCin/ATA/*Pass.csv"),
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/RoboFEI/ATA/*Pass.csv"),
    analise_auxiliar.get_array_from_pattern(
        "ROBOCUP-2021-VIRTUAL/DIVISION-B/TIGERs_Mannheim/ATA/*Pass.csv")
])

X, y = analise_auxiliar.get_x_y_passes(array_passe, 1.12)

[X_train, X_test, y_train, y_test] = train_test_split(
    X, y, test_size=.2, random_state=randint(0, 1000)
)

automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=600,
    per_run_time_limit=60,
    tmp_folder='/tmp/autosklearn_regression_example_tmp',
).fit(X_train, y_train, X_test, y_test)

print(automl.leaderboard(detailed=True))
print(automl.get_models_with_weights())
print(automl.get_params())
print(automl.show_models())
print(automl.sprint_statistics())


print("Score train: {:.3f}/100".format(100*automl.score(X_train, y_train)))
print("Score test: {:.3f}/100".format(100*automl.score(X_test, y_test)))
