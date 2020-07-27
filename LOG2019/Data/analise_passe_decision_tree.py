from glob import glob
import numpy
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from matplotlib import pyplot
import joblib
import time

nparray = numpy.array
pyplot.style.use('dark_background')


file_names = glob("../ALL/*Passe.csv")

# print(file_names)

array_passe: nparray = []


for f in file_names:
    array_passe.append(
        numpy.genfromtxt(
            f,
            dtype=int,
            delimiter=";",
            skip_header=1
        )
    )

array_passe = numpy.concatenate(array_passe)

y: nparray = array_passe[:, 0]
X: nparray = array_passe[:, [1, 2, 3,  4, 4, 6, 7, 8]]

tree_out: DecisionTreeRegressor = DecisionTreeRegressor(
    criterion='mse',
    splitter='best',
    max_depth=4,
    min_samples_split=33*1e-3,
    min_samples_leaf=80*1e-3,
    min_weight_fraction_leaf=87e-3,
    max_features='auto',
    random_state=38,
    max_leaf_nodes=6,
    min_impurity_decrease=0,
    min_impurity_split=None,
    presort='deprecated',
    ccp_alpha=40
).fit(X, y)

joblib.dump(tree_out, "models/avaliacao_passe_tree.sav")

x_axis: nparray = range(1, 50, 1)
score_train: nparray = []
score_test: nparray = []

start: float = time.time()
for i in x_axis:

    [X_train, X_test, y_train, y_test] = train_test_split(
        X, y, test_size=.2, random_state=i
        )

    tree: DecisionTreeRegressor = DecisionTreeRegressor(
        criterion='mse',
        splitter='best',
        max_depth=4,
        min_samples_split=33*1e-3,
        min_samples_leaf=80*1e-3,
        min_weight_fraction_leaf=87e-3,
        max_features='auto',
        random_state=i,
        max_leaf_nodes=6,
        min_impurity_decrease=0,
        min_impurity_split=None,
        presort='deprecated',
        ccp_alpha=40
    ).fit(X_train, y_train)

    score_test.append(tree.score(X_test, y_test))
    score_train.append(tree.score(X_train, y_train))

end: float = time.time()

print("Score test: ", numpy.mean(score_test))
print("Score train: ", numpy.mean(score_train))
print("Time of operation: {} ms".format(
    (end-start)*1e3/(numpy.size(x_axis)*numpy.size(y)))
      )

pyplot.plot(x_axis, score_train, 'r-', label='Train score')
pyplot.plot(x_axis, score_test, 'c-', label='Test score')
pyplot.xlabel('???')
pyplot.ylabel('score')
pyplot.legend(loc="upper right")
pyplot.grid()

pyplot.show()
