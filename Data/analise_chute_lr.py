from glob import glob
import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot
import pickle
import time

nparray = numpy.array
pyplot.style.use('dark_background')

file_names = glob("/home/robofei/Documents/DataAnalyse/ALL/*Chute.csv")

array_chute: nparray = []


for f in file_names:
    array_chute.append(
        numpy.genfromtxt(
            f,
            dtype=int,
            delimiter=";",
            skip_header=1
        )
    )

array_chute = numpy.concatenate(array_chute)

y: nparray = array_chute[:, 0]
X: nparray = array_chute[:, [1, 2, 3]]

lr_out: LinearRegression = LinearRegression().fit(X, y)

pickle.dump(lr_out, open("models/avaliacao_chute_lr.sav", 'wb'))


x_axis: nparray = range(1, 200)
score_train: nparray = []
score_test: nparray = []

start: float = time.time()

for i in x_axis:

    [X_train, X_test, y_train, y_test] = train_test_split(X,
                                                          y,
                                                          test_size=.2,
                                                          random_state=i)

    lr: LinearRegression = LinearRegression().fit(X_train, y_train)

    score_test.append(lr.score(X_test, y_test))
    score_train.append(lr.score(X_train, y_train))

end: float = time.time()

print("Score test: ", numpy.mean(score_test))
print("Score train: ", numpy.mean(score_train))
print("Time of operation: {} ms".format(
    (end-start)*1e3/(numpy.size(x_axis)*numpy.size(y)))
      )

pyplot.plot(x_axis, score_train, 'r-', label='Train score')
pyplot.plot(x_axis, score_test, 'c-', label='Test score')
pyplot.xlabel('random_state')
pyplot.ylabel('score')
pyplot.legend(loc="upper right")
pyplot.grid()

pyplot.show()
