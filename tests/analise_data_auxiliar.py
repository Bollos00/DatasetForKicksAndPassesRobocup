import analise_auxiliar
import numpy
from matplotlib import pyplot


def plot_data_analise(array_x: numpy.ndarray,
                      array_y: numpy.ndarray,
                      maximum: numpy.float64 = 250,
                      x_label: str = "???",
                      poly_degree=5):

    step = maximum/5

    if(maximum < 100):
        array_x = array_x*100/maximum
        maximum = numpy.max(array_x)
        step = maximum/5

    z = numpy.polyfit(array_x, array_y, poly_degree)
    p = numpy.poly1d(z)

    pyplot.plot(range(numpy.int64(step/2), numpy.int64(maximum+1-step/2)),
                p(range(numpy.int64(step/2), numpy.int64(maximum+1-step/2))),
                "r--")

    if(step > 1000):
        pyplot.xlabel(x_label)
        pyplot.ylabel("Avaliação")
        pyplot.axis([0, maximum+1, 0, 251])
        pyplot.show()
        return

    pesos: numpy.ndarray = numpy.full((5, 5), 0, dtype=numpy.uint16)

    for i in range(5):
        for j in range(5):
            generator = (
                a for a, b in zip(array_x, array_y) if (
                    numpy.int64(a) in range(numpy.int64(i*step), numpy.int64((i+1)*step + 1)))
                and (numpy.int64(b) in range(numpy.int64(j*50), numpy.int64((j+1)*50 + 1))
                     )
                )
            pesos[i, j] = sum(1 for k in generator)
    print(pesos, '\n')
    for i in range(5):
        for j in range(5):
            pyplot.scatter(step/2 + step*i,
                           25 + 50*j,
                           s=pesos[i, j]*step,
                           c="#00ff00")

    pyplot.xlabel(x_label)
    pyplot.ylabel("Avaliação")
    pyplot.axis([0, maximum+1, 0, 251])

    pyplot.show()
