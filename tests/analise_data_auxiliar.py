import analise_auxiliar
import numpy
from matplotlib import pyplot


def plot_data_analise(array_x: numpy.ndarray,
                      array_y: numpy.ndarray,
                      maximum: numpy.float64 = 270,
                      x_label: str = "???",
                      poly_degree=5):

    step = maximum/6

    z = numpy.polyfit(array_x, array_y, poly_degree)
    p = numpy.poly1d(z)

    pyplot.plot(range(int(step/2), int(maximum+1-step/2)),
                p(range(int(step/2), int(maximum+1-step/2))),
                "r--")

    # if(step > 1000):
    #     pyplot.xlabel(x_label)
    #     pyplot.ylabel("Avaliação")
    #     pyplot.axis([0, maximum+1, 0, 251])
    #     pyplot.show()
    #     return

    pesos: numpy.ndarray = numpy.full((6, 6), 0, dtype=numpy.uint16)

    for i in range(6):
        for j in range(6):
            generator = (
                1 for a, b in zip(array_x, array_y) if (
                    (a >= i*step and a < (i+1)*step)
                    and (b >= j*step and b < (j+1)*step)
                )
            )
            pesos[i, j] = sum(1 for k in generator)

    print(numpy.sum(pesos))
    print(pesos, '\n')

    for i in range(6):
        for j in range(6):
            pyplot.scatter(step/2 + step*i,
                           step/2 + step*j,
                           s=(pesos[i, j]*.5)**2,
                           c="#0000ff")

    pyplot.xlabel(x_label)
    pyplot.ylabel("Avaliação")
    pyplot.xticks([45, 90, 135, 180, 225])
    pyplot.yticks([45, 90, 135, 180, 225])
    pyplot.axis([0, maximum+1, 0, maximum+1])
    pyplot.grid(visible=True, axis='both')
    pyplot.show()
