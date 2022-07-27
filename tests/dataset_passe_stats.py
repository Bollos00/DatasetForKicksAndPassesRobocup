import numpy
import analise_auxiliar
from matplotlib import pyplot

pyplot.switch_backend('Qt5Agg')
# pyplot.style.use('dark_background')

er_force_data = analise_auxiliar.get_array_from_pattern(
    "ROBOCUP-2021-VIRTUAL/DIVISION-B/ER_FORCE/ATA/*Pass.csv")

kiks_data = analise_auxiliar.get_array_from_pattern(
    "ROBOCUP-2021-VIRTUAL/DIVISION-B/KIKS/ATA/*Pass.csv")

robocin_data = analise_auxiliar.get_array_from_pattern(
    "ROBOCUP-2021-VIRTUAL/DIVISION-B/RoboCin/ATA/*Pass.csv")

robofei_data = analise_auxiliar.get_array_from_pattern(
    "ROBOCUP-2021-VIRTUAL/DIVISION-B/RoboFEI/ATA/*Pass.csv")

tigers_data = analise_auxiliar.get_array_from_pattern(
    "ROBOCUP-2021-VIRTUAL/DIVISION-B/TIGERs_Mannheim/ATA/*Pass.csv")

_, er_force_y = analise_auxiliar.get_x_y_passes(er_force_data, 1.12)
_, kiks_y = analise_auxiliar.get_x_y_passes(kiks_data, 1.12)
_, robocin_y = analise_auxiliar.get_x_y_passes(robocin_data, 1.12)
_, robofei_y = analise_auxiliar.get_x_y_passes(robofei_data, 1.12)
_, tigers_y = analise_auxiliar.get_x_y_passes(tigers_data, 1.12)

er_force_count = [
    numpy.count_nonzero(er_force_y == 0),
    numpy.count_nonzero(er_force_y == 50),
    numpy.count_nonzero(er_force_y == 100),
    numpy.count_nonzero(er_force_y == 150),
    numpy.count_nonzero(er_force_y == 200),
    numpy.count_nonzero(er_force_y == 250),
]

kiks_count = [
    numpy.count_nonzero(kiks_y == 0),
    numpy.count_nonzero(kiks_y == 50),
    numpy.count_nonzero(kiks_y == 100),
    numpy.count_nonzero(kiks_y == 150),
    numpy.count_nonzero(kiks_y == 200),
    numpy.count_nonzero(kiks_y == 250),
]

robocin_count = [
    numpy.count_nonzero(robocin_y == 0),
    numpy.count_nonzero(robocin_y == 50),
    numpy.count_nonzero(robocin_y == 100),
    numpy.count_nonzero(robocin_y == 150),
    numpy.count_nonzero(robocin_y == 200),
    numpy.count_nonzero(robocin_y == 250),
]

robofei_count = [
    numpy.count_nonzero(robofei_y == 0),
    numpy.count_nonzero(robofei_y == 50),
    numpy.count_nonzero(robofei_y == 100),
    numpy.count_nonzero(robofei_y == 150),
    numpy.count_nonzero(robofei_y == 200),
    numpy.count_nonzero(robofei_y == 250),
]

tigers_count = [
    numpy.count_nonzero(tigers_y == 0),
    numpy.count_nonzero(tigers_y == 50),
    numpy.count_nonzero(tigers_y == 100),
    numpy.count_nonzero(tigers_y == 150),
    numpy.count_nonzero(tigers_y == 200),
    numpy.count_nonzero(tigers_y == 250),
]

total_count = [er_force_count, kiks_count, robocin_count, robofei_count, tigers_count]
total_count = [sum(x) for x in zip(*total_count)]

labels = ['0', '50', '100', '150', '200', '250']

barWidth = 0.12
fig = pyplot.subplots(figsize =(30, 24))

br1 = numpy.arange(len(labels))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]

pyplot.bar(
    br1, er_force_count, color='r', width=barWidth,
    edgecolor='grey', label='ER-Force'
)
pyplot.bar(
    br2, kiks_count, color='g', width=barWidth,
    edgecolor='grey', label='KIKS'
)
pyplot.bar(
    br3, robofei_count, color='b', width=barWidth,
    edgecolor='grey', label='RoboFEI'
)
pyplot.bar(
    br4, tigers_count, color='m', width=barWidth,
    edgecolor='grey', label='TIGERs'
)
pyplot.bar(
    br5, robocin_count, color='c', width=barWidth,
    edgecolor='grey', label='RobôCIn'
)

# Adding Xticks
pyplot.xlabel('Avaliação da jogada')
pyplot.ylabel('Quantidade de amostras')
pyplot.xticks([r + 3*barWidth/2 for r in range(len(labels))], labels)
 
pyplot.legend()
pyplot.show()