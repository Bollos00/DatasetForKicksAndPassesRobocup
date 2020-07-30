import numpy
import analise_auxiliar
import analise_data_auxiliar

array_chute: numpy.ndarray = analise_auxiliar.get_array_from_pattern("ALL/*Chute.csv")

angulo_livre_caminho: numpy.ndarray = array_chute[:, [0, 1]]  # X is 1 and Y is 0
distancia_bola: numpy.ndarray = array_chute[:, [0, 2]]
liberdade_marcacao: numpy.ndarray = array_chute[:, [0, 3]]

analise_data_auxiliar.plot_data_analise(angulo_livre_caminho, "Ângulo livre caminho")
analise_data_auxiliar.plot_data_analise(distancia_bola, "Distância bola")
analise_data_auxiliar.plot_data_analise(liberdade_marcacao, "Liberdade marcação")
