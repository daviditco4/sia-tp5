from layer import Layer
import numpy as np


class MultiLayerPerceptron:
    def __init__(self, perceptrons_for_layers, activation_function):
        """
        perceptrons_for_layers: Un arreglo cuyo primer elemento es la cantidad de datos iniciales, y consecuentemente la cantidad de perceptrones por capa
        activation_function: The activation function to execute in the layers
        """

        # Genero los layers
        layers = []
        for index in range(1, len(perceptrons_for_layers)):
            layers.append(Layer(perceptrons_for_layers[index], perceptrons_for_layers[index - 1], activation_function))

        self.layers = layers
        self.layers_count = len(perceptrons_for_layers)

    @staticmethod
    def from_weight_list(perceptrons_for_layer, activatiton_function, weights_list):
        # Los pesos son una matriz que viene por capa ascendente
        ans = MultiLayerPerceptron(perceptrons_for_layers=perceptrons_for_layer,
                                   activation_function=activatiton_function)
        for (idx, layer) in enumerate(ans.layers):
            #         Para cada capa, necesito inicializar los pesos
            layer.set_perceptron_weights(weights_list[idx])
        return ans

    def get_all_weights(self):
        return [layer.get_perceptrons_weights_with_bias() for layer in self.layers]

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    # x: input de entrada
    # n: tasa de aprendizaje (eta)
    # error: diferencias entre lo esperado y lo calculado para cada neurona de salida
    def backward_2(self, error, x, n, gradients):
        activation_diff = self.layers[len(self.layers) - 1].get_perceptrons_activation_diff()

        # delta = []
        # for index in range(len(activation_diff)):
        #    delta.append(activation_diff[index]*error[index])
        # Calculamos el d_i para cada neurona de salida

        # d = []
        # for index in range(len(activation_diff)):
        #    d_aux = activation_diff[index] * error[index]
        #    d.append(d_aux)

        # d = None
        # if gradients is None:
        #     activation_diff_diagonal = np.diag(activation_diff)
        #     d = np.matmul(error, activation_diff_diagonal)
        # else:
        activation_diff_diagonal = np.diag(activation_diff)
        d = np.matmul(gradients, activation_diff_diagonal)
        d_n = n * d
        # d_n = []
        # for d_i in d:
        #    d_n.append(d_i * n)

        # Caso particular: Perceptron simple

        delta_w = []
        # if len(self.layers) == 1:
        #     x = np.insert(x, 0, 1)
        #     delta_w = np.matmul(d_n, x)
        #     # for j in range(len(d_n)):
        #     #    delta_w.append([])
        #     #    for k in range(len(x)):
        #     #        delta_w_aux = d_n[j] * x[k]
        #     #        delta_w[j].append(delta_w_aux)
        #     return [delta_w]

        # Obtengo los valores de salida de la anteultima capa
        values = self.layers[len(self.layers) - 2].get_perceptrons_activation()

        # TODO: esto esta mal, revisar para el caso de que la ultima sea 1
        # for j in range(len(d_n)):
        #    delta_w.append([])
        #    for k in range(len(values)):
        #        delta_w[j].append(d_n[j] * values[k])

        delta_w = np.matmul(np.split(d_n, len(d_n)), np.split(values, 1))

        # print(delta_w)
        # delta_w = np.array(delta_w)
        final_delta_w = [delta_w]
        # final_delta_w = []

        # Obtengo los W antes de actualizar
        # olds_w = self.layers[len(self.layers)-1].get_perceptrons_weights()
        olds_w = self.layers[len(self.layers) - 1].get_perceptron_weights_transposed()

        # Ahora hacemos desde mlp el add delta_weights
        # self.layers[len(self.layers)-1].add_perceptrons_delta_weights(delta_w)

        # Itero entre todas las capas
        for index in range(len(self.layers) - 2, -1, -1):
            if index != 0:
                values = self.layers[index - 1].get_perceptrons_activation()
            else:
                # Si estamos en la capa inferior de la red, utilizamos los datos de entrada
                values = np.insert(x, 0, 1)
            # Calculamos el delta_w y los d_i de los nodos de la capa actual
            delta_w, d = self.layers[index].backward(d, olds_w, n, values)
            # Nos quedamos con una copia de los pesos viejos para la proxima capa
            olds_w = self.layers[index].get_perceptron_weights_transposed()
            # olds_w = self.layers[index].get_perceptrons_weights()

            final_delta_w.append(delta_w)

            # Ahora hacemos desde mlp el add delta_weights
            # self.layers[len(self.layers)-1].add_perceptrons_delta_weights(delta_w)
            # Actualizamos los pesos de las neuronas de la capa actual
            # self.layers[index].add_perceptrons_delta_weights(delta_w)

        # retorno una lista de matrices con los delta_w en tipo np para la suma
        return final_delta_w

    def backward(self, error, x, n, gradients=None):
        # Obtenemos el theta'(h) de la capa de salida
        activation_diff = self.layers[len(self.layers) - 1].get_perceptrons_activation_diff()

        # delta = []
        # for index in range(len(activation_diff)):
        #    delta.append(activation_diff[index]*error[index])
        # Calculamos el d_i para cada neurona de salida

        # d = []
        # for index in range(len(activation_diff)):
        #    d_aux = activation_diff[index] * error[index]
        #    d.append(d_aux)

        d = None
        if gradients is None:
            activation_diff_diagonal = np.diag(activation_diff)
            d = np.matmul(error, activation_diff_diagonal)
        else:
            activation_diff_diagonal = np.diag(activation_diff)
            d = np.matmul(gradients, activation_diff_diagonal)
        d_n = n * d
        # d_n = []
        # for d_i in d:
        #    d_n.append(d_i * n)

        # Caso particular: Perceptron simple

        delta_w = []
        if len(self.layers) == 1:
            x = np.insert(x, 0, 1)
            delta_w = np.matmul(d_n, x)
            # for j in range(len(d_n)):
            #    delta_w.append([])
            #    for k in range(len(x)):
            #        delta_w_aux = d_n[j] * x[k]
            #        delta_w[j].append(delta_w_aux)
            return [delta_w]

        # Obtengo los valores de salida de la anteultima capa
        values = self.layers[len(self.layers) - 2].get_perceptrons_activation()

        # TODO: esto esta mal, revisar para el caso de que la ultima sea 1
        # for j in range(len(d_n)):
        #    delta_w.append([])
        #    for k in range(len(values)):
        #        delta_w[j].append(d_n[j] * values[k])

        delta_w = np.matmul(np.split(d_n, len(d_n)), np.split(values, 1))

        # print(delta_w)
        delta_w = np.array(delta_w)
        final_delta_w = [delta_w]

        # Obtengo los W antes de actualizar
        # olds_w = self.layers[len(self.layers)-1].get_perceptrons_weights()
        olds_w = self.layers[len(self.layers) - 1].get_perceptron_weights_transposed()

        # Ahora hacemos desde mlp el add delta_weights
        # self.layers[len(self.layers)-1].add_perceptrons_delta_weights(delta_w)

        # Itero entre todas las capas
        for index in range(len(self.layers) - 2, -1, -1):
            if index != 0:
                values = self.layers[index - 1].get_perceptrons_activation()
            else:
                # Si estamos en la capa inferior de la red, utilizamos los datos de entrada
                values = np.insert(x, 0, 1)
            # Calculamos el delta_w y los d_i de los nodos de la capa actual
            delta_w, d = self.layers[index].backward(d, olds_w, n, values)
            # Nos quedamos con una copia de los pesos viejos para la proxima capa
            olds_w = self.layers[index].get_perceptron_weights_transposed()
            # olds_w = self.layers[index].get_perceptrons_weights()

            final_delta_w.append(delta_w)

            # Ahora hacemos desde mlp el add delta_weights
            # self.layers[len(self.layers)-1].add_perceptrons_delta_weights(delta_w)
            # Actualizamos los pesos de las neuronas de la capa actual
            # self.layers[index].add_perceptrons_delta_weights(delta_w)

        # retorno una lista de matrices con los delta_w en tipo np para la suma
        return final_delta_w, np.matmul(olds_w, d)

    def apply_delta_w(self, final_delta_w):
        for index, layer in enumerate(self.layers):
            layer.add_perceptrons_delta_weights(final_delta_w[len(self.layers) - index - 1])









