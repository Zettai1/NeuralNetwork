import numpy as np

class NeuralNetwork():

    def __init__(self):
        #np.random.seed(1)

        self.synaptic_weights = 2 * np.random.random((11, 1)) - 1

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def train(self, training_inputs, training_outputs, training_iterations):

        for iteration in range(training_iterations):

            output = self.think(training_inputs)
            error = training_outputs - output
            adjustments = np.dot(training_inputs.T, error * self.sigmoid_derivative(output))
            self.synaptic_weights += adjustments

    def think(self, inputs):

        inputs = inputs.astype(float)
        output = self.sigmoid(np.dot(inputs, self.synaptic_weights))

        return output


if __name__ == "__main__":

    neural_network = NeuralNetwork()

    print("Random synaptic weights: ")
    print(neural_network.synaptic_weights)

    training_inputs = np.array([[0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11], #
                                [0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.1,0.11,0.12], #
                                [0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14], #
                                [0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15], #                                
                                [0.07,0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17], #
                                [0.08,0.09,0.10,0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18], #
                                [0.11,0.12,0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.21], #
                                [0.13,0.14,0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23], #                                
                                [0.15,0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25], #
                                [0.16,0.17,0.18,0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.26], #
                                [0.19,0.20,0.21,0.22,0.23,0.24,0.25,0.26,0.27,0.28,0.29], #
                                [0.23,0.24,0.25,0.26,0.27,0.28,0.29,0.30,0.31,0.32,0.33], #                                
                                [0.27,0.28,0.29,0.30,0.31,0.32,0.33,0.34,0.35,0.36,0.37], #
                                [0.29,0.30,0.31,0.32,0.33,0.34,0.35,0.36,0.37,0.38,0.39], #
                                [0.49,0.50,0.51,0.52,0.53,0.54,0.55,0.56,0.57,0.58,0.59], #
                                [0.59,0.60,0.61,0.62,0.63,0.64,0.65,0.66,0.67,0.68,0.69], #
                                [0.69,0.70,0.71,0.72,0.73,0.74,0.75,0.76,0.77,0.78,0.79]]) #

    training_outputs = np.array([[0.12,0.13,0.15,0.16,0.18,0.19,0.22,0.24,0.26,0.27,0.30,0.34,0.38,0.40,0.60,0.70,0.80]]).T

    W = (input("training iterations (maximum: 100): "))

    neural_network.train(training_inputs, training_outputs, (int(W) *  10000))

    print("synaptic weights after training: ")
    print(neural_network.synaptic_weights)

    a = int(input("input 1: "))
    b = int(input("input 2: "))
    c = int(input("input 3: "))
    d = int(input("input 4: "))
    e = int(input("input 5: "))
    f = int(input("input 6: "))
    g = int(input("input 7: "))
    h = int(input("input 8: "))
    i = int(input("input 9: "))
    j = int(input("input 10: "))
    k = int(input("input 11: "))
    A = str(a/100)
    B = str(b/100)
    C = str(c/100)
    D = str(d/100)
    E = str(e/100)
    F = str(f/100)
    G = str(g/100)
    H = str(h/100)
    I = str(i/100)
    J = str(j/100)
    K = str(k/100)

    print("New situation: input data =  ", a, b, c, d, e, f, g, h, i, j, k)
    print("Output data: ")
    print((neural_network.think(np.array([A, B, C, D, E, F, G, H, I, J, K]))) * 100)
    print("╔═╗")
    print("╚═╝")
