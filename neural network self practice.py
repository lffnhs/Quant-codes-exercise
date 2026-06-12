import numpy as np
import matplotlib.pyplot as plt

def activation_func(x):
    #sigmoid function
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    fx = activation_func(x)
    return fx * (1 - fx)

def loss_func(y_pred, y_act):
    return ((y_act - y_pred) ** 2).mean()

#this network has one hidden layer and one output layer with one output
#n_neuron is the number of neurons the network wants to have, neurons is an array of neurons with weights and biases
class simple_neural_network():
    def __init__(self, n_neuron, input):
        self.neurons = []
        self.output = []
        n = len(input[0]) 
        for _ in range(n_neuron): #neuron + output
            neuron = np.array([])
            for _ in range(n + 1):
               neuron = np.append(neuron, np.random.normal())
            self.neurons.append(neuron)
        for _ in range(n_neuron + 1):
            self.output = np.append(self.output, np.random.normal())
        self.output = np.array(self.output)
        self.neurons = np.array(self.neurons)
    
    #use to predict the outcomes of new datapoint
    def predict(self, data):
        out = []
        for i in range(len(self.neurons)):
            temp = activation_func(np.dot(self.neurons[i][:-1], data)+self.neurons[i][-1])
            out.append(temp)
        out = np.array(out)    
        y_pred = np.dot(out, self.output[:-1]) + self.output[-1]
        return activation_func(y_pred)
    
    #use in training the data               
    def feedforward(self, data):
        out = []
        y_pred = []
        f_out = []
        for d in range(len(data)):
            first_iteration = []
            f_first_iteration = []
            for i in range(len(self.neurons)):
                temp = np.dot(self.neurons[i][:-1], data[d])+self.neurons[i][-1]
                first_iteration.append(temp)
                f_first_iteration.append(activation_func(temp))
            out.append(first_iteration)
            f_out.append(f_first_iteration)
        out = np.array(out)
        f_out = np.array(f_out)
        for i in f_out:
            y_pred.append(np.dot(i, self.output[:-1]) + self.output[-1]) #before activation function
        return out, f_out, np.array(y_pred)
    
    def train(self, data, y_true):
        learn_rate = 0.1
        epochs = 1000
        loss = []
        partial_deriv = []
        
        for epoch in range(epochs):
            out, f_out, y_pred = self.feedforward(data)
            for i in range(len(data)):
                dL_dypred = -2 * (y_true[i] - activation_func(y_pred[i]))
                
                partial_deriv_temp = []
                partial_deriv = []
                
                for o in range(len(self.output) - 1):
                    partial_deriv_temp.append(dL_dypred * self.output[o] * deriv_sigmoid(y_pred[i]))
                
                for neurons in range(len(self.neurons)):
                    neuron = []
                    for parameters in range(len(data[0])):
                        neuron.append(partial_deriv_temp[neurons] * data[i][parameters] * deriv_sigmoid(out[i][neurons]))
                    neuron.append(deriv_sigmoid(out[i][neurons]))
                    partial_deriv.append(np.array(neuron))
                
                partial_d_out = []
                for output in range(len(self.neurons)):
                    partial_d_out.append(dL_dypred * f_out[i][output] * deriv_sigmoid(y_pred[i]))
                partial_d_out.append(dL_dypred * deriv_sigmoid(y_pred[i]))
                
                
                self.neurons -= learn_rate * np.array(partial_deriv)
                self.output-= learn_rate * np.array(partial_d_out)
                
            loss.append(loss_func(np.array(list(map(activation_func, y_pred))), y_true))
        
        
        plt.plot(loss)
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.show()
        return
    
#example training dataset. I tried to use person's height and weight to predict his/her gender. The training dataset is shifted by their averages.
data = np.array([

    [-9, -2],  
    [3, 0],   
    [6, 11],   
    [12, 6],
    [-7, -7], 
    [-2, -6]
])

y_trues = np.array([
1,
1,
0,
0,
1,
1,
])

network = simple_neural_network(2, data)
network.train(data, y_trues)

#using the model to predict new data
print(network.predict(np.array([10, 11])))
