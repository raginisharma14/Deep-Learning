"""
MIT License
Copyright (c) 2017 Ragini Sharma
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import numpy
import math 
import random
class xor_net(object):
    
    """
    This is a sample class for miniproject 1.

    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent 
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.   
                          
    """
    """
        This method is used to build the neural network
        Args:
            param1: takes no of input nodes
            param2: takes no of hidden nodes in hidden layer1.
            param3: takes no of hidden nodes in hidden layer2.
            param4: takes no of output nodes.
        Returns: Built network

    """
    def build_network(self, n_inputs, n_hidden1, n_hidden2, n_outputs):
        network = list()
        
       
        hidden_layer1 = [{'weights':[random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden1)]
        network.append(hidden_layer1)
        hidden_layer2 = [{'weights':[random.random() for i in range(n_hidden1 + 1)]} for i in range(n_hidden2)]
        network.append(hidden_layer2)
        output_layer = [{'weights':[random.random() for i in range(n_hidden2 + 1)]} for i in range(n_outputs)]
        network.append(output_layer)
        return network
    """
        This method calculates the activation of each neuron.
        Args:
            param1: takes the weights and inputs of each neuron in each layer of the network
            
        Returns: activated Value.

    """
    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            
            activation += weights[i] * inputs[i]
        return activation
    """
        This method is the sigmoid function
        Args:
            param1: takes the activation value 
            
        Returns: Sigmoid value of the activation. 

    """
    def sigmoid(self, activation):
        return 1.0 / (1.0 + math.exp(-activation))
    """
        This method is used to do the forward propogation of the neural network
        Args:
            param1: takes the built network
            param2: takes each sample of the dataset.
        Returns: Predicted Value.

    """
    def forward_propagation(self,network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for node in layer:
                activation = self.activate(node['weights'], inputs)
                node['output'] = self.sigmoid(activation)
                new_inputs.append(node['output'])
            inputs = new_inputs
        return inputs

    def sigmoid_derivative(self,output):
        return output * (1.0 - output)

    """
        This method does the backpropogation. Calculate the gradients for each sample
        and modify the weights.
        Args:
             Param1: Takes network
             Param2: Takes the expected values
        

    """
    def back_propagation(self,network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i != len(network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for node in network[i + 1]:
                        error += (node['weights'][j] * node['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    node = layer[j]
                    errors.append(expected[j] - node['output'])
            for j in range(len(layer)):
                node = layer[j]
                node['delta'] = errors[j] * self.sigmoid_derivative(node['output'])

    def update_weights(self,network, row, l_rate):
        for i in range(len(network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [node['output'] for node in network[i - 1]]
            for node in network[i]:
                for j in range(len(inputs)):
                    node['weights'][j] += l_rate * node['delta'] * inputs[j]
                node['weights'][-1] += l_rate * node['delta']
    """
        This method is used to train the neural network
        Args:
            param1:  takes the network
            param2:  takes the dataset.
            param3:  takes the learning rate
            param4:  No of epochs
            param5:  No Of output nodes.

    """
	"""
		Here the neural network uses stochastic gradient descent and not batch gradient descent as we are looping through one sample at a time.
		Each iteration has one sample. Loss is calculated for that particular sample, and then we are updating weights in the same iteration itself.
	"""
    def train_network(self,network, train, l_rate, n_epoch, n_outputs):
        for epoch in range(n_epoch):
            sum_error = 0
            for row in train:
                outputs = self.forward_propagation(network, row)
                expected = [0 for i in range(n_outputs)]
                
                expected[int(row[-1])] = 1
              
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                self.back_propagation(network, expected)
                self.update_weights(network, row, l_rate)
            print('epoch=%d, learningrate=%.3f, loss=%.3f' % (epoch, l_rate, sum_error))
            self.network = network
            
    def __init__(self, data, labels):
        
        rows, columns = data.shape
        labels = numpy.array([labels], dtype=int)
        a = numpy.concatenate((data, labels.T), axis =1)
      
        n_inputs = len(a[0]) - 1
        
        n_outputs = len(set([row[-1] for row in a]))
        # build network with 2 hidden layers and 10 nodes in each hidden layer
        network = self.build_network(n_inputs, 10,10, n_outputs)
        
        self.train_network(network, a, 0.5, 100, n_outputs)
        for layer in network:
            print(layer)
        
    def get_params (self):
        self.params = []  # [(w,b),(w,b)]         
        """ 
        Method that should return the model parameters.
        self.params = []  # [(w,b),(w,b)]         

        Returns:
            tuple of numpy.ndarray: (w, b). 

        Notes:
            This code will return an empty list for demonstration purposes. A list of tuples of 
            weoghts and bias for each layer. Ordering should from input to outputt

        """
        return self.params

    def get_predictions (self, x):
        """
        Method should return the outputs given unseen data

        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.

        Returns:    
            numpy.ndarray: ``y`` which is a 1D array of predictions of the same length as axis 0 of 
                            ``x`` 
        Notes:
            Temporarily returns random numpy array for demonstration purposes.                            
        """        
        # Here is where you write a code to evaluate the data and produce predictions.
        a, b = x.shape
        
        rows, columns = x.shape
        labels = numpy.zeros(shape=(rows))
        labels = numpy.array([labels])
        a = numpy.concatenate((x, labels.T), axis =1)
        rows,columns = a.shape
       
        result = numpy.zeros(shape= rows)
        i=0
        for row in a:
            outputs = self.forward_propagation(self.network, row)
            result[i]= outputs.index(max(outputs))
            i =i+1
        return result

class mlnn(xor_net):
    """
        Build neural network with 6 hidden layers and different nodes in each layer.
    """
    
    def initialize_network(self, n_inputs, n_hidden1, n_hidden2, n_hidden3, n_hidden4, n_hidden5, n_hidden6, n_outputs):
        network = list()
       
       
        hidden_layer1 = [{'weights':[random.random() for i in range(n_inputs + 1)]} for i in range(n_hidden1)]
        network.append(hidden_layer1)
        hidden_layer2 = [{'weights':[random.random() for i in range(n_hidden1 + 1)]} for i in range(n_hidden2)]
        network.append(hidden_layer2)
        hidden_layer3 = [{'weights':[random.random() for i in range(n_hidden2 + 1)]} for i in range(n_hidden3)]
        network.append(hidden_layer3)
        hidden_layer4 = [{'weights':[random.random() for i in range(n_hidden3 + 1)]} for i in range(n_hidden4)]
        network.append(hidden_layer4)
        hidden_layer5 = [{'weights':[random.random() for i in range(n_hidden4 + 1)]} for i in range(n_hidden5)]
        network.append(hidden_layer5)
        hidden_layer6 = [{'weights':[random.random() for i in range(n_hidden5 + 1)]} for i in range(n_hidden6)]
        network.append(hidden_layer6)
        output_layer = [{'weights':[random.random() for i in range(n_hidden6 + 1)]} for i in range(n_outputs)]
        network.append(output_layer)
        return network

    """
        This method calculates the activation of each neuron.
        Args:
            param1: takes the weights and inputs of each neuron in each layer of the network
            
        Returns: activated Value.

    """
    def activate(self, weights, inputs):
        activation = weights[-1]
        for i in range(len(weights)-1):
            
            activation += weights[i] * inputs[i]
        return activation
    """
        This method is the sigmoid function
        Args:
            param1: takes the activation value 
            
        Returns: Sigmoid value of the activation. 

    """
    def sigmoid(self, activation):
        return 1.0 / (1.0 + math.exp(-activation))
    """
        This method is used to do the forward propogation of the neural network
        Args:
            param1: takes the built network
            param2: takes each sample of the dataset.
        Returns: Predicted Value.

    """
    def forward_propagation(self,network, row):
        inputs = row
        for layer in network:
            new_inputs = []
            for node in layer:
                activation = self.activate(node['weights'], inputs)
                node['output'] = self.sigmoid(activation)
                new_inputs.append(node['output'])
            inputs = new_inputs
        return inputs

    def sigmoid_derivative(self,output):
        return output * (1.0 - output)

    """
        This method does the backpropogation. Calculate the gradients for each sample
        and modify the weights.
        Args:
             Param1: Takes network
             Param2: Takes the expected values
        

    """
    def back_propagation(self,network, expected):
        for i in reversed(range(len(network))):
            layer = network[i]
            errors = list()
            if i != len(network)-1:
                for j in range(len(layer)):
                    error = 0.0
                    for node in network[i + 1]:
                        error += (node['weights'][j] * node['delta'])
                    errors.append(error)
            else:
                for j in range(len(layer)):
                    node = layer[j]
                    errors.append(expected[j] - node['output'])
            for j in range(len(layer)):
                node = layer[j]
                node['delta'] = errors[j] * self.sigmoid_derivative(node['output'])

    def update_weights(self,network, row, l_rate):
        for i in range(len(network)):
            inputs = row[:-1]
            if i != 0:
                inputs = [node['output'] for node in network[i - 1]]
            for node in network[i]:
                for j in range(len(inputs)):
                    node['weights'][j] += l_rate * node['delta'] * inputs[j]
                node['weights'][-1] += l_rate * node['delta']
    """
        This method is used to train the neural network
        Args:
            param1:  takes the network
            param2:  takes the dataset.
            param3:  takes the learning rate
            param4:  No of epochs
            param5:  No Of output nodes.

    """
    def train_network(self,network, train, l_rate, n_epoch, n_outputs):
        for epoch in range(n_epoch):
            sum_error = 0
            for row in train:
                outputs = self.forward_propagation(network, row)
                expected = [0 for i in range(n_outputs)]
                
                expected[int(row[-1])] = 1
              
                sum_error += sum([(expected[i]-outputs[i])**2 for i in range(len(expected))])
                self.back_propagation(network, expected)
                self.update_weights(network, row, l_rate)
            print('epoch=%d, learningrate=%.3f, loss=%.3f' % (epoch, l_rate, sum_error))
            self.network = network
            

    def __init__(self, data, labels):
        
        rows, columns = data.shape
        print ("columns", columns)
        print(data)
        print(labels)
        labels = numpy.array([labels], dtype=int)
        a = numpy.concatenate((data, labels.T), axis =1)
      
        n_inputs = len(a[0]) - 1
        
        n_outputs = len(set([row[-1] for row in a]))
        network = self.initialize_network(n_inputs, 20, 20, 20, 10,10,5,n_outputs)
        
        self.train_network(network, a, 0.2, 100, n_outputs)
        for layer in network:
            print(layer)
        
    def get_params (self):
        self.params = []  # [(w,b),(w,b)]         
        """ 
        Method that should return the model parameters.
        self.params = []  # [(w,b),(w,b)]         

        Returns:
            tuple of numpy.ndarray: (w, b). 

        Notes:
            This code will return an empty list for demonstration purposes. A list of tuples of 
            weoghts and bias for each layer. Ordering should from input to outputt

        """
        return self.params

    def get_predictions (self, x):
        """
        Method should return the outputs given unseen data

        Args:
            x: array similar to ``x`` in ``data``. Might be of different size.

        Returns:    
            numpy.ndarray: ``y`` which is a 1D array of predictions of the same length as axis 0 of 
                            ``x`` 
        Notes:
            Temporarily returns random numpy array for demonstration purposes.                            
        """        
        # Here is where you write a code to evaluate the data and produce predictions.
        a, b = x.shape
        
        rows, columns = x.shape
        labels = numpy.zeros(shape=(rows))
        labels = numpy.array([labels])
        a = numpy.concatenate((x, labels.T), axis =1)
        rows,columns = a.shape
       
        result = numpy.zeros(shape= rows)
        i=0
        for row in a:
            outputs = self.forward_propagation(self.network, row)
            result[i]= outputs.index(max(outputs))
            i =i+1
        return result


if __name__ == '__main__':
    pass 
