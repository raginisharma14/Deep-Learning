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


import numpy as np
from numpy.linalg import inv
from numpy.linalg import pinv

class regressor(object):
    """
    This is a sample class for miniproject 1.

    Args:
        data: Is a tuple, ``(x,y)``
              ``x`` is a two or one dimensional ndarray ordered such that axis 0 is independent 
              data and data is spread along axis 1. If the array had only one dimension, it implies
              that data is 1D.
              ``y`` is a 1D ndarray it will be of the same length as axis 0 or x.   
                          
    """
    def gradient_descent_calculation(self,theta,alpha,noOfIterations):
     
        sample_size = self.x.shape[0]
        actual_value = (self.y)
        count =0
        lamda = 5
        
        while(count <= noOfIterations):
            
               
                predicted_value = np.dot(self.x, theta)
                costfunction =np.sum( (predicted_value - actual_value)** 2) / (2 * sample_size)
                print "costfunction", costfunction, "Iteration", count
                gradientvalue = (np.dot(self.x.T, (predicted_value - actual_value) ))/sample_size
                theta = (theta*(1- (alpha*lamda)/sample_size)) - alpha*gradientvalue
                count =count + 1
       
        return theta
    
    def __init__(self, data):
        
        self.x, self.y = data
       
        # Here is where your training and all the other magic should happen. 
        # Once trained you should have these parameters with ready.
        noOfIterations = 10000
        alpha = 0.005
        self.x = np.concatenate((np.ones((self.x.shape[0],1)), self.x), axis = 1)
        theta = np.ones((self.x.shape[1],1))
        result = self.gradient_descent_calculation(theta, alpha, noOfIterations)                   
        self.w = result[1:]
        self.b = result[:1]
       
      
  
        
    def get_params (self):
        """ 
        Method that should return the model parameters.

        Returns:
            tuple of numpy.ndarray: (w, b). 

        Notes:
            This code will return a random numpy array for demonstration purposes.

        """
        
        return (self.w, self.b)

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

       
      
        result = np.dot(x, self.w) + self.b
        return result
        

if __name__ == '__main__':
    pass 
