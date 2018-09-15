# Deep Neural Network
  
**Initialization**
  
* The weight and bias matrices for all the layers of the neural network must be initialised randomly .  
* Done using the numpy.random.randn() in the code.  
* The dimensions of the matrices must be properly checked for each of the layers .
  
**Forward Propagation**
  
* The Z matrix is calculated for each layer in the neural net .  
* The corresponding activation matrix is produced by applying to the Z matrix ,the activation function for that layer.  
* The final layer activation matrix is taken as the output (hypothesis) of the neural network.  
* Store the Z matrix , Weight matrix , bias vector and the activation from previous layer in cache , for use during backprop.  
  
**Computing Cost**
  
* From the final layer activation obtained and the given actual output from the training data , compute the cost ( error value).  
  
**Backward Propagation**
  
* Compute the dZ for the given layer using the dA matrix obtained from the next layer.  
* Using dZ and cache compute gradients ( dW,db) and dA for the previous layer.  
  
**Update Parameters**
  
* Update the parameters (W and b) of each layer after every iteration of the gradient descent algorithm , until the cost is minimized.  
  

  


