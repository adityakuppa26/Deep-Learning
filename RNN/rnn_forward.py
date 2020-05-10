import numpy as np

def softmax(x):  
    # stable softmax, avoids under/over flows
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def rnn_forward_step(x, a_prev, params):
    Waa = params['Waa']
    Wax = params['Wax']
    Wya = params['Wya']
    ba = params['ba']
    by = params['by']
    
    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + ba)
    y = softmax(np.dot(Wya, a_next) + by)
    cache = (x, a_prev, a_next, params)
    
    return y, a_next, cache

def rnn_forward(X, a0, params):
    # X shape : [n_X, m, T], where n_X - no. of units in input, m - # exmps, T - time step
    # a shape : [n_a, m , T]
    # a0 shape: [n_a, m]
    
    caches =[]
    cache = None
    n_X, m, T = X.shape
    n_Y, n_a = params['Wya'].shape
    
    a  = np.zeros((n_a, m, T))
    Y = np.zeros((n_Y, m , T))
    a_prev = a0
    
    for t in range(T):
        y, a_prev, cache = rnn_forward_step(X[:, :, t], a_prev, params)
        Y[:, :, t] = y
        a[:, :, t] = a_prev
        caches.append(cache)
    
    return Y, a, caches
