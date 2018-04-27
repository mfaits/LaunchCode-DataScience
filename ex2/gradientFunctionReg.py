import numpy as np

#from gradientFunction import gradientFunction
from sigmoid import sigmoid

def gradientFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    m = len(y)   # number of training examples
    FlatY = y.values.flatten()
    #X = X.as_matrix()
# ====================== YOUR CODE HERE ======================
# Instructions: Compute the gradient of a particular choice of theta.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta


# =============================================================
    
    grad =  np.zeros(np.size(theta,0))
    
    innerterm = sigmoid(np.dot(X,np.transpose(theta)))-FlatY

    
    for i in range(np.size(theta,0)):
        term = np.multiply(innerterm, X[:,i])
        
        
        if (i==0):
                grad[i] = np.sum(term)/m
                
        else:    
                grad[i] = (np.sum(term)/m) + ((Lambda / m)*theta[i])  
        
        
    return grad