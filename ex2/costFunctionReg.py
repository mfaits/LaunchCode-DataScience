from costFunction import costFunction
import numpy as np
from sigmoid import sigmoid

def costFunctionReg(theta, X, y, Lambda):
    """
    Compute cost and gradient for logistic regression with regularization

    computes the cost of using theta as the parameter for regularized logistic regression and the
    gradient of the cost w.r.t. to the parameters.
    """
    # Initialize some useful values
    m = len(y)   # number of training examples

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta.
#               You should set J to the cost.
#               Compute the partial derivatives and set grad to the partial
#               derivatives of the cost w.r.t. each parameter in theta

# =============================================================

    step1 = np.dot(-y.T, np.log(sigmoid(np.dot(X,np.transpose(theta)))))
    step2 = np.dot((1-y.T), np.log(1-sigmoid(np.dot(X,np.transpose(theta)))))

    UnregJ = np.sum(step1 - step2) / m
    step3 = theta[1:]**2

    J = UnregJ + ((Lambda/(2*m)*np.sum(step3)))
    return J
