import numpy as np
def linearRegCostFunction(X, y, theta, Lambda):
    """computes the
    cost of using theta as the parameter for linear regression to fit the
    data points in X and y. Returns the cost in J and the gradient in grad
    """
# Initialize some useful values

    m = y.size # number of training examples
    J = 0

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost and gradient of regularized linear 
#               regression for a particular choice of theta.
#
#               You should set J to the cost and grad to the gradient.
#
    step1 = np.power(((np.dot(X,theta))-y),2)
    step2 = sum(step1)
    J = step2/(2*m)
    step3 = np.power(theta,2)
    step4 = sum(step3)
    grad = [0,0]
    innerTerm = np.dot(X,theta)-y
    grad[0] = (1/m)*sum(innerTerm*X[:,0])
    grad[1] = (1/m)*sum(innerTerm*X[:,1])+ ((Lambda/(2*m))*step4)

# =========================================================================

    return J, grad