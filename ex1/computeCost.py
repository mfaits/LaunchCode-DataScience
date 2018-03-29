import numpy as np

def computeCost(X, y, theta):
    """
       computes the cost of using theta as the parameter for linear 
       regression to fit the data points in X and y
    """
    m = y.size
    J = 0

# ====================== YOUR CODE HERE ======================
# Instructions: Compute the cost of a particular choice of theta
#               You should set J to the cost.
    step1 = np.power(((np.dot(X,theta))-y),2)
    step2 = sum(step1)
    J = step2/(2*m)

# =========================================================================

    return J


