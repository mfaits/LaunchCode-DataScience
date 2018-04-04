import numpy as np


def normalEqn(X,y):
    """ Computes the closed-form solution to linear regression
       normalEqn(X,y) computes the closed-form solution to linear
       regression using the normal equations.
    """
    theta = np.zeros(np.size(X,1))
# ====================== YOUR CODE HERE ======================
# Instructions: Complete the code to compute the closed form solution
#               to linear regression and put the result in theta.
#

# ---------------------- Sample Solution ----------------------
    Xdot = np.dot(X.T,X)
    InvXdot = np.linalg.inv(Xdot)
    XdotY = np.dot(X.T,y)
    theta = np.dot(InvXdot,XdotY)
    
# -------------------------------------------------------------

    return theta

# ============================================================

