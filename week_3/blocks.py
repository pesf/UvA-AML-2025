import numpy as np

#######################################################
# put `w2_sigmoid_forward` and `w2_sigmoid_grad_input` here #
#######################################################
def w2_sigmoid_forward(x_input):
    """sigmoid nonlinearity
    # Arguments
        x_input: np.array of size `(n_objects, n_in)`
    # Output
        the output of sigmoid layer
        np.array of size `(n_objects, n_in)`
    """
    output = 1 / (1 + np.exp(-x_input))
    return output

def w2_sigmoid_grad_input(x_input, grad_output):
    """sigmoid nonlinearity gradient.
        Calculate the partial derivative of the loss
        with respect to the input of the layer
    # Arguments
        x_input: np.array of size `(n_objects, n_in)`
        grad_output: np.array of size `(n_objects, n_in)`
            dL / df
    # Output
        the partial derivative of the loss
        with respect to the input of the function
        np.array of size `(n_objects, n_in)`
        dL / dh
    """
    sigmoid = 1 / (1 + np.exp(-x_input))
    grad_input = grad_output * sigmoid * (1 - sigmoid)  
    return grad_input


#######################################################
# put `w2_nll_forward` and `w2_nll_grad_input` here    #
#######################################################

def w2_nll_forward(target_pred, target_true):
    """Compute the value of NLL
        for a given prediction and the ground truth
    # Arguments
        target_pred: predictions - np.array of size `(n_objects, 1)`
        target_true: ground truth - np.array of size `(n_objects, 1)`
    # Output
        the value of NLL for a given prediction and the ground truth
        scalar
    """
    output = -np.mean(
        target_true * np.log(target_pred + 1e-15)
        + (1 - target_true) * np.log(1 - target_pred + 1e-15)
    )
    return output

def w2_nll_grad_input(target_pred, target_true):
    """Compute the partial derivative of NLL
        with respect to its input
    # Arguments
        target_pred: predictions - np.array of size `(n_objects, 1)`
        target_true: ground truth - np.array of size `(n_objects, 1)`
    # Output
        the partial derivative
        of NLL with respect to its input
        np.array of size `(n_objects, 1)`
    """
    grad_input = -(target_true / (target_pred + 1e-15) - (1 - target_true) / (1 - target_pred + 1e-15)) / target_pred.shape[0]
    return grad_input