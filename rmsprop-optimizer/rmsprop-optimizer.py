import numpy as np

def rmsprop_step(w, g, s, lr=0.001, beta=0.9, eps=1e-8):
    """
    Perform one RMSProp update step.
    """
    w, g, s = map(np.asarray, [w, g, s])
    
    # Update the moving average of the squared gradients
    s_new = beta * s + (1 - beta) * g**2

    # Corrected update: lr in numerator, eps in denominator
    # Note: eps is usually inside the sqrt or added to it
    w_new = w - (lr / (np.sqrt(s_new) + eps)) * g

    return w_new, s_new
