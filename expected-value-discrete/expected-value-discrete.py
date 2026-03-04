import numpy as np

def expected_value_discrete(x, p):
    """
    Returns: float expected value
    """
    # Write code here
    if np.sum(p) != 1 :
        raise ValueError
    else : 
        x = np.array(x, dtype = np.float64)
        p = np.array(p, dtype = np.float64)
        e = np.sum(x*p)
    
    return e
