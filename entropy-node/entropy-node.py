import numpy as np

def entropy_node(y):
    """
    Compute entropy for a single node using stable logarithms.
    """
    
    y = np.array(y, dtype = np.float64)
    if y is None :
        return 0
    a = np.unique(y, return_counts = True)
    p = np.float64(a[1] / len(y))

    e = abs(np.sum(p * np.log2(p)))
    
    return e