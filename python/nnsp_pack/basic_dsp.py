import numpy as np
import scipy
filter = scipy.signal.lfilter
b_hpf = np.array([1.0, -1.0])
a_hpf = np.array([1.0, -0.965267479])

def dc_remove(x:np.float32)->np.float32:
    """remove audio dc term

    Args:
        x (float32): input

    Returns:
        float32: output
    """
    y = filter(b_hpf, a_hpf, x)
    return y
