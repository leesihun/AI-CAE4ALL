"""Independent fp64 numpy reference for the 2D spectral convolution corner
scheme (section 8.3/16), coded directly against numpy's rfft2/irfft2 rather
than sharing any code with model/spectral.py.
"""

import numpy as np


def reference_spectral_conv_2d(x: np.ndarray, weight_blocks, modes) -> np.ndarray:
    """x: [Cin, R1, R2] real. weight_blocks: length-2 list of complex arrays
    [Cin, Cout, m1, m2]: block 0 is the (+, +) corner, block 1 is the (-, +)
    corner (matching SpectralConvNd's corner order for d=2:
    itertools.product([1, -1], repeat=1) == [(1,), (-1,)]).
    Returns y: [Cout, R1, R2] real.
    """
    Cin, R1, R2 = x.shape
    Cout = weight_blocks[0].shape[1]
    m1, m2 = modes

    X = np.fft.rfft2(x.astype(np.float64), axes=(1, 2), norm='ortho')  # [Cin, R1, R2//2+1]
    freq_shape = X.shape[1:]
    Y = np.zeros((Cout,) + freq_shape, dtype=np.complex128)

    corners = [(slice(0, m1), slice(0, m2)), (slice(-m1, None), slice(0, m2))]
    for block, (s1, s2) in zip(weight_blocks, corners):
        Xc = X[:, s1, s2]                       # [Cin, m1, m2]
        Yc = np.einsum('ipq,iopq->opq', Xc, block)  # [Cout, m1, m2]
        Y[:, s1, s2] = Yc

    y = np.fft.irfft2(Y, s=(R1, R2), axes=(1, 2), norm='ortho')
    return y
