import numpy as np

def fast_3x3_rank23(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    One of the schemes found for 3x3 matrix multiplication with a SOTA arithmetic complexity of 83.

    Parameters
    ----------
    A, B : (3, 3) np.ndarray
        Input matrices (row-major).

    Returns
    -------
    C : (3, 3) np.ndarray
        The product A @ B obtained with only 23 scalar multiplies and 60 additions / subtractions.
    """
    # Flatten to the A0...A8, B0...B8 shorthand
    A0,A1,A2,A3,A4,A5,A6,A7,A8 = A.ravel()
    B0,B1,B2,B3,B4,B5,B6,B7,B8 = B.ravel()

    # pre-additions
    t0 = A0 - A3
    t1 = A4 + A5
    t2 = A6 + A8
    t3 = A1 + A2
    t4 = A7 - t1
    t5 = t0 + t2

    u0 = B0 - B2
    u1 = B4 - B7
    u2 = B1 + u0
    u3 = B5 - B8
    u4 = B6 + u3
    u5 = u1 + u2

    # 23 scalar products
    M0  = (-t3) * (-B7)
    M1  = (-A3 + A4 - A7) * (-u1)
    M2  = (A1 - A3) * u5
    M3  = (-t0) * (-u0)
    M4  = (-A5) * u3
    M5  = (A8 + t4) * B7
    M6  = (-A8) * (-B2 + B7 + B8)
    M7  =  t4 * (B5 + B7)
    M8  = (-A7) * (-B3)
    M9  = (A1 + A5) * (-u4)
    M10 = (-t5) * (B2 - B6)
    M11 = (-A6) *  B1
    M12 = (A2 - A5 + t5) * (-B6)
    M13 = (-A0 + A1) *  u2
    M14 = (-A3) *  B2
    M15 = (A6 + t0) * (B0 - B6)
    M16 =  A7 * (B4 + B5)
    M17 =  t3 * (-B6 + B8)
    M18 = (-t2) *  B2
    M19 = (-A1) * (-B3 + u4 - u5)
    M20 = (-A1 + A4) *  B3
    M21 = (-t1) * (-B5)
    M22 =  A3 * (B1 + u1)

    # v-aggregates
    v0 = M4  - M14
    v1 = M2  + M22
    v2 = M7  + M21
    v3 = M9  - v0
    v4 = M10 - M18
    v5 = M3  - v1
    v6 = M5  - v2
    v7 = M12 + v3
    v8 = v4  + v7

    # outputs
    C0 =  M19 + v5 - v8
    C1 =  M0  - M13 - v5
    C2 =  M17 - v8
    C3 =  M19 + M20 - v1 - v3
    C4 = -M1  + M16 + M22 - v2
    C5 =  M21 + v0
    C6 = -M3  + M8  + M15 + v4
    C7 = -M11 + M16 + v6
    C8 = -M6  - M18 - v6

    return np.array([C0, C1, C2,
                     C3, C4, C5,
                     C6, C7, C8]).reshape(3, 3)

# Test: compare against NumPy for a batch of random matrices
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for _ in range(1_000):
        A = rng.integers(-9, 10, size=(3, 3))
        B = rng.integers(-9, 10, size=(3, 3))
        assert np.array_equal(fast_3x3_rank23(A, B), A @ B)
    print("All 1,000 random tests passed")