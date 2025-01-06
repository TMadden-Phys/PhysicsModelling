import numpy as np; import numba as nb

@nb.njit('(f8[:,:], f8[:,:], f8[:,:])')
def calc_inner_product(vectors: np.ndarray[np.float64, np.float64],sphere_vecs: np.ndarray[np.float64, np.float64],out: np.ndarray[np.float64, np.float64]):
    ix, iy = vectors.shape
    jx, jy = sphere_vecs.shape
    for i in nb.prange(ix):
        for k in range(jx):
            tmp = 0
            for j in range(jy):
                tmp += vectors[i,j]*sphere_vecs[k,j]
            out[i,k] = tmp

@nb.njit('(f8[:,:,:], f8[:,:])')
def rotate_vector(Rotations: np.ndarray[np.float64, np.float64], vectors: np.ndarray[np.float64, np.float64]):
    ix, iy, iz = Rotations.shape
    jx, jy = vectors.shape
    tmp_array = np.empty(jy, dtype = np.float64)
    for i in nb.prange(ix):
        for j in range(jy):
            tmp_array[j] = 0
            for k in range(iz):
                tmp_array[j] += Rotations[i,j,k]*vectors[i,k]
        for j in range(jy):
            vectors[i,j] = tmp_array[j]

@nb.njit('f8[:,:](f8[:,:], f8[:], f8[:])', parallel = True)
def planar_map(ring, e_j, e_k):
    '''
    Maps a 3D shape to the 2D plane orthogonal to a vector.
    '''
    ix, iy = ring.shape
    idx = np.empty((ix, 2))
    for i in nb.prange(ix):
        tmp = 0
        tmp2 = 0
        for j in range(iy):
            tmp += ring[i,j]*e_j[j]
            tmp2 += ring[i,j]*e_k[j]
        idx[i, 0] = tmp; idx[i, 1] = tmp2
    return idx

def general_rotation(n1, n2, n3, angle):
        '''Generates a rotation matrix for a rotation of theta about an arbitrary axis'''
        cosine = np.cos(angle)
        one_minus  = 1 - cosine
        sine = np.sin(angle)
        return np.array((
            (cosine + one_minus*n1**2, n1*n2*one_minus- n3*sine, n1*n3*one_minus + n2*sine),
            (n1*n2*one_minus + n3*sine, cosine + one_minus*n2**2, n2*n3*one_minus - n1*sine),
            (n1*n3*one_minus - n2*sine, n2*n3*one_minus + n1*sine, cosine + one_minus*n3**2))
        )