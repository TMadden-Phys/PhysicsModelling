import numpy as np; import numba as nb
from .LinAlg import planar_map
from . import AccretionDisk as AD


@nb.njit('f8(f8, f8, u2, f8)')
def calc_point_flux(area_const, sine_theta, temp, cosine_angle):
    '''
    Calculates the flux from a point on a surface to an observer
    '''
    return area_const * sine_theta * temp * cosine_angle

@nb.njit('f8(f8[:], u2[:,:], f8[:], f8, b1[:])')
def calc_surface_flux(inner_product, temps, sin_thetas, const, mask):
    '''
    Calculates the flux from a surface to an observer
    '''
    ix, iy = temps.shape
    tmp = 0
    for j in range(ix):
        for k in range(iy):
            if mask[j*iy + k] == True:
                tmp += calc_point_flux(const, sin_thetas[k], temps[j,k], inner_product[j*iy + k])
    return tmp

@nb.njit('b1[:](f8[:])')
def calc_hemisphere_points(inner_prod):
    '''
    Calculates the points that lie on the hemisphere in the direction of the observer
    '''
    ix = inner_prod.shape[0]
    temp_mask = np.zeros(inner_prod.shape[0]).astype(np.bool_)
    for i in range(ix):
        if inner_prod[i] > 0:
            temp_mask[i] = True
    return temp_mask

class BaseFlux:
    params = ['inner_product', 'temperatures', 'sin_thetas', 'area_constant']
    
    @staticmethod
    @nb.njit('(f8[:,:], u2[:,:], f8[:], f8, f8[:])', parallel = True)
    def calc_hemisphere_flux(inner_prod: np.float64, temps: np.uint16, sin_thetas: np.float64, const: np.float64, out: np.float64):
        '''
        Calculates the flux from the sphere's surface, first filters to see if the points lie on the same hemisphere as the observer ray.
        '''
        ix, iy = temps.shape
        jx = inner_prod.shape[0]
        for i in nb.prange(jx):
            hemisphere_mask = calc_hemisphere_points(inner_prod[i])
            out[i] = calc_surface_flux(inner_prod[i], temps, sin_thetas, const, hemisphere_mask)

# @nb.njit('(f8[:,:,:], f8[:,:], f8[:,:], f8[:,:],f8[:,:,:], f8[:,:], u2[:,:], f8[:], f8, f8[:])', parallel = False)
def calc_disk_flux(ortho_vectors, sphere_vectors, disk_projection, convex_idx, hull_extrema, inner_prod: np.float64, temps: np.uint16, sin_thetas: np.float64, const: np.float64, out: np.float64):
    '''
    Calculates the flux from the sphere's surface, first filters to see if the points lie on the same hemisphere as the observer ray, then also checks if the points are blocked by the disk.
    '''
    ix, iy = temps.shape
    jx = inner_prod.shape[0]
    for i in nb.prange(jx):
        
        hemisphere_mask = calc_hemisphere_points(inner_prod[i])
        e_i, e_j = ortho_vectors[i]
        planar_sphere = planar_map(sphere_vectors[hemisphere_mask], e_i, e_j)
        disk_mask = AD.MultiPointPoly_(planar_sphere, disk_projection, convex_idx, hulls_extrema=hull_extrema)
        hemisphere_mask[hemisphere_mask] = disk_mask & hemisphere_mask[hemisphere_mask]
        out[i] = calc_surface_flux(inner_prod[i], temps, sin_thetas, const, hemisphere_mask)

class DiskFlux:
    params = ['obs_indexes', 'ortho_vectors', 'sphere_vectors', 'disk_projections', 'disk_hull_indexes', 'disk_extremas', 'inner_product', 'temperatures', 'sin_thetas', 'area_constant']
    
    @staticmethod
    @nb.njit
    def calc_hemisphere_flux(obs_indexes, ortho_vectors, sphere_vectors, disk_projections, disk_hull_indexes, disk_extremas, inner_prod: np.float64, temps: np.uint16, sin_thetas: np.float64, const: np.float64, out: np.float64):
        for row_idx in range(obs_indexes.shape[0]):
            row_ID = np.arange(row_idx * obs_indexes.shape[-1], (row_idx + 1)*obs_indexes.shape[-1])
            calc_disk_flux(ortho_vectors[row_ID], sphere_vectors, disk_projections[row_idx], disk_hull_indexes[row_idx], disk_extremas[row_idx], inner_prod[row_ID], temps, sin_thetas, const, out)


    