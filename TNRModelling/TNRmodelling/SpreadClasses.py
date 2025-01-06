import numpy as np; import pandas as pd; import time; import math
from numba.experimental import jitclass; import numba as nb


# class EllipticSpread:
#     def __init__(self, ellipse_axes_ratio):
#         self.axes_ratio = ellipse_axes_ratio
    
#     def flowspeed_variation(self, eccentricity, cosine_phi_pos):
#         return (1 / np.sqrt(1 - (eccentricity * cosine_phi_pos)**2))
    
#     def convert_to_ecc(self, ab_ratio):
#         # eccentricity = sqrt(1 - 1/ab_ratio**2)
#         return math.sqrt(1-1/ab_ratio**2)

#     def gen_params(self, NeutronStarClass, GeneralSimClass):
#         avg_rise_vel = np.pi/GeneralSimClass.rise_time; avg_cool_vel = np.pi/GeneralSimClass.cool_time
#         rise_bound = np.zeros(NeutronStarClass.angle_divisions); rise_velocities = avg_rise_vel * self.flowspeed_variation(self.convert_to_ecc(self.axes_ratio), NeutronStarClass.cos_phis)
#         cool_bound = np.zeros(NeutronStarClass.angle_divisions); cool_velocities = avg_cool_vel * self.flowspeed_variation(self.convert_to_ecc(self.axes_ratio), NeutronStarClass.cos_phis)
#         params_dict = dict(
#                         rise_bound = rise_bound,
#                         cool_bound = cool_bound,
#                         rise_velocities = rise_velocities,
#                         cool_velocities = cool_velocities
#                         )
#         return params_dict    

#     @staticmethod
#     @nb.njit('(f8[:], f8[:], u2[:,:], u2)')
#     def inbounds_single(boundary: np.float64, thetas: np.float64, out:np.uint16, temp: np.uint16):
#         '''
#         Calculates if the sphere points are within the flow bound if the flow boundary is aligned to the sphere points. 

#         Calculates for single boundaries.
#         '''
#         ix,iy = out.shape
#         for i in range(ix):
#             for j in range(iy):
#                 if thetas[j] < boundary[i]:
#                     out[i,j] = temp

elliptic_spec = [
    ('axes_ratio', nb.float64),
    ('boundary_array', nb.float64[:]),
    ('velocities_array', nb.float64[:])
]

@jitclass(elliptic_spec)
class EllipticSpread:
    def __init__(self, ellipse_axes_ratio, flow_time, NS_phi_array):
        self.axes_ratio = ellipse_axes_ratio
        avg_velocity = np.pi/flow_time
        self.boundary_array = np.zeros_like(NS_phi_array); self.velocities_array = avg_velocity * self.flowspeed_variation(self.convert_to_ecc(self.axes_ratio), np.cos(NS_phi_array))
        
    def flowspeed_variation(self, eccentricity, cosine_phi_pos):
        return (1 / np.sqrt(1 - (eccentricity * cosine_phi_pos)**2))
    
    def convert_to_ecc(self, ab_ratio):
        # eccentricity = sqrt(1 - 1/ab_ratio**2)
        return math.sqrt(1-1/ab_ratio**2)

    def expand_flow(self, dt):
        self.boundary_array += self.velocities_array * dt

    def inbounds_single(self, thetas: np.float64, out:np.uint16, temp: np.uint16):
        '''
        Calculates if the sphere points are within the flow bound if the flow boundary is aligned to the sphere points. 

        Calculates for 1D boundaries.
        '''
        ix,iy = out.shape
        for i in range(ix):
            for j in range(iy):
                if thetas[j] < self.boundary_array[i]:
                    out[i,j] = temp
    
    def reset_sim_params(self):
        self.boundary_array = np.zeros_like(self.boundary_array)

rect_spec = [
    ('slices', nb.float64[:]),
    ('slice_size', nb.float64),
    ('slice_midpoints', nb.float64[:]),
    ('velocity_variation', nb.float64[:]),
    ('radius_variation', nb.float64[:]),
    ('slice_angular_distance', nb.float64[:]),
    ('slice_bins', nb.float64[:]),
    ('slice_indexes', nb.int32[:,:]),
    ('reshaped_sphere', nb.float64[:,:]),
    ('spread_boundary', nb.float64[:]),
    ('thetas', nb.float64[:]),
    ('phi', nb.float64),
    ('axes_ratio', nb.float64),
    ('avg_velocity', nb.float64)
]

@jitclass(rect_spec)
class RectSliceSpread:
    def __init__(self, rectangles, sphere_coords: nb.types.UniTuple, axes_ratio, flow_time):
        self.axes_ratio = axes_ratio; self.avg_velocity = np.pi/(2*flow_time)
        self.slices = np.arange(rectangles, dtype=np.float64)/rectangles
        self.slice_size = 1/rectangles
        self.slice_midpoints = self.slices+self.slice_size/2
        self.velocity_variation = np.ones_like(self.slice_midpoints, dtype = np.float64)
        # 1 - 2*np.arcsin(self.slice_midpoints)/np.pi
        self.radius_variation = np.ones_like(self.slice_midpoints, dtype = np.float64)
        # np.cos(np.arcsin(self.slice_midpoints))
        self.slice_angular_distance = np.arcsin(self.slice_midpoints)
        self.slice_bins = np.append(self.slices,1)
        self.slice_indexes = np.zeros_like(sphere_coords[0], dtype= np.int32)
        self.reshaped_sphere = np.zeros_like(sphere_coords[2])
        self.spread_boundary = np.zeros_like(self.slices, dtype = np.float64)
        self.thetas = np.zeros_like(self.slices, dtype = np.float64)
        self.arange_sphere(sphere_coords[0], sphere_coords[2])
        self.phi = 0.
        self.index_sphere_coords(sphere_coords[0])
    
    def index_sphere_coords(self, indexing_axis):
        ix, iy = indexing_axis.shape
        for i in range(ix):
            for j in range(iy):
                for k in range(self.slice_bins.size-1):
                    test_coord = abs(indexing_axis[i,j])
                    if (test_coord > self.slice_bins[k]) & (test_coord <= self.slice_bins[k+1]):
                        self.slice_indexes[i,j] = k
    
    def arange_sphere(self, x, z):
        ix, iy = z.shape
        for i in range(ix):
            for j in range(iy):
                self.reshaped_sphere[i,j] = z[i,j] - math.cos(math.asin(x[i,j]))
    
    def in_box(self, temperatures, temperature, tolerance = 1e-7):
        ix, iy = self.reshaped_sphere.shape
        for i in nb.prange(ix):
            for j in nb.prange(iy):
                if self.spread_boundary[self.slice_indexes[i,j]] > 0:
                    if abs(self.reshaped_sphere[i,j]) <= self.spread_boundary[self.slice_indexes[i,j]]+tolerance:
                        temperatures[i,j] = temperature

    def expand_flow(self, dt):
        dphi = self.avg_velocity*dt; dtheta = self.axes_ratio*dphi
        self.phi += dphi
        for m in range(self.spread_boundary.shape[0]):
            if self.slice_angular_distance[m] < self.phi:
                self.thetas[m] += dtheta
                if self.thetas[m] >= np.pi:
                    self.spread_boundary[m] = 2.
                else:
                    self.spread_boundary[m] = self.radius_variation[m] * (1-math.cos(self.velocity_variation[m]*self.thetas[m]))
    
    def reset_sim_params(self):
        self.thetas = np.zeros_like(self.slices, dtype = np.float64)
        self.spread_boundary = np.zeros_like(self.slices, dtype = np.float64)
        self.phi = 0.