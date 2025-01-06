import numpy as np; import numba as nb; import math
from . SpreadClasses import RectSliceSpread, EllipticSpread


class EllipticFlow:
    # params = ['dt', 'rise_velocities', 'cool_velocities', 'temperatures', 'rise_time', 'rise_bound', 'cool_bound', 'theta_array']
    
    # @staticmethod
    # @nb.njit()
    # def adjust_flow(idx, dt, rise_velocities, cool_velocities, temperatures, rise_time, rise_bound, cool_bound, theta_array):
    #     if idx * dt < rise_time:
    #         rise_bound += rise_velocities * dt
    #         EllipticSpread.inbounds_single(rise_bound, theta_array, out=temperatures, temp = np.uint16(1))
    #     if idx * dt > rise_time:
    #         cool_bound += cool_velocities * dt
    #         EllipticSpread.inbounds_single(cool_bound, theta_array, out=temperatures, temp = np.uint16(0))
    
    params = ['dt', 'theta_array', 'temperatures']
    def adjust_flow(cls: EllipticSpread, dt, theta_array, temperatures, temperature):
        cls.expand_flow(dt)
        cls.inbounds_single(theta_array, temperatures, temperature)
        
    

class RectSliceFlow:
    params = ['dt', 'temperatures']
    def adjust_flow(cls: RectSliceSpread, dt, temperatures, temperature):
        cls.expand_flow(dt)
        cls.in_box(temperatures, temperature)
    

