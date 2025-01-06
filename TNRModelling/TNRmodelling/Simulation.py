import numpy as np; import pandas as pd; import time
import matplotlib.pyplot as plt
from . import SpreadClasses as Spreads
from . import AccretionDiskClasses as AccClass
from . import LinAlg
from . import FluxModelling as Flux
from . import FlowModelling as Flow
import inspect; from functools import reduce

class BaseVariables:
    observer_ray = np.array((1,0,0)); NS_rotation_axis = np.array((0,0,1)); system_rotation_axis = np.array((0,1,0))

class NeutronStar:
    def __init__(self, angle_divisions):
        self.angle_divisions = angle_divisions
        self.coords, self.NS_temps, self.NS_thetas, self.NS_phis = self.__sim_setup_NS(self.angle_divisions)
        self.sphere_vectors = self.__gen_coords()
        self.sin_thetas = np.sin(self.NS_thetas); self.cos_phis = np.cos(self.NS_phis)
        self.delta_theta = abs(self.NS_thetas[1]-self.NS_thetas[0]); self.delta_phi = abs(self.NS_phis[1]-self.NS_phis[0])
        self.area_const = self.delta_phi*self.delta_theta
        self.shapes = (self.NS_thetas.shape[-1], self.NS_phis.shape[-1])
    
    def __gen_coords(self):
        x, y, z = self.coords
        sphere_vectors = np.stack((x.flatten(), y.flatten(), z.flatten()), axis = -1)
        return sphere_vectors

    def __sim_setup_NS(self, angle_divisions):
        # get both cartesian and polar coordinates of the sphere
        coords, NS_thetas, NS_phis = self.gen_sphere(angle_divisions = angle_divisions)
        print(np.round(abs(NS_thetas[1]-NS_thetas[0]),5), 'mesh theta angle difference')
        print(np.round(abs(NS_phis[1]-NS_phis[0]),5), 'mesh phi angle difference')
        # create a temperature variable for every point on the sphere
        NS_temps = np.zeros((NS_phis.size, NS_thetas.size), dtype = np.uint16)
        return coords, NS_temps, NS_thetas, NS_phis

    
    def gen_sphere(self, R=1, angle_divisions = 50):
        ''' Generates a basic sphere'''
        thetas = np.linspace(0,np.pi, angle_divisions, endpoint= False)
        phis = np.linspace(0,2*np.pi, angle_divisions, endpoint = False)
        xs , ys = np.meshgrid(thetas, phis)
        x = R*np.sin(xs)*np.cos(ys)
        y = R*np.sin(xs)*np.sin(ys)
        z = R*np.cos(xs)
        return (x,y,z), thetas, phis
    
    def gen_params(self):
        params_dict = dict(
                        sphere_vectors = self.sphere_vectors,
                        temperatures = self.NS_temps,
                        area_constant = self.area_const,
                        theta_array = self.NS_thetas,
                        sin_thetas = self.sin_thetas,
                        cos_phis = self.cos_phis
                        )
        return params_dict
    
class GeneralSim:
    def __init__(self, NS_Hz, dt, system_angles, observer_angles, total_models, rise_time, cool_time):
        combinations = np.meshgrid(system_angles, observer_angles, indexing='xy')
        self.obs_index = np.indices(combinations[0].shape)[0]
        self.sys_angle_combinations, self.obs_angle_combinations = [angle_comb.flatten() for angle_comb in combinations]
        self.NS_Hz = NS_Hz; self.dt = dt; self.total_models = total_models
        self.rise_time = rise_time; self.cool_time = cool_time; self.model_time = self.rise_time+self.cool_time
        self.iterations = int(np.ceil(self.model_time / self.dt))
        self.timesteps = np.arange(self.iterations)*self.dt
        self.observer_vec_sims, self.NS_rot_axes =  self.rotations(BaseVariables.observer_ray,BaseVariables.NS_rotation_axis, self.sys_angle_combinations, BaseVariables.system_rotation_axis, self.obs_angle_combinations)
        self.NS_rotation_array = self.__gen_rotation_array()
        self.ortho_vectors = self.gen_ortho_vectors(self.sys_angle_combinations, BaseVariables.system_rotation_axis, self.obs_angle_combinations)
        
    def __gen_rotation_array(self):
        delta_rotation = 2 * np.pi * self.NS_Hz * self.dt
        NS_rotation_array = np.zeros((self.total_models, 3,3))
        print(NS_rotation_array.shape, self.NS_rot_axes.shape)
        for i in range(self.total_models):
            NS_rotation_array[i] = LinAlg.general_rotation(*self.NS_rot_axes[i], angle = delta_rotation)
        return NS_rotation_array
    
    def rotations(self, observer_point, NS_rotation_axis, system_rotation_angles, system_rot_axis, observer_rotation_angles):
        observer_vec_sims = np.zeros((system_rotation_angles.shape[0], *observer_point.shape)); NS_rot_ax = np.zeros((system_rotation_angles.shape[0], *observer_point.shape))
        for i in range(system_rotation_angles.shape[0]):
            system_rotation = LinAlg.general_rotation(*system_rot_axis, system_rotation_angles[i])
            NS_rot_ax[i] =  system_rotation @ NS_rotation_axis
            observer_vec_sims[i] =  system_rotation @ LinAlg.general_rotation(*system_rot_axis, observer_rotation_angles[i]) @ observer_point
        return observer_vec_sims, NS_rot_ax
    
    def gen_ortho_vectors(self, system_rotation_angles, system_rot_axis, observer_rot_angles):
        ortho_vecs = np.zeros((system_rotation_angles.shape[0], 2 , 3))
        for i in range(system_rotation_angles.shape[0]):
            system_rotation = LinAlg.general_rotation(*system_rot_axis, system_rotation_angles[i])
            ortho_vecs[i,0] = system_rotation @ LinAlg.general_rotation(*system_rot_axis, observer_rot_angles[i]) @ np.array((0,0,1))
            ortho_vecs[i,1] = system_rotation @ LinAlg.general_rotation(*system_rot_axis, observer_rot_angles[i]) @ np.array((0,1,0))
        return ortho_vecs
    
    def gen_params(self, NeutronStarClass):
        divide = 4
        steps = self.iterations//divide
        result = np.zeros((self.iterations, self.total_models))
        inner_product = np.empty((self.total_models, NeutronStarClass.sphere_vectors.shape[0]))
        param_dict = dict(
                        observer_vectors = self.observer_vec_sims,
                        obs_indexes = self.obs_index,
                        ortho_vectors = self.ortho_vectors,
                        rotation_array = self.NS_rotation_array,
                        inner_product = inner_product,
                        rise_time = self.rise_time,
                        dt = self.dt,
                        iterations = self.iterations,
                        result = result,
                        steps = steps
                        )
        return param_dict

class CombineSim:
    def __init__(self, NeutronStarClass, GeneralSimClass, RiseSpreadClass = None, CoolSpreadClass = None, DiskClass = None, GeneralRelativityClass = None):
        self.ClassesDict = dict(
            NeutronStarClass = NeutronStarClass,
            GeneralSimClass = GeneralSimClass,
            DiskClass = DiskClass,
            GeneralRelativityClass = GeneralRelativityClass
        )
        self.FlowClassDict = dict(
            RiseSpreadClass = RiseSpreadClass,
            CoolSpreadClass = CoolSpreadClass,
        )
        if isinstance(RiseSpreadClass, Spreads.EllipticSpread):
            self.flow_func = Flow.EllipticFlow
        elif isinstance(RiseSpreadClass, Spreads.RectSliceSpread):
            self.flow_func = Flow.RectSliceFlow
        else:
            # for tests
            self.flow_func = None
        if DiskClass == None:
            self.flux_func = Flux.BaseFlux
        else:
            self.flux_func = Flux.DiskFlux
    
    def setup_funcparams(self):
        full_param_dict = []
        for value in self.ClassesDict.values():
            if value is not None:
                full_param_dict.append(value.gen_params(*[self.ClassesDict[j] for j in inspect.getfullargspec(value.gen_params)[0] if j != 'self']))
        full_param_dict = reduce(lambda d1, d2: d1|d2, full_param_dict)
        flux_params = [full_param_dict[i] for i in self.flux_func.params]
        if self.flow_func is not None:
            flow_params = [full_param_dict[i] for i in self.flow_func.params]
        else:
            flow_params = None
        return full_param_dict, flow_params, flux_params
    
    def main_loop(self):
        all_params, flow_params, flux_params = self.setup_funcparams()
        # dynamic variables 
        print('iterations', all_params['iterations'])
        for i in range(all_params['iterations']):
            if i % all_params['steps'] == 0:
                t0 = time.perf_counter()
            LinAlg.calc_inner_product(all_params['observer_vectors'], all_params['sphere_vectors'], out = all_params['inner_product'])
            if i*all_params['dt'] < all_params['rise_time']:
                self.flow_func.adjust_flow(self.FlowClassDict['RiseSpreadClass'], *flow_params, temperature = np.uint16(1))
            if i*all_params['dt'] > all_params['rise_time']:
                self.flow_func.adjust_flow(self.FlowClassDict['CoolSpreadClass'], *flow_params, temperature = np.uint16(0))
            self.flux_func.calc_hemisphere_flux(*flux_params, out = all_params['result'][i])
            LinAlg.rotate_vector(all_params['rotation_array'], all_params['observer_vectors'])
            LinAlg.rotate_vector(all_params['rotation_array'], all_params['ortho_vectors'][:,0])
            LinAlg.rotate_vector(all_params['rotation_array'], all_params['ortho_vectors'][:,1])
            if (((i - all_params['steps']//4) % all_params['steps']) == 0):
                tfin = time.perf_counter() - t0
                print(np.round(tfin, 4), 's/{} loops'.format(all_params['steps']//4))
                print(np.round(tfin/(all_params['steps']//4), 4), 's/loop')
                print(np.round(tfin * 4, 4), 'step time/s')
                # fig = plt.figure(); ax = fig.add_subplot(projection = '3d')
                # ax.scatter(*all_params['sphere_vectors'].T, c = all_params['temperatures'], s = 3)
        for flowclass in self.FlowClassDict.values():
            flowclass.reset_sim_params()
        return all_params['result'], np.arange(all_params['iterations'])*all_params['dt'], all_params['iterations']*all_params['dt']
    

def save_data(permutations, temperatures, timesteps, save_dir, save_name):
    indexes = pd.MultiIndex.from_arrays((np.round(permutations, 7)),names = ['flash position', 'observer position'])
    df = pd.DataFrame(np.round(temperatures,7), index = indexes, columns=np.round(timesteps, 7))
    df.to_csv('{}\{}.csv'.format(save_dir, save_name))

def RunEverything(
    NS_Hz, angle_divisions, dt,
    flash_divisions, fmin, fmax, flash_axes_ratio, observer_divisions, omin, omax,
    rise_time, cool_time,
    acc_disk, loop_resolution, angular_resolution, tilt_angle, inner_radius, disk_radius, disk_width, observer_azim, domain_divisions,
    save_model, save_dir, save_name):
    system_angles = np.pi*np.linspace(fmin, fmax, flash_divisions); observer_angles = np.pi*np.linspace(omin, omax, observer_divisions)

    # NEUTRON STAR
    NeutronStarClass = NeutronStar(angle_divisions)
    GeneralSimClass = GeneralSim(NS_Hz=NS_Hz, dt = dt, system_angles = system_angles, observer_angles=observer_angles, total_models=flash_divisions*observer_divisions,
                             rise_time=rise_time, cool_time = cool_time)
    
    # SPREADS
    RiseSpreadClass = Spreads.EllipticSpread(
    ellipse_axes_ratio= flash_axes_ratio, flow_time = rise_time, NS_phi_array=NeutronStarClass.NS_phis)
    CoolSpreadClass = Spreads.EllipticSpread(
        ellipse_axes_ratio= flash_axes_ratio, flow_time = cool_time, NS_phi_array=NeutronStarClass.NS_phis)
    
    # ACCRETION DISK
    if acc_disk == True:
        disk_crosssect = AccClass.EllipticCrossSection(disk_radius, disk_width, inner_radius, loop_resolution)
        disk_shape = AccClass.AccretionDisk(disk_crosssect, angular_resolution, tilt_angle)
        ProjectionClass = AccClass.DiskProjectionsContainer(disk_shape, observer_angles, observer_azim, domain_divisions)
    else:
        ProjectionClass = None

    # MODEL
    model = CombineSim(NeutronStarClass, GeneralSimClass, RiseSpreadClass, CoolSpreadClass, ProjectionClass)
    temperatures, timesteps, model_time = model.main_loop()
    if save_model == True:
        save_data(np.stack(np.meshgrid(system_angles, observer_angles, indexing='xy'), axis = -1).reshape(flash_divisions*observer_divisions,-1).T, temperatures, timesteps, save_dir, save_name)
    return model, model_time, timesteps, temperatures

