import numpy as np; import pandas as pd; import time; import math
import matplotlib.pyplot as plt
from . import AccretionDisk as AD
from .LinAlg import planar_map


class DiskCrossSection:
    def __init__(self, equation, divisions, *args):
        self.equation = equation; self.divisions = divisions
        self.cross_section = self.gen_cross_section(*args)

    def gen_cross_section(self, *args):
        x = np.linspace(0, 2*np.pi, self.divisions)
        return self.equation(x, *args)

    def plot_crosssection(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(*self.cross_section)


class EllipticCrossSection(DiskCrossSection):
    def __init__(self, cross_section_width, cross_section_height, inner_radius, divisions):
        super().__init__(self.ellipse, divisions, cross_section_width, cross_section_height, inner_radius)
        self.cross_section_height = cross_section_height; self.cross_section_width = cross_section_width
        self.inner_radius = inner_radius
    
    def ellipse(self, domain_points, cross_section_width, cross_section_height, inner_radius):
        x_arr = 0.5*cross_section_width*(1+np.cos(domain_points)) + inner_radius
        y_arr = cross_section_height*np.sin(domain_points)
        return np.stack((x_arr+1, y_arr))


class AccretionDisk:
    def __init__(self, cross_section_class, rotation_divisions, tilt_angle):
        self.rotation_divisions = rotation_divisions
        self.tilt_angle = tilt_angle
        self.remapped_cross_section = np.stack(
            (cross_section_class.cross_section[0],
             np.zeros_like(cross_section_class.cross_section[0]),
             cross_section_class.cross_section[1]), axis = -1)
        self.disk = AD.volume_revolution(self.remapped_cross_section, rotation_divisions, tilt_angle = tilt_angle)
    
    def plot_disk(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection = '3d')
        ax.scatter(*self.disk.T, s=1)
        ax.axes.set_zlim3d(np.min(self.disk.T[0]),np.max(self.disk.T[0]))
    

class DiskProjection:
    def __init__(self, AccretionDisk: AccretionDisk, observer_polar_angle, observer_azim_angle, plane_divisions):
        self.disk = AccretionDisk.disk; self.observer_theta = observer_polar_angle; self.observer_azim = observer_azim_angle; self.plane_divisions = plane_divisions
        self.observer_ray = np.array(
                    (np.cos(self.observer_azim)*np.sin(self.observer_theta),
                    np.sin(self.observer_azim)*np.sin(self.observer_theta),
                    np.cos(self.observer_theta)))
        self.ortho_vectors = np.stack(
            (np.array((-np.sin(self.observer_azim), np.cos(self.observer_azim), 0)),
             np.array((np.cos(self.observer_azim)*np.cos(self.observer_theta), np.sin(self.observer_azim)*np.cos(self.observer_theta), -np.sin(self.observer_theta)))))
        self.cartesian_lines = self.gen_planardivisions()
        self.disk_projection, self.disk_hull_indexes, self.disk_extrema = self.gen_projection()
    
    def gen_projection(self):
        ring_filtered = self.disk.reshape(self.disk.shape[0]*self.disk.shape[1], self.disk.shape[2])
        ring_filtered2 = ring_filtered[np.dot(ring_filtered, self.observer_ray) > 0]
        point_cloud = np.round(planar_map(ring_filtered2, *self.ortho_vectors), 7)
        indexes = AD.split_convhull(point_cloud, self.cartesian_lines)
        ordered_indexes = AD.reorder_convex_hull(point_cloud, indexes)
        return point_cloud, ordered_indexes, AD.gen_argmax(point_cloud, ordered_indexes)
    
    def gen_planardivisions(self):
        thetas = np.linspace(0,2*np.pi, self.plane_divisions, endpoint=False)
        xy = np.stack((np.cos(thetas), np.sin(thetas)), axis = -1)
        return xy
    
    def plot_projection(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.scatter(*self.disk_projection.T, s = 1)
    
    def plot_divided_projection(self):
        fig = plt.figure()
        ax = fig.add_subplot()
        for line in self.cartesian_lines:
            ax.plot(*(np.arange(2)*line[:,None]), color = 'blue', linewidth = 1)
        for poly in range(self.disk_hull_indexes.shape[0]):
            ordered_points = self.disk_projection[(self.disk_hull_indexes[poly][~np.isnan(self.disk_hull_indexes[poly])]).astype(np.int32)]
            ax.plot(*ordered_points.T, linewidth = 0.5, color = 'hotpink')

class DiskProjectionsContainer:
    def __init__(self, AccretionDisk: AccretionDisk, observer_polar_angles: np.ndarray[np.float64, np.float64], observer_azim_angle: np.float64, plane_divisions: np.int32):
        self.observer_thetas = observer_polar_angles; self.observer_azim = observer_azim_angle
        self.AccretionDisk = AccretionDisk
        self.disk_projections, self.disk_hull_indexes, self.disk_extremas, self.projection_classes = self.gen_disk_container(plane_divisions)

    def gen_disk_container(self, plane_divisions):
        disk_projections = list(); disk_hull_indexes = list(); disk_extremas = list(); projection_classes = list()
        for i in range(self.observer_thetas.shape[0]):
            single_projection = DiskProjection(self.AccretionDisk, self.observer_thetas[i], self.observer_azim, plane_divisions)
            projection_classes.append(single_projection)
            disk_projections.append(single_projection.disk_projection)
            disk_hull_indexes.append(single_projection.disk_hull_indexes)
            disk_extremas.append(single_projection.disk_extrema)
        return disk_projections, disk_hull_indexes, disk_extremas, projection_classes
    
    def plot_projections(self):
        fig = plt.figure()
        sqr = math.ceil(math.sqrt(self.observer_thetas.shape[0]))
        axes = [fig.add_subplot(sqr, sqr, i+1) for i in range(self.observer_thetas.shape[0])]
        for i in range(self.observer_thetas.shape[0]):
            axes[i].scatter(*self.projection_classes[i].disk_projection.T, s = 1, c = 'hotpink')
        fig.tight_layout()

    def plot_divided_projection(self):
        fig = plt.figure()
        sqr = math.ceil(math.sqrt(self.observer_thetas.shape[0]))
        axes = [fig.add_subplot(sqr, sqr, i+1) for i in range(self.observer_thetas.shape[0])]
        for i in range(self.observer_thetas.shape[0]):
            for line in self.projection_classes[i].cartesian_lines:
                axes[i].plot(*(np.arange(2)*line[:,None]), color = 'blue', linewidth = 1)
            for poly in range(self.disk_hull_indexes[i].shape[0]):
                ordered_points = self.projection_classes[i].disk_projection[(self.projection_classes[i].disk_hull_indexes[poly][~np.isnan(self.projection_classes[i].disk_hull_indexes[poly])]).astype(np.int32)]
                axes[i].plot(*ordered_points.T, linewidth = 0.5, color = 'hotpink')
    
    def gen_params(self):
        return dict(
            disk_projections = self.disk_projections,
            disk_hull_indexes = self.disk_hull_indexes,
            disk_extremas = self.disk_extremas,
        )