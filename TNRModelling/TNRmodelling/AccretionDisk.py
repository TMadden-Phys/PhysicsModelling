import numpy as np; import numba as nb; import math; from scipy.spatial import ConvexHull; import matplotlib.pyplot as plt

@nb.njit
def volume_revolution(coords, slices, tilt_angle = 0):
    '''
    Returns an angled surface of revolution from a 2D slice of the surface.
    '''
    ix,iy = coords.shape
    theta = 2*np.pi/slices

    # rotation around the z axis
    r0 = np.array(((np.cos(theta),-np.sin(theta)),(np.sin(theta), np.cos(theta)))); rotation = r0.copy(); rot_temp = np.zeros_like(r0)
    jx, jy = r0.shape

    # tilt of the surface about the y axis.
    tilt_matrix = np.array(((np.cos(tilt_angle), 0 , -np.sin(tilt_angle)), (0,1,0), (np.sin(tilt_angle), 0, np.cos(tilt_angle))))

    # generates the surface of revolution
    template = np.zeros((slices, ix, iy))
    for i in range(slices):
        # generates all of the successive rotations about the axis
        # could refactor to just multiply the angle and form the array from that
        for j in range(jx):
            for k in range(jy):

                tmp = 0
                for m in range(jy):
                    tmp += r0[j,m]*rotation[m,k]
                
                rot_temp[j,k] = tmp
        rotation = rot_temp.copy()

        # perform the rotation about the z axis
        for j in range(ix):
            for k in range(iy):
                if k < 2:
                    tmp = 0
                    for m in range(2):
                        tmp += rotation[k,m]*coords[j,m]
                    template[i,j,k] = tmp
                else:
                    template[i,j,k] = coords[j,k]

    # perform the tilting of the surface about y axis
    for i in range(slices):
        for j in range(ix):
            for k in range(iy):
                tmp = 0
                for m in range(iy):
                    tmp += tilt_matrix[k,m] * template[i,j,m]
                template[i,j,k] = tmp
    return template

@nb.njit('f8[:,:](f8[:,:],f8[:,:])')
def split_points(point_cloud, lines):
    '''
    Takes a 2D point cloud and returns groups of the points depending on which slice of a plane they lie in.
    '''
    ix, iy = lines.shape
    jx, jy= point_cloud.shape
    template = np.empty((ix,jx))
    for i in range(ix):
        for j in range(jx):
            cross = lines[i, 0] * point_cloud[j, 1]  - lines[i,1] * point_cloud[j,0]
            cross2 = lines[(i+1) % ix, 0] * point_cloud[j, 1]  - lines[(i+1) % ix,1] * point_cloud[j,0]
            if (cross >= 0) & (cross2 < 0):
                template[i,j] = j
            else:
                template[i,j] = np.nan
    return template

@nb.njit('f8[:,:](f8[:,:], f8[:], i2)')
def replace_index_inplace(index_array, conv_pnts, idx):
    '''
    Iterates through an index array containing nans and finds the indexes that lie on the convex hull
    '''
    for j in range(index_array.shape[1]):
        if index_array[idx,j] == np.nan:
            continue
        truth = False
        for k in range(conv_pnts.shape[0]):
            if j == conv_pnts[k]:
                truth = truth | True
        if truth == False:
            index_array[idx,j] = np.nan
        else:
            pass

    return index_array

def split_convhull(point_cloud, lines):
    '''
    Splits a conccave polygon that is concave about teh origin into triangular slices.
    Then performs a convex hull algorithm on these slices and stitches them together to get an approximation of the concave polygon'''
    index_array = split_points(point_cloud, lines)
    ix,iy = index_array.shape
    for i in range(ix):
        mask = ~np.isnan(index_array[i])
        if point_cloud[mask].shape[0] > 3:
            conv = ConvexHull(point_cloud[mask])
            conv_index = index_array[i][mask][conv.vertices]
            index_array = replace_index_inplace(index_array, conv_index, i)
    return index_array

@nb.njit('f8[:,:](f8[:,:], f8[:,:])')
def reorder_convex_hull(pointcloud: np.array, convexhull_idx: np.array):
    ''' 
    Given an array containing nans and points that lie on a set of convex hulls
     return a reduced size array containing the points on the convex hull.
    '''
    ix, iy = pointcloud.shape
    jx, jy = convexhull_idx.shape
    max_shape = pointcloud[~np.isnan(convexhull_idx[0])].shape[0]
    # find the maximum number of elements of the array needed (Memory management)
    for i in range(1,jx,1):
        if pointcloud[~np.isnan(convexhull_idx[i])].shape[0] > max_shape:
            max_shape = pointcloud[~np.isnan(convexhull_idx[i])].shape[0]

    # Takes a set of convex hull points and orders them based on the angle between the x axis
    # and the lowest y value point. If there are multiple lowest y values, take the lowest x value.
    filtered_indexes = np.empty((jx,max_shape))
    for i in range(jx):
        for j in range(max_shape):
            filtered_indexes[i,j] = np.nan
    
        filtered_points = pointcloud[~np.isnan(convexhull_idx[i])]
        if filtered_points.size == 0:
            continue
        shifted_points = np.empty(filtered_points.shape[0])
        min_val = filtered_points[0]; min_idx = 0

        for j in range(filtered_points.shape[0]):

            if filtered_points[j,1] < min_val[1]:
                min_val = filtered_points[j]; min_idx = j
            if filtered_points[j,1] == min_val[1]:
                if filtered_points[j,0] < min_val[0]:
                    min_val = filtered_points[j]; min_idx = j
        
        shifted_points[min_idx]  = -1
        for j in range(filtered_points.shape[0]):
            norm = 0
            for k in range(filtered_points.shape[-1]):
                norm += (filtered_points[j,k]-min_val[k]) ** 2
            
            if j != min_idx:
                shifted_points[j] = (filtered_points[j,0] - min_val[0])/ math.sqrt(norm)
        
        for j in range(filtered_points.shape[0]):
            filtered_indexes[i,j] = convexhull_idx[i][~np.isnan(convexhull_idx[i])][np.argsort(shifted_points)[j]]
    return filtered_indexes

@nb.njit('b1(f8[:,:], f8[:])')
def PointPolySingle(poly_vertices, point):
    # takes the cross product of the vectors formed as follows
    # (polyvertex[j+1] - polyvertex[j]) x (target_point[i] - vertex[j]) 
    x1, y1 = poly_vertices[0]
    x2, y2 = poly_vertices[1]
    x3, y3 = point
    cross = (x2-x1) * (y3-y1) - (y2-y1) * (x3-x1)
    # get the sign of the cross product
    if cross < 0:
        truth = True
    else:
        truth = False
    # need to add a case for if it is ~zero -> on/ very close to the boundary
    # go through each vertex of the polygon and check the sign of the cross product
    for j in range(1,poly_vertices.shape[0],1):
        x1, y1 = poly_vertices[j]
        x2, y2 = poly_vertices[(j+1) % poly_vertices.shape[0]]
        x3, y3 = point
        cross = (x2-x1) * (y3-y1) - (y2-y1) * (x3-x1)
        if cross < 0:
            truth2 = truth ^ True
            if truth2 == True:
                # if the sign flips -> outside of convex polygon
                # update the correct mask index to remove this point
                return False
        else:
            truth2 = truth ^ False
            if truth2 == True:
                # if the sign flips -> outside of convex polygon
                # update the correct mask index to remove this point
                return False
    return True

@nb.njit('b1[:](f8[:,:], f8[:,:], u2)', parallel = True)
def bounding_box(point_cloud, extrema, processes):
    mask = np.zeros(point_cloud.shape[0]).astype(np.bool_)
    chunk = math.floor(point_cloud.shape[0]/processes)
    for core in nb.prange(processes+1):
        for i in range(chunk):
            idx = core*chunk + i
            if idx == point_cloud.shape[0]:
                break
            if (point_cloud[idx,1] <= extrema[1,1]) & (point_cloud[idx,1] >= extrema[0,1]):
                mask[idx] = True
            if (point_cloud[idx,0] >= extrema[1,0]) | (point_cloud[idx,0] <= extrema[0,0]):
                mask[idx] = False 
    return mask

@nb.njit('b1[:](f8[:,:], f8[:,:])', parallel = False)
def bounding_box_(point_cloud, extrema):
    mask = np.zeros(point_cloud.shape[0]).astype(np.bool_)
    for i in range(point_cloud.shape[0]):
        if (point_cloud[i,1] <= extrema[1,1]) & (point_cloud[i,1] >= extrema[0,1]):
            mask[i] = True
        if (point_cloud[i,0] >= extrema[1,0]) | (point_cloud[i,0] <= extrema[0,0]):
            mask[i] = False
    return mask

@nb.njit('b1[:](f8[:,:], f8[:,:], f8[:,:])', parallel = True)
def PointPoly(point_cloud, poly_vertex, extrema): 
    '''
    Function takes a point cloud and the vertexes of a convex polygon and returns the points where the point cloud lies inside the polygon.
    '''

    # this creates a bounding rectangle around the convex shape and filters points outside the bounding rectangle -> faster than doing every point
    processes = 256
    mask = bounding_box(point_cloud, extrema, processes)
    # Iteration happens then over the reduced set of points
    reduced_points = point_cloud[mask]

    # need to create a template index array for accessing the correct indexes in mask while using a reduced array
    red_idx = np.arange(mask.shape[0])[mask]
    chunk = math.floor(reduced_points.shape[0]/processes)
    for core in nb.prange(processes+1):
        for i in range(chunk):
            idx = core*chunk + i
            if idx == reduced_points.shape[0]:
                break
            mask[red_idx[idx]] = PointPolySingle(poly_vertex, reduced_points[idx])
            
    return mask

@nb.njit('b1[:](f8[:,:], f8[:,:], f8[:,:])')
def PointPoly_(point_cloud, poly_vertex, extrema): 
    '''
    Function takes a point cloud and the vertexes of a convex polygon and returns the points where the point cloud lies inside the polygon.
    '''
    # this creates a bounding rectangle around the convex shape and filters points outside the bounding rectangle -> faster than doing every point
    mask = bounding_box_(point_cloud, extrema)
    # Iteration happens then over the reduced set of points
    reduced_points = point_cloud[mask]

    # need to create a template index array for accessing the correct indexes in mask while using a reduced array
    red_idx = np.arange(mask.shape[0])[mask]
    for i in range(reduced_points.shape[0]):
        mask[red_idx[i]] = PointPolySingle(poly_vertex, reduced_points[i])
            
    return mask

@nb.njit('b1[:](f8[:,:],f8[:,:], f8[:,:], f8[:,:,:])')
def MultiPointPoly_(test_pointcloud: np.array, bound_pointcloud: np.array, ordered_indexes: np.array, hulls_extrema):
    '''
    Takes a point cloud in a 2D plane and identifies points that lie inside multiple convex hulls.

    test_pointcloud -> the points to bounds check
    bound_pointcloud -> the bounding shape
    ordered_indexes -> ordered sets of indexes that extract the convex hulls from bound_pointcloud
    '''
    tmask = np.zeros(test_pointcloud.shape[0]).astype(np.bool_)
    # goes through each convex polygon
    for poly in nb.prange(ordered_indexes.shape[0]):
        ordered_points = bound_pointcloud[(ordered_indexes[poly][~np.isnan(ordered_indexes[poly])]).astype(np.int32)]
        # if there isn't a polygon skip over
        if ordered_points.size > 0:
            mask = PointPoly_(test_pointcloud, ordered_points, hulls_extrema[poly])
            # updates the overall mask to include points within the boundaries
            for j in range(tmask.shape[0]):
                # mask[j] | tmask[j]
                tmask[j] = mask[j] | tmask[j]
    return tmask

@nb.njit('b1[:](f8[:,:],f8[:,:], f8[:,:], f8[:,:,:])', parallel = True)
def MultiPointPoly(test_pointcloud: np.array, bound_pointcloud: np.array, ordered_indexes: np.array, hulls_extrema):
    '''
    Takes a point cloud in a 2D plane and identifies points that lie inside multiple convex hulls.

    test_pointcloud -> the points to bounds check
    bound_pointcloud -> the bounding shape
    ordered_indexes -> ordered sets of indexes that extract the convex hulls from bound_pointcloud
    '''
    tmask = np.zeros(test_pointcloud.shape[0]).astype(np.bool_)
    # goes through each convex polygon
    for poly in nb.prange(ordered_indexes.shape[0]):
        ordered_points = bound_pointcloud[(ordered_indexes[poly][~np.isnan(ordered_indexes[poly])]).astype(np.int32)]
        # if there isn't a polygon skip over
        if ordered_points.size > 0:
            mask = PointPoly_(test_pointcloud, ordered_points, hulls_extrema[poly])
            # updates the overall mask to include points within the boundaries
            for j in range(tmask.shape[0]):
                # mask[j] | tmask[j]
                tmask[j] = tmask[j] | mask[j]
    return tmask

def gen_argmax(bound_pointcloud: np.array, ordered_indexes: np.array):
        max_points = np.zeros((ordered_indexes.shape[0], 2, 2))
        for poly in range(ordered_indexes.shape[0]):
            ordered_points = bound_pointcloud[(ordered_indexes[poly][~np.isnan(ordered_indexes[poly])]).astype(np.int32)]
            if ordered_points.size > 0:
                for j in range(max_points.shape[-1]):
                    max_points[poly, 0, j] = ordered_points[np.argmin(ordered_points[:,j])][j]
                    max_points[poly, 1, j] = ordered_points[np.argmax(ordered_points[:,j])][j]
            else:
                for j in range(max_points.shape[-1]):
                    max_points[poly, 0, j] = np.nan
                    max_points[poly, 1, j] = np.nan
        return max_points

def gen_overall_hull(point_cloud):
    convex_hull = ConvexHull(point_cloud)
    max_points = np.zeros((2, 2))
    for j in range(max_points.shape[-1]):
        max_points[0, j] = point_cloud[convex_hull.vertices][np.argmin(point_cloud[convex_hull.vertices][:,j])][j]
        max_points[1, j] = point_cloud[convex_hull.vertices][np.argmax(point_cloud[convex_hull.vertices][:,j])][j]
    return convex_hull.vertices, max_points

