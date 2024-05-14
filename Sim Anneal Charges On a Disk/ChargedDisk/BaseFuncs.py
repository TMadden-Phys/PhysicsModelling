import numpy as np
# so it updates in main

class Single_pass:
    ''' The single pass code is deprecated and was used to run a singular simulation
    The Mult_pass class contains the updated code which runs multiple simulations in parallel.
    This code is not commented as much as it is not used for the main simulations.'''
    def gen_coords(disk_radius: float, num_charges: int):
        # generates a 2D array of randomised polar coordinates distributing the charge around the disk
        polar_arr = np.random.rand(2, num_charges) * \
            np.array((disk_radius, 2*np.pi))[:, np.newaxis]
        angle = polar_arr[1].reshape(num_charges, 1)
        # computes the xy coordinates of the charges
        xy = polar_arr[0, :, np.newaxis]*np.hstack((np.cos(angle), np.sin(angle)))
        return xy

    def Choose_Charge(coords, delta, disk_radius):
        in_bounds = False
        while in_bounds == False:
            # gets a random charge index
            index = np.random.randint(0, coords.shape[0])
            # extracts the singular charge data
            charge_coord = coords[index].copy()
            charge_coord += delta*np.random.uniform(-1, 1, size=2)
            in_bounds = np.where(np.linalg.norm(charge_coord, axis=-1) > disk_radius, False, True)
        # creates the new array that can be used to calc energy
        coords_clone = coords.copy()
        coords_clone[index] = charge_coord
        return coords_clone

    def calc_energy(coords, charge=1):
        # this is unlikely to be a bottleneck so halving the computation time here probably isn't necessary
        vect_diff = Single_pass.vector_difference(coords)
        # find the indices
        i, j = np.indices(vect_diff.shape)
        # takes the upper triangular values removing double counts
        distances = vect_diff[j>=i]
        return (charge**2) * np.sum(1/distances)

    def Accept_change(coords_chg, coords, temp):
        orig_energy = Single_pass.calc_energy(coords)
        new_energy = Single_pass.calc_energy(coords_chg)
        energy_diff = new_energy - orig_energy
        
        if energy_diff < 0:
            return coords_chg, new_energy
        if energy_diff > 0:
            if np.random.rand() < np.exp(-energy_diff/temp):
                return coords_chg, new_energy
            else:
                return coords, orig_energy

    def vector_difference(charge_coords):
        # normally the dims will be 2 but could extend to higher dimensions
        charges, dims = charge_coords.shape
        template = np.empty((charges,charges-1, dims))
        # basic loop
        for i in range(charges):
            template[i] = np.delete(charge_coords,i,0)
        subtraction = np.expand_dims(charge_coords,1) - template
        # returns an array of shape (no. of charges, dimensions)
        return np.linalg.norm(subtraction,axis=-1)


class Mult_pass:
    '''Class containing the basic functions for the simulation, this code allows multiple simulations to be run in parallel'''
    def gen_coords(re_runs,disk_radius: float, num_charges: int):
        '''generates a 3D array of randomised polar coordinates distributing the charges around the disk and initialising multiple
        simulations'''
        polar_arr = np.random.rand(re_runs,2, num_charges) * \
            np.array((disk_radius, 2*np.pi))[..., np.newaxis]
        # extracts the angles from the polar array
        angle = polar_arr[:,1]
        # computes the xy coordinates of the charges in each simulation
        xy = polar_arr[:,0, :, np.newaxis]*np.stack((np.cos(angle), np.sin(angle)), axis = -1)
        return xy

    def Choose_Charge(coords, delta, disk_radius):
        ''' chooses a random charge from each simulation and moves it randomly in some direction.
        Also checks each move to make sure that the charges are not moving outside the radius of the disk.'''
        re_runs = coords.shape[0]
        num_charges = coords.shape[1]
        index = np.random.randint(0, num_charges,size = re_runs) # gets a random charge index in each simulation
        charge_coord = coords[np.arange(re_runs),index].copy() # extracts the singular charge data
        move = delta*np.random.uniform(-1, 1, size=charge_coord.shape) # generates a random move for each charge
        charge_coord += move # moves the charge
        in_bounds = np.where(np.linalg.norm(charge_coord, axis=-1) > disk_radius) # checks to see if the coordinates are outisde of the disk
        while charge_coord[in_bounds].shape[0] > 0: # while the move would mean that a charge is outside the disk
            index[in_bounds] = np.random.randint(0, num_charges,size = len(in_bounds)) # chooses another random charge index for the simualtion where the charge is outside the disk
            charge_coord[in_bounds] = coords[in_bounds, index[in_bounds]].copy() # extracts the new charge coordinates data from the array of charge coordinates
            charge_coord[in_bounds] += delta*np.random.uniform(-1, 1, size=charge_coord[in_bounds].shape) # moves the charges with a new random move 
            in_bounds = np.where(np.linalg.norm(charge_coord, axis=-1) > disk_radius) # checks whether the new move is outside of the radius

        temp = coords.copy() # creates a template array
        temp[np.arange(re_runs),index] = charge_coord.copy() # places the updated charge coordinates into the array
        return temp # return the new charge coordinates

    def calc_energy(coords, charge=1):
        ''' Calculates the energy of a state in each simulation'''
        # this is unlikely to be a bottleneck so halving the computation time here probably isn't necessary
        # it is vectorised calculation so I could cache results and use that
        # calculates the vector difference in cartesian coordinates between all the charges
        vect_diff = Mult_pass.vector_difference(coords)
        # find the indices of the array, can then slice the arrya using these indices
        indices = np.indices(vect_diff.shape)
        j,k = indices[-2], indices[-1]
        # this removes double counts and then reshapes the slice into the correct simulations
        # basically taking the upper traingular of the matrices generated
        distances = vect_diff[k>=j].reshape(vect_diff.shape[0], -1)
        # then returns the energy formula, a sum of 1/r across the last axis of the 2D array.
        # the half is not needed bcause I already removed the double counts in the line above
        return (charge**2) * np.sum(1/distances, axis = -1)

    def vector_difference(charge_coords):
        '''Function to calculate the vector distance of each charge with all the other charges expanded to calculate across all parallel simulations'''
        # extracts the size of the array from it's shape
        re_runs, charges, dims = charge_coords.shape
        # creates an empty template matrix in which to place the results
        # the 3rd axis being length charge-1 is to remove the zeros getting the vector difference
        # of the charge with itself
        template = np.empty((re_runs, charges,charges-1, dims))
        # loops over the charges, this could maybe be vectorised, or use numba and loop lifting but works fine, it is array manipulation
        # rather than intense computation so shouldn't cause too much drain on resources
        for i in range(charges):
            # vectorised across all simulations
            # take each element across the second axis and place the charge coordinates
            # with the charge coordinate corresponding to that element deleted
            # i.e. remove the coordinates of the first charge in the first element along that axis etc...
            template[:,i] = np.delete(charge_coords,i,1)
        # expand the dimesnion of the charge coords so that it can be correctly matche to the shape of the template
        # here we are calculating r_1 - r_2 or the distance of each charge from the other charges
        subtraction = np.expand_dims(charge_coords,2) - template
        # then returns the norm of the vectors, or the distance between all the charges
        return np.linalg.norm(subtraction,axis=-1)

    def Accept_change(coords_chg, coords, temperature):
        ''' Function to accept Lower and Higher energy states calculating the 
        chance that higher energy states will be accepted.
        Input - the changed coords with a moved charge, original coords of the charges and the temperature value
        output - An Array of all the simulations where the changes have been accepted or rejected.'''

        # calculates the new and old energies
        orig_energy = Mult_pass.calc_energy(coords)
        new_energy = Mult_pass.calc_energy(coords_chg)
        # calculates the energy difference
        energy_diff = new_energy - orig_energy
        # creates a template array in which to place the results
        temp_arr = np.zeros(coords.shape)
        indexes = np.where(energy_diff < 0)[0] # obtains the indexes where the energy has decreased
        temp_arr[indexes] = coords_chg[indexes] # places the changed coords corresponding to the simulations where the energy has decreased
        index_energ = np.where(energy_diff > 0)[0] # finds the indexes where the energy has increased
        test = np.random.rand(index_energ.shape[0]) # generates a 1D array of random values to compare to the temperature equation 
        indexes = np.where(test < np.exp(-energy_diff[index_energ]/temperature)) # obtains the indexes where the random value is less than the temperature equation
        temp_arr[index_energ[indexes]] = coords_chg[index_energ[indexes]]# inserts the relevant changed coordinates into the template array
        indexes = np.where(test > np.exp(-energy_diff[index_energ]/temperature)) # finds the indexes where the random value is more than the temperature equation - doesn't satisfy conditions
        temp_arr[index_energ[indexes]] = coords[index_energ[indexes]] # inserts the original coords into the simulations where the condition to accept higher energy was not satisfied
        return temp_arr, Mult_pass.calc_energy(temp_arr) # returns the template array with the correct results for the pass of the simulation and the current energy of each simulation