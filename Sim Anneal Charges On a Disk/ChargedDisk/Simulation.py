from . import BaseFuncs
from .BaseFuncs import Single_pass as SP
from .BaseFuncs import Mult_pass as MP
import numpy as np

# so it updates in main 
def Run_Sim_SP(disk_radius, num_charges, delta, temp, charge_movements, decimals=6):
    '''This function runs the single pass base functions and runs a singular simulation, 
    this code is deprecated and the vectorised version is used instead. comments will be made on the vectorised version'''
    energy_lst = []
    charges = SP.gen_coords(disk_radius, num_charges)
    temp_coords = SP.Choose_Charge(charges, delta, disk_radius)
    updated_coords, energy = SP.Accept_change(temp_coords, charges, temp)

    energy_lst.append(energy)
    iters = 1
    while True:
        # each loop here is a reduction of temperature
        for j in range(charge_movements):
            # each loop here moves a charge.
            temp_coords = SP.Choose_Charge(updated_coords, delta, disk_radius)

            updated_coords, energy = SP.Accept_change(
                temp_coords, updated_coords, temp)
        energy_lst.append(energy)
        
        if iters % 100 == 0:
            # print(energy, (round(energ_lst[iters], 8) == round(energ_lst[iters-100], 8)))
            if (round(energy_lst[iters], decimals) == round(energy_lst[iters-100], decimals)):
                break
        
        iters += 1
        
        temp = temp * 0.96
    
    return updated_coords, energy_lst, iters

def Run_Sim_MP(disk_radius, num_charges, delta, temp, charge_movements, runs = 3, decimals=6,reduction = 0.96):
    ''' Function to run the vectorised base functions for multiple simulations. '''
    completed_sims_arr = np.zeros(runs) # creates an array to check when all the simulations have finished
    charges = MP.gen_coords(runs, disk_radius, num_charges) # generates the charges and their coordinates
    temp_coords = MP.Choose_Charge(charges, delta, disk_radius) # creates a temporary array of coordinates with the moved charge
    updated_coords, energy = MP.Accept_change(temp_coords, charges, temp) # updates the coordinates and energy after checking if the conditions are satisfied

    energy_lst = energy[:,np.newaxis] # creates an array of energy values for each simulation
    iters = 1 # sets the iterations to 1 as one pass has been made
    final_coords = np.empty(updated_coords.shape) # creates a template array in which to place the final coordinates
    prev_sum = 0
    while True:
        # each loop of the while statement here is a reduction of temperature
        # runs until a condition is met
        for j in range(charge_movements):
            # each loop here moves a charge.
            temp_coords = MP.Choose_Charge(updated_coords, delta, disk_radius) # create a temporary array of the new coordinates
            updated_coords, energy = MP.Accept_change(
                temp_coords, updated_coords, temp) # decides to accept or reject the new coordinates
        energy_lst = np.append(energy_lst, energy[:,np.newaxis], axis = -1) # appends the energy after each iteration to the energies array
        
        if iters % 100 == 0: # checks every 100 iterations, arbitrary choice but gives good results
            indexes = np.where(\
                np.round(energy_lst[:,iters], decimals = decimals)\
                      == np.round(energy_lst[:,iters-100]\
                        , decimals = decimals))[0] # finds the indexes where the energy has not changed to 'decimals' decimal places
            completed_sims_arr[indexes] = 1 # updates the array to say that a simulation has finished
            final_coords[indexes] = updated_coords[indexes] # places the run simulation coordinates into a final array
            if np.sum(completed_sims_arr) == runs:
                # if the sum of the completed sims array is equal to the number of simulations running
                # all the simulations have run, so the program stops
                break
        if iters % 500 == 0:
            if np.sum(completed_sims_arr) > prev_sum:
                # used to update me on the progress of the simulations
                print('{} out of {} simulations completed.'.format(np.sum(completed_sims_arr), runs))
                prev_sum = np.sum(completed_sims_arr)
        iters += 1 # increase the number of iterations
        temp = temp * reduction # reduce the temperature
    
    return final_coords, energy_lst, iters # return the final state coordinates, the list of energies and the number of iterations