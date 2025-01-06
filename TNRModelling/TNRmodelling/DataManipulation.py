import numpy as np

def hist_data(timesteps, temperatures, model_time, bin_width = 0.25):
    if len(temperatures.shape) > 1:
        lower_bound = 0
        upper_range = int(np.ceil(model_time))
        iterations_bins = int(np.ceil(upper_range/bin_width))
        count = np.zeros((temperatures.shape[0],iterations_bins))
        for j in range(temperatures.shape[0]):
            for i in range(iterations_bins):
                count[j,i] = np.sum(temperatures[j][(timesteps> lower_bound) & (timesteps < (i+1)*bin_width)])
                lower_bound = i*bin_width
    else:
        lower_bound = 0
        upper_range = int(np.ceil(model_time))
        iterations_bins = int(np.ceil(upper_range/bin_width))
        count = np.zeros(iterations_bins)
        for i in range(iterations_bins):
            count[i] = np.sum(temperatures[(timesteps> lower_bound) & (timesteps < (i+1)*bin_width)])
            lower_bound = i*bin_width
    return np.arange(iterations_bins)*bin_width, count

def relative_values(counts: np.ndarray[np.int32, np.int32]):
    max_flux = np.max(counts, axis = -1)
    relative_counts = counts/max_flux[:,None]
    return relative_counts

def time_range(relative_counts, timesteps, threshold):
    min_max = np.empty((relative_counts.shape[0],2))
    for i in range(min_max.shape[0]):
        min_max[i, 0] = np.min(timesteps[(relative_counts[i]) > threshold]); min_max[i, 1] = np.max(timesteps[(relative_counts[i]) > threshold])
    return min_max