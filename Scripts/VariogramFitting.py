#%%
import numpy as np
from scipy.optimize import curve_fit
from numba import jit

@jit(nopython=True)
def calculate_distances_and_differences(x, y, values):
    """
    Calculate pairwise distances and differences for variogram calculation.

    Parameters:
    - x (array-like): x-coordinates of the points.
    - y (array-like): y-coordinates of the points.
    - values (array-like): Values at the corresponding points.

    Returns:
    - distances (array-like): Pairwise distances between points.
    - differences (array-like): Squared differences between values of points.
    """
    n_points = len(values)
    num_pairs = n_points * (n_points - 1) // 2
    distances = np.empty(num_pairs, dtype=np.float64)
    differences = np.empty(num_pairs, dtype=np.float64)
    
    index = 0
    for i in range(n_points):
        for j in range(i + 1, n_points):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            distances[index] = np.sqrt(dx * dx + dy * dy)  # Euclidean distance
            differences[index] = (values[i] - values[j]) ** 2
            index += 1

    return distances, differences

@jit(nopython=True)
def calculate_binned_statistics(distances, differences, bin_edges):
    """
    Calculate mean semivariance for each bin of distances.

    Parameters:
    - distances (array-like): Pairwise distances between points.
    - differences (array-like): Squared differences between values of points.
    - bin_edges (array-like): Edges of distance bins.

    Returns:
    - binned_distances (array-like): Mean distance of each bin.
    - binned_semivariance (array-like): Mean semivariance of each bin.
    """
    num_bins = len(bin_edges) - 1
    binned_distances = np.zeros(num_bins, dtype=np.float64)
    binned_semivariance = np.zeros(num_bins, dtype=np.float64)
    counts = np.zeros(num_bins, dtype=np.int64)

    for i in range(len(distances)):
        for bin_idx in range(num_bins):
            if bin_edges[bin_idx] <= distances[i] < bin_edges[bin_idx + 1]:
                binned_distances[bin_idx] += distances[i]
                binned_semivariance[bin_idx] += differences[i] / 2.0
                counts[bin_idx] += 1
                break

    for bin_idx in range(num_bins):
        if counts[bin_idx] > 0:
            binned_distances[bin_idx] /= counts[bin_idx]
            binned_semivariance[bin_idx] /= counts[bin_idx]

    return binned_distances, binned_semivariance

def gaussian_variogram_model(h, sill, range_):
    """Gaussian variogram model."""
    return sill * (1 - np.exp(-((h / range_) ** 2)))

def fit_gaussian_variogram(x, y, values, use_median=False, num_bins=20):
    """
    Fit a Gaussian variogram to the mean/median of binned values.

    Parameters:
    - x (array-like): x-coordinates of the data points.
    - y (array-like): y-coordinates of the data points.
    - values (array-like): Values at the corresponding points (x, y).
    - use_median (bool): Whether to use the median for binning.
    - num_bins (int): Number of bins for distance intervals.

    Returns:
    - float: Correlation length (range) of the fitted Gaussian variogram.
    """
    # Calculate distances and differences
    distances, differences = calculate_distances_and_differences(x, y, values)

    # Bin the distances
    bin_edges = np.linspace(0, distances.max(), num_bins + 1)
    binned_distances, binned_semivariance = calculate_binned_statistics(
        distances, differences, bin_edges
    )

    # Remove empty bins (distances or semivariances with zero counts)
    valid_bins = ~np.isnan(binned_distances) & ~np.isnan(binned_semivariance)
    binned_distances = binned_distances[valid_bins]
    binned_semivariance = binned_semivariance[valid_bins]

    # Fit the Gaussian variogram model
    initial_guess = [binned_semivariance.max(), binned_distances.mean()]  # [sill, range]
    popt, _ = curve_fit(gaussian_variogram_model, binned_distances, binned_semivariance, p0=initial_guess, bounds=(0, np.inf))

    sill, range_ = popt
    return range_