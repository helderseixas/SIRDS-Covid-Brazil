import seaborn as sns
import numpy as np

def centimeter_to_inch(centimeters):
    return centimeters * 1/2.54

def get_default_colors_categorical_seaborn(n=5):
    colors_custom = ['#FEC4DC', '#3F80B3', '#610099', '#CCA86C', '#91E693', 'lightgray']
    pallet_custom = sns.color_palette(colors_custom[:n])
    return pallet_custom

def calculate_confidence_interval(samples):
    # Number of bootstrap resamples
    num_resamples = 1000

    # Create an array to store resampled means
    resample_means = np.zeros(num_resamples)

    # Perform bootstrapping
    for i in range(num_resamples):
        resample = np.random.choice(samples, size=len(samples), replace=True)
        resample_means[i] = np.mean(resample)

    # Calculate confidence interval
    lower_bound = np.percentile(resample_means, 2.5)
    upper_bound = np.percentile(resample_means, 97.5)

    return lower_bound, upper_bound
