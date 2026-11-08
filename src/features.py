
import numpy as np


def extract_features(signals):
    """
    signals: (number of windows, 6 axes, 128 samples)
    returns: (number of windows, 24 features)
    """

    # Average sensor reading
    means = np.mean(signals, axis=2)

    # How much readings fluctuate
    stds = np.std(signals, axis=2)

    # Overall signal strength
    rms = np.sqrt(np.mean(signals ** 2, axis=2))

    # Difference between highest and lowest readings
    ranges = np.ptp(signals, axis=2)

    # Combine the four measurements for all six axes
    features = np.concatenate(
        [means, stds, rms, ranges],
        axis=1
    )

    return features
