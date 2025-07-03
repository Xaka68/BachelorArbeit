import numpy as np

def should_stop(history, threshold=0.001):
    if len(history['accuracy']) < 2:
        return False
    return np.abs(history['accuracy'][-1] - history['accuracy'][-2]) < threshold

