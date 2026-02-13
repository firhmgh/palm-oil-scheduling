import numpy as np

def fifo_policy(blocks):
    """
    Pilih blok dengan buah siap panen terbanyak
    """
    return np.argmax([b['ready_fruit'] for b in blocks])

def oldest_first(blocks):
    """
    Pilih blok tertua
    """
    return np.argmax([b['age'] for b in blocks])
