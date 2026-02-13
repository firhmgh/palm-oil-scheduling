import numpy as np

def fifo_policy(blocks):
    """
    Heuristik 1: Pilih blok pertama yang memiliki buah (FIFO berdasarkan urutan blok)
    """
    for i, b in enumerate(blocks):
        if b['ready_fruit'] > 0:
            return i
    return 0 # Default ke blok 0 jika semua kosong

def oldest_first(blocks):
    """
    Heuristik 2: Pilih blok dengan umur (age) tertua (Prioritas Rendemen)
    """
    return int(np.argmax([b['age'] for b in blocks]))