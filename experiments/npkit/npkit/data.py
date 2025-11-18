import numpy as np

class NumpyDataLoader:

    def __init__(self, X, y, batch_size: int, shuffle: bool = True, drop_last: bool = False):

        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

        assert self.X.shape[0] == len(self.y), "X and y must have the same number of samples"

    def __iter__(self):
        self.indices = np.arange(self.X.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0
        return self

    def __next__(self):
        if self.current_idx >= self.X.shape[0]:
            raise StopIteration
        
        batch_indices = self.indices[self.current_idx:self.current_idx + self.batch_size]
        if self.drop_last and len(batch_indices) < self.batch_size:
            raise StopIteration
        
        batch_X = self.X[batch_indices]
        batch_y = self.y[batch_indices]

        self.current_idx += self.batch_size
        return batch_X, batch_y
    
    def __len__(self):
        if self.drop_last:
            return self.X.shape[0] // self.batch_size
        else:
            return (self.X.shape[0] + self.batch_size - 1) // self.batch_size