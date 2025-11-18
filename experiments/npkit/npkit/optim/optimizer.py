class BaseOptimizer(object):
    
    def step(self, loss=None, grad=None):
        raise NotImplementedError
    
    def __str__(self):
        return self.__name__
    
    def __repr__(self):
        return self.__name__