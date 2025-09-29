
class ParameterOptimizer:
    def __init__(self, simulator, comparator, optimizer_config):
        self.simulator = simulator
        self.comparator = comparator
        self.optimizer_config = optimizer_config

    def optimize(self, experimental_shapes, initial_params):
        # iterative optimization loop (e.g., scipy.optimize)
        # return best parameters
        pass