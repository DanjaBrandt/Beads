
class BeadAnalysisPipeline:
    def __init__(self, segmenter, simulator, comparator, optimizer):
        self.segmenter = segmenter
        self.simulator = simulator
        self.comparator = comparator
        self.optimizer = optimizer

    def run(self, image_stacks_dict, initial_params, timepoints):
        # 1. Segment experimental shapes
        experimental_shapes = {
            t: self.segmenter.segment(stack)
            for t, stack in image_stacks_dict.items()
        }

        # 2. Optimize material parameters
        #best_params = self.optimizer.optimize(experimental_shapes, initial_params)

        #return best_params