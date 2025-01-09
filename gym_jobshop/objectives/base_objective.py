class Objective:
    def evaluate(self, solution):
        pass

class CompositeObjective(Objective):
    def __init__(self, objectives, weights):
        self.objectives = objectives
        self.weights = weights

    def evaluate(self, solution):
        return sum(w * obj.evaluate(solution) for obj, w in zip(self.objectives, self.weights))
