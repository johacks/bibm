import numpy as np
from NPGA.util import IsNonDominatedableFast


class ErrorProportion:
    def __init__(self, fnFrontier, epsilon=0.001):
        self.fnFrontier = fnFrontier
        self.epsilon = epsilon

    def runMetric(self, statistics):
        errors = 0
        for sol in statistics.ParetoSet:
            frontierPoint = self.fnFrontier(sol.Fitness[0])
            inFrontier = abs(frontierPoint - sol.Fitness[1]) <= self.epsilon
            errors += (0 if inFrontier else 1)
        return errors / len(statistics.ParetoSet)

    def __str__(self) -> str:
        return "Proporción de error"


class Coverage:
    def __init__(self, coveringSet):
        self.coveringSet = coveringSet

    def runMetric(self, statistics):
        points = np.zeros((len(self.coveringSet) + len(statistics.ParetoSet)),
                          dtype=float)
        points[:len(statistics.ParetoSet)] = np.array(statistics.ParetoSet)
        points[len(statistics.ParetoSet):] = np.array(self.coveringSet)
        dominatedPoints = IsNonDominatedableFast(points)
        dominatedPoints = dominatedPoints[:len(statistics.ParetoSet)]
        return np.count_nonzero(dominatedPoints == bool(False)) / len(
            dominatedPoints)


class GenerationalDistance:
    def __init__(self, frontier, p=2):
        self.p = p
        self.frontier = np.array(frontier)
 
    def runMetric(self, statistics):
        distances = []
        for point in statistics.ParetoSet:
            p = np.array([[point.Fitness[0], point.Fitness[1]]])
            distances.append(np.min(np.linalg.norm(p - self.frontier, axis=1)))
        distances = np.array(distances)
        return (np.power(np.sum(np.power(distances, self.p)), 1 / self.p) /
                len(distances))


class MaximumParetoFrontierError:
    def __init__(self, frontier):
        self.frontier = np.array(frontier)

    def runMetric(self, statistics):
        distances = []
        for point in statistics.ParetoSet:
            p = np.array([[point.Fitness[0], point.Fitness[1]]])
            distances.append(np.min(np.linalg.norm(p - self.frontier, axis=1)))
        distances = np.array(distances)
        return np.max(distances)
