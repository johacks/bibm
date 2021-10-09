import numpy as np
from NPGA.util import IsNonDominatedableFast
from NPGA import NichedParetoGeneticAlgorithm


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


class DiversitySpacing:
    def __init__(self):
        pass

    def runMetric(self, statistics):
        points = []
        for point in statistics.ParetoSet:
            p = np.array([[point.Fitness[0], point.Fitness[1]]])
            points.append(p)
        points = np.array(points)
        distances = np.linalg.norm(points - points[:, None], axis=-1, ord=1)
        np.fill_diagonal(distances, np.Inf)
        minDistances = np.min(distances, axis=0)
        meanDistance = np.mean(minDistances)
        return np.sqrt(
                np.sum(np.power(np.subtract(minDistances, meanDistance), 2)) /
                len(minDistances))
                

class DiversityMaximunExtension:
    def __init__(self):
        pass

    def runMetric(self, statistics):
        points = []

        for point in statistics.ParetoSet:
            p = np.array([point.Fitness[0], point.Fitness[0]])
            points.append(p)

        points = np.array(points)

        # Get max and min fitness for function
        maxim_points = []
        min_points = []
        for i in range(points.shape[1]):
            index_max = np.where(points==np.max(points[:,i]))
            index_min = np.where(points==np.max(points[:,i]))
            
            maxim_points.append(points[index_max])
            min_points.append(points[index_min])

        # Calculate metric
        count = 0
        for i in range(len(maxim_points)):
            count += (maxim_points[i] - min_points[i])[0]**2

        return np.sqrt(count)

        
        