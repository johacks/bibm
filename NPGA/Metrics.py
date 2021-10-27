import numpy as np

# Convergence metrics

# Discarded
"""
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
"""


class GenerationalDistance:
    def __init__(self, frontier, p=2):
        self.p = p
        self.frontier = np.array(frontier)

    def runMetric(self, statistics):
        # Collect points in estimated pareto frontier
        ps = np.array([np.array(p.Fitness) for p in statistics.ParetoSet])
        # Minimum distances between estimated and real pareto front
        ds = np.min(np.linalg.norm(ps-self.frontier[:, None], axis=-1), axis=0)
        return np.power(np.sum(np.power(ds, self.p)), 1 / self.p) / len(ds)


# Diversity metrics


class Spacing:
    def __init__(self):
        pass

    def runMetric(self, statistics):
        # Collect points in estimated pareto frontier
        points = np.array([np.array(p.Fitness) for p in statistics.ParetoSet])
        # Compute minimum (Manhattan) distances between these points
        distances = np.linalg.norm(points - points[:, None], axis=-1, ord=1)
        np.fill_diagonal(distances, np.Inf)
        # Standard deviation of minimum distances
        return np.std(np.min(distances, axis=0))


# Discarded
"""
class DiversityMaximumExtension:
    def __init__(self):
        pass

    def runMetric(self, statistics):
        points = []

        for point in statistics.ParetoSet:
            p = np.array([point.Fitness[0], point.Fitness[1]])
            points.append(p)

        points = np.array(points)

        # Get max and min fitness for function
        maxim_points = []
        min_points = []
        for i in range(points.shape[1]):
            index_max = np.argmax(points[:,i])
            index_min = np.argmin(points[:,i])

            maxim_points.append(points[index_max])
            min_points.append(points[index_min])

        # Calculate metric
        count = 0
        for i in range(len(maxim_points)):
            count += (maxim_points[i] - min_points[i])[0]**2

        return np.sqrt(count)
"""


class Extension:
    def __init__(self, frontier):
        self.frontier = np.array(frontier)

    def runMetric(self, statistics):
        # Collect points in estimated pareto frontier
        points = np.array([np.array(p.Fitness) for p in statistics.ParetoSet])
        # Compute extreme points in known frontier
        frontier_f1_ext = self.frontier[np.argmin(self.frontier[:, 0])]
        frontier_f2_ext = self.frontier[np.argmin(self.frontier[:, 1])]
        # Compute extreme points in solution pareto
        pts_f1_ext_idx = np.argmin(points[:, 0])
        pts_f2_ext_idx = np.argmin(points[:, 1])
        # Compute distances between extremes
        d1_e = np.linalg.norm(frontier_f1_ext-points[pts_f1_ext_idx], ord=1)
        d2_e = np.linalg.norm(frontier_f2_ext-points[pts_f2_ext_idx], ord=1)
        d_e = d1_e + d2_e
        # Compute rest of distances
        # Compute minimum (Manhattan) distances between these points
        mindistances = np.linalg.norm(points - points[:, None], axis=-1, ord=1)
        np.fill_diagonal(mindistances, np.Inf)
        curr_point = pts_f1_ext_idx
        mindistances_aux = np.copy(mindistances)
        mindistances_aux[:, pts_f2_ext_idx] = np.Inf
        distances = []
        curr_point = pts_f1_ext_idx
        while len(distances) < len(points) - 2:
            next_point = np.argmin(mindistances_aux[curr_point])
            distances.append(mindistances_aux[curr_point, next_point])
            mindistances_aux[:, curr_point] = np.Inf
            curr_point = next_point
        distances.append(mindistances[curr_point, pts_f2_ext_idx])
        distances = np.array(distances)
        d_mean = np.mean(distances)
        return ((d_e + np.sum(np.abs(distances-d_mean))) /
                (d_e + len(points)*d_mean))

