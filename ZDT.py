import math
import NPGA
import matplotlib.pyplot as plt
import numpy as np
import textwrap
from PIL import Image
import os


class ZDTBase:

    def __init__(self) -> None:
        self.generation = 1

    gGlobalParetoValue = 1
    gLocalParetoValue = None
    gDeceptiveParetoValue = None
    x1_domain = [0, 1]
    x1_bits = 16
    xrest_domain = [0, 1]
    xrest_bits = 16
    m = 30

    f1_min_representation = 0
    f1_max_representation = 1
    f2_min_representation = 0
    f2_max_representation = 4

    @classmethod
    def f2(cls, x, f1):
        g = cls.g(x)
        h = cls.h(f1, g)
        return g * h

    @classmethod
    def globalParetoFront(cls, f1):
        g = cls.gGlobalParetoValue
        h = cls.h(f1, g)
        return g * h

    @classmethod
    def localParetoFront(cls, f1):
        g = cls.gLocalParetoValue
        h = cls.h(f1, g)
        return g * h

    @classmethod
    def deceptiveParetoFront(cls, f1):
        g = cls.gDeceptiveParetoValue
        h = cls.h(f1, g)
        return g * h

    @classmethod
    def eval(cls, x):
        f1 = cls.f1(x)
        return f1, cls.f2(x, f1)

    @classmethod
    def scaleMinMax(cls, x, xmin, xmax, mindesired, maxdesired):
        return ((x - xmin) / (xmax - xmin) * (maxdesired - mindesired)
                + mindesired)

    @classmethod
    def graytodec(cls, bin_list):
        """
        Convert from Gray coding to binary coding.
        We assume big endian encoding.
        """
        b = bin_list[0]
        d = int(b) * (2 ** (len(bin_list) - 1))
        for i, e in enumerate(range(len(bin_list) - 2, -1, -1)):
            b = str(int(b != bin_list[i + 1]))
            d += int(b) * (2 ** e)
        return d

    @classmethod
    def decodechromosome(cls, bits):
        x = np.zeros((cls.m,), dtype=np.float64)

        for i in range(cls.m):
            bitAmount = (cls.x1_bits, cls.xrest_bits)[int(i != 0)]
            var_domain = (cls.x1_domain, cls.xrest_domain)[int(i != 0)]
            max_current = math.pow(2, bitAmount) - 1
            dec = cls.graytodec(bits[:bitAmount])
            bits = bits[bitAmount:]
            x[i] = cls.scaleMinMax(dec, 0, max_current,
                                   var_domain[0], var_domain[1])
        return x

    @classmethod
    def getfitness(cls, candidate):
        x = cls.decodechromosome(candidate)
        F1, F2 = cls.eval(x)
        return [[F1, "minimize"], [F2, "minimize"]]

    def display(self, statistics):
        f1x = []
        f2x = []
        for point in statistics.ParetoSet:
            f1x.append(point.Fitness[0])
            f2x.append(point.Fitness[1])

        xpop = []
        ypop = []
        for individual in statistics.population:
            xpop.append(individual.Fitness[0])
            ypop.append(individual.Fitness[1])

        plt.figure(1)
        plt.clf()
        plt.axis([self.f1_min_representation, self.f1_max_representation,
                  self.f2_min_representation, self.f2_max_representation])
        plt.xlabel("F1(x)")
        plt.ylabel("F2(x)")
        plt.plot(xpop, ypop, "ko", label="individuals")
        plt.plot(f1x, f2x, "ro", label="pareto front")
        plt.title("Zitzler-Deb-Thiele's function {}  -  GENERATION: {}".format(
            self.problem_number, self.generation))

        f1Pareto = np.linspace(self.f1_min_representation,
                               self.f1_max_representation, 100)
        f2Pareto = np.array([self.globalParetoFront(x_i) for x_i in f1Pareto])
        plt.plot(f1Pareto, f2Pareto, '-g', label='Global Pareto-optimal Front')

        if self.gLocalParetoValue is not None:
            f2Pareto = np.array([self.localParetoFront(x_i)
                                 for x_i in f1Pareto])
            plt.plot(f1Pareto, f2Pareto, '-y',
                     label='Best local Pareto-optimal Front')

        if self.gDeceptiveParetoValue is not None:
            f2Pareto = np.array([self.deceptiveParetoFront(x_i)
                                 for x_i in f1Pareto])
            plt.plot(f1Pareto, f2Pareto, '-g',
                     label='Best deceptive Pareto-optimal Front')

        plt.legend(loc="upper center", bbox_to_anchor=(0.5, 1.00), shadow=True,
                   ncol=2)
        plt.grid()
        plt.draw()
        plt.savefig(f'images/{self.problem_number}/gen{self.generation}.png')
        plt.pause(0.0001)
        plt.show(block=False)

        self.generation = self.generation + 1

    def test(self, population_size=200, max_generation=400,
             crossover_rate=0.65, mutation_rate=1/170, niche_radius=0.02,
             candidate_size=4, t_dom_p=0.13):
        geneset = "01"
        genelen = [self.x1_bits + self.m * self.xrest_bits]

        def fnDisplay(statistic):
            self.display(statistic)

        def fnGetFitness(genes):
            return self.getfitness(genes)

        optimalFitness = [0, 0]

        GA = NPGA.NichedParetoGeneticAlgorithm(
            fnGetFitness,
            fnDisplay,
            optimalFitness,
            geneset,  # posibles valores de genes (alfabeto)
            genelen,  #
            population_size=population_size,
            max_generation=max_generation,
            crossover_rate=crossover_rate,
            mutation_rate=mutation_rate,
            niche_radius=niche_radius,
            candidate_size=candidate_size,
            prc_tournament_size=t_dom_p,
            fastmode=True,
        )
        for file in os.listdir(f'images/{self.problem_number}'):
            if file.endswith('.png'):
                os.remove(f'images/{self.problem_number}/{file}')
        GA.Evolution()
        plt.show()

        # filepaths
        fp_in = [f"images/{self.problem_number}/gen{g}.png"
                 for g in range(1, max_generation + 1)]
        fp_out = f"images/{self.problem_number}/evolution.gif"

        # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
        img, *imgs = [Image.open(f) for f in fp_in]
        img.save(fp=fp_out, format='GIF', append_images=imgs,
                 save_all=True, duration=5000 // max_generation, loop=0)


class ZDT1(ZDTBase):
    gGlobalParetoValue = 1
    gLocalParetoValue = None
    gDeceptiveParetoValue = None
    x1_domain = [0, 1]
    x1_bits = 16
    xrest_domain = [0, 1]
    xrest_bits = 16
    m = 30
    problem_number = 1

    @staticmethod
    def f1(x):
        return x[0]

    @staticmethod
    def g(x):
        return 1 + 9 * (np.sum(x[1:]) / (len(x) - 1))

    @staticmethod
    def h(f1, g):
        return 1 - np.sqrt(f1 / g)


class ZDT2(ZDT1):
    problem_number = 2

    @staticmethod
    def h(f1, g):
        return 1 - (f1 / g) ** 2


class ZDT3(ZDTBase):
    gGlobalParetoValue = 1
    gLocalParetoValue = None
    gDeceptiveParetoValue = None
    x1_domain = [0, 1]
    x1_bits = 16
    xrest_domain = [0, 1]
    xrest_bits = 16
    m = 30
    problem_number = 3

    f2_max_representation = 5
    f2_min_representation = -1

    @staticmethod
    def f1(x):
        return x[0]

    @staticmethod
    def g(x):
        return 1 + 9 * (np.sum(x[1:]) / (len(x) - 1))

    @staticmethod
    def h(f1, g):
        f1divg = f1 / g
        return 1 - np.sqrt(f1divg) - f1divg * np.sin(10 * np.pi * f1)


class ZDT4(ZDTBase):
    gGlobalParetoValue = 1
    gLocalParetoValue = 1.25
    gDeceptiveParetoValue = None
    x1_domain = [0, 1]
    x1_bits = 16
    xrest_domain = [-5, 5]
    xrest_bits = 16
    m = 10
    problem_number = 4

    f1_max_representation = 1
    f2_max_representation = 45

    @staticmethod
    def f1(x):
        return x[0]

    @staticmethod
    def g(x):
        x_rest = x[1:]
        return 1 + 10 * len(x_rest) + (
            np.sum(x_rest * x_rest - 10 * np.cos(4 * np.pi * x_rest)))

    @staticmethod
    def h(f1, g):
        return 1 - np.sqrt(f1 / g)


class ZDT5(ZDTBase):
    gGlobalParetoValue = 10
    gLocalParetoValue = None
    gDeceptiveParetoValue = 11
    x1_domain = [0, 1]
    x1_bits = 30
    xrest_domain = [0, 1]
    xrest_bits = 5
    m = 11
    problem_number = 5

    f1_min_representation = 1
    f1_max_representation = 30
    f2_max_representation = 8

    @classmethod
    def decodechromosome(cls, bits):
        x = []
        x.append(bits[:cls.x1_bits])
        bits = bits[cls.x1_bits:]
        x += textwrap.wrap(bits, cls.xrest_bits)
        return x

    @staticmethod
    def f1(x):
        ret = 1 + x[0].count('1')
        if (ret == 0):
            print('???')
        return ret

    @staticmethod
    def g(x):
        c = 0
        for x_i in x[1:]:
            u = x_i.count('1')
            if u < 5:
                c += (2 + u)
            elif u == 5:
                c += 1
        return c

    @staticmethod
    def h(f1, g):
        return 1 / f1


class ZDT6(ZDTBase):
    gGlobalParetoValue = 1
    gLocalParetoValue = None
    gDeceptiveParetoValue = None
    x1_domain = [0, 1]
    x1_bits = 16
    xrest_domain = [0, 1]
    xrest_bits = 16
    m = 10
    problem_number = 6

    f1_max_representation = 1
    f2_max_representation = 9

    @staticmethod
    def f1(x):
        return 1 - np.exp(-4 * x[0]) * (np.sin(6 * np.pi * x[0]) ** 6)

    @staticmethod
    def g(x):
        return 1 + 9 * ((np.sum(x[1:]) / (len(x) - 1)) ** 0.25)

    @staticmethod
    def h(f1, g):
        return 1 - ((f1 / g) ** 2)
