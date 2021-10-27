from simanneal import Annealer
import numpy as np
import random

N = 27


class NQueensBaseAnnealer(Annealer):

    def __init__(self, initial_state=None, load_state=None):
        super().__init__(initial_state=initial_state, load_state=load_state)
        self.epoch = 0

    def anneal(self):
        return super().anneal()

    def energy(self):
        n_collisions = 0
        # Calculate number of collisions
        for (i, j) in enumerate(self.state):
            for (m, n) in enumerate(self.state[i+1:]):
                if n == j:
                    n_collisions += 1
                elif abs(j - n) == m + 1:
                    n_collisions += 1
        if n_collisions == 0:
            # self.user_exit = True
            pass
        return n_collisions

    def move(self):
        self.epoch += 1
        # swap places of queens
        a = random.randint(0, len(self.state) - 1)
        b = random.randint(0, len(self.state) - 1)
        self.state[a], self.state[b] = self.state[b], self.state[a]


if __name__ == '__main__':
    ann = NQueensBaseAnnealer(np.arange(N))
    schedule = ann.auto(minutes=1)
    print()
    print(ann.epoch)
    print(schedule)
    ann.user_exit = False
    ann.set_schedule(schedule)
    best_sol, best_cost = ann.anneal()
    print()
    print(ann.epoch)
    print(best_sol)
    for i in range(N):
        for j in range(N):
            if best_sol[j] == i:
                print('X', end=' ')
            else:
                print('.', end=' ')
        print()
    print(best_cost)