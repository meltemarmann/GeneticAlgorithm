# -*- coding: utf-8 -*-
import random
import matplotlib.pyplot as plt 
import numpy as np

#parameters
n = 20; #number of firms
alpha = 1023;
c = 0; #cost c <alpha
l = 10; #the length of binary strings
p_cross = 0.5009;
p_mut = 0.00409;
T = 1000; #te number of iterations
price = 0;

def calculate_fitness(q, total_output):
    price = alpha - total_output;
    fitness = [0]*20;
    for i in range(n):
        fitness[i] = q[i]*price - q[i]*c;
    return fitness;

def crossover(fitness, chromosom):
    total_fitness = sum(fitness)
    relative_fitness = [0] * n
    if total_fitness != 0:
        relative_fitness = [f / total_fitness for f in fitness]
    else:
        relative_fitness = [(1/n) for f in fitness]
    rep_generation = []

    for i in range(n):
        cumulative_prob = 0
        prob = random.random()
        for j in range(n):
            cumulative_prob += relative_fitness[j]
            if prob < cumulative_prob:
                rep_generation.append(chromosom[j].copy())
                break
    
    random.shuffle(rep_generation)

    # Create pairs by pairing adjacent individuals
    pairs = []
    for i in range(0, len(rep_generation) - 1, 2):
        pairs.append((rep_generation[i], rep_generation[i + 1]))

    cross_gen = []
    for pair in pairs:
        parent1, parent2 = pair

        q1 = 0
        q2 = 0
        for j in range(l):
            q1 = q1 + (2 ** j) * parent1[j]
            q2 = q2 + (2 ** j) * parent2[j]

        q1 = (q1 * alpha) // (2 ** l - 1)
        q2 = (q2 * alpha) // (2 ** l - 1)

        fitness_p1 = q1 * price - q1 * c
        fitness_p2 = q2 * price - q2 * c

        rand_swap = random.random()
        child1, child2 = [], []

        if rand_swap <= p_cross:  # swap
            rand_k = random.randint(1, l - 1)
            for i in range(l):
                if i < rand_k:
                    child1.append(parent1[i])
                    child2.append(parent2[i])
                else:
                    child1.append(parent2[i])
                    child2.append(parent1[i])

            for j in range(l):
                rand_mut = random.random()
                if rand_mut <= p_mut:
                    child1[j] = 1 - child1[j]
            for j in range(l):
                rand_mut = random.random()
                if rand_mut <= p_mut:
                    child2[j] = 1 - child2[j]

            c1 = 0
            c2 = 0
            for j in range(l):
                c1 = c1 + (2 ** j) * child1[j]
                c2 = c2 + (2 ** j) * child2[j]

            c1 = (c1 * alpha) // (2 ** l - 1)
            c2 = (c2 * alpha) // (2 ** l - 1)

            fitness_c1 = c1 * price - c1 * c
            fitness_c2 = c2 * price - c2 * c

            if fitness_c1 >= fitness_p1:
                cross_gen.append(child1)
            else:
                cross_gen.append(parent1)

            if fitness_c2 >= fitness_p2:
                cross_gen.append(child2)
            else:
                cross_gen.append(parent2)

        else:
            child1 = parent1
            child2 = parent2

            for j in range(l):
                rand_mut = random.random()
                if rand_mut <= p_mut:
                    child1[j] = 1 - child1[j]
            for j in range(l):
                rand_mut = random.random()
                if rand_mut <= p_mut:
                    child2[j] = 1 - child2[j]

            c1, c2 = 0, 0
            for j in range(l):
                c1 = c1 + (2 ** j) * child1[j]
                c2 = c2 + (2 ** j) * child2[j]

            c1 = (c1 * alpha) // (2 ** l - 1)
            c2 = (c2 * alpha) // (2 ** l - 1)

            fitness_c1 = c1 * price - c1 * c
            fitness_c2 = c2 * price - c2 * c

            if fitness_c1 >= fitness_p1:
                cross_gen.append(child1)
            else:
                cross_gen.append(parent1)

            if fitness_c2 >= fitness_p2:
                cross_gen.append(child2)
            else:
                cross_gen.append(parent2)
    
    return cross_gen

def main():
    # Initialization
    chromosom = [[0] * l for _ in range(n)]
    q = [0] * n
    total_output = 0  # Q

    for i in range(n):
        chromosom[i] = [random.randint(0, 1) for _ in range(l)]

    time = []
    price_array = []
    fitness_array = []
    ind_output = []
    population_variance = []
    for k in range(T):
        q = [0] * n

        for i in range(n):
            for j in range(l):
                q[i] = q[i] + (2**j) * chromosom[i][j]

            q[i] = ((q[i] * alpha) / (2**l - 1))/n

        total_output = sum(q)
        ind_output.append(total_output)
        population_variance.append(sum((qi - total_output/n)**2 for qi in q)/n)
        price = max(0, alpha - total_output);
        print(q)
        fitness = [0] * n

        for i in range(n):
            fitness[i] = q[i] * price - q[i] * c

        chromosom = crossover(fitness, chromosom)
        # print(fitness)
        time.append(k)
        price_array.append(price)
        fitness_array.append(sum(fitness))
        

    plt.plot(time, ind_output)
    plt.title("Total Industry Output Over Time")
    plt.xlabel("T")
    plt.ylabel("Industry Output")
    plt.yticks(np.arange(min(ind_output), max(ind_output)+1, 50))
    plt.tight_layout()
    plt.show()

    plt.plot(time, population_variance)
    plt.title("Variance Over Time")
    plt.xlabel("T")
    plt.ylabel("Variance")
    plt.yticks(np.arange(min(population_variance), max(population_variance)+1, 20))
    plt.tight_layout()
    plt.show()

if __name__ == '__main__': 
	main()