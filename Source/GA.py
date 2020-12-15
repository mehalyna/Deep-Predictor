import random
# manages project library


def getParents(population, fitnessValues, numOfParents):  # selects parents based on fitness
    parents = []

    for x in range(numOfParents):  # iterates through parents and selects the best "n"
        fitMaxIndex = fitnessValues.index(max(fitnessValues))
        parents.append(population[fitMaxIndex])
        fitnessValues[fitMaxIndex] = -1

    return parents


def breed(parents, numOfOffspring, schemaCount):  # data from the population is combined and "bred" to create offspring
    offspring = []
    numOfParents = len(parents)

    for x in range(numOfOffspring):  # iterates through the population and combines data from random parents to create offspring
        parentIndex1 = random.randint(0, numOfParents - 1)
        parentIndex2 = random.randint(0, numOfParents - 1)

        child = []

        for y in range(schemaCount):
            child.append((parents[parentIndex1][y] + parents[parentIndex2][y]) / 2)

        child[0] = int(child[0])
        offspring.append(child)

    return offspring


def crossover(offspring, crossoverChance, schemaCount):  # crosses random genes between offspring
    numOfOffspring = len(offspring)

    for x in range(numOfOffspring):  # iterates through offspring, and randomly exchanges genes
        for y in range(schemaCount):
            chanceVal = random.random()

            if crossoverChance > chanceVal:
                crossover1 = random.randint(0, numOfOffspring - 1)
                crossover2 = random.randint(0, numOfOffspring - 1)

                temp = offspring[crossover1][y]
                offspring[crossover1][y] = offspring[crossover2][y]
                offspring[crossover2][y] = temp

    return offspring


def mutate(offspring, mutationChance, schemaCount):  # mutates some offspring

    for x in range(len(offspring)):  # iterates through offspring and generates a random value
        chanceVal = random.random()

        if mutationChance > chanceVal:  # if chance specified is lower than the random value, random mutation occurs
            for y in range(schemaCount):
                offspring[x][y] = offspring[x][y] * random.random() * 2

            offspring[x][0] = int(offspring[x][0])

    return offspring