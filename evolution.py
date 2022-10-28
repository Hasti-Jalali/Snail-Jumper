import copy
import numpy as np
import math
import json
from player import Player


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"
        self.generation_number = 0

    def q_tournament(self, num_players, players, q):
        result = []
        for i in range(num_players):
            q_selected = np.random.choice(players, q)
            result.append(max(q_selected, key=lambda player: player.fitness))
        return result

    def roulette_wheel(self, num_players, players):
        total_fitness = sum([player.fitness for player in players])
        probability = [player.fitness / total_fitness for player in players]
        result = np.random.choice(players, size=num_players, p=probability)
        return result

    def calc_probability(self, players):
        total_fitness = sum([player.fitness for player in players])
        probability = [player.fitness / total_fitness for player in players]
        for i in range(1, len(players)):
            probability[i] += probability[i - 1]
        return probability

    def sus(self, players, num_players):
        probability = self.calc_probability(players)
        result = []
        step = (probability[len(probability) - 1] - np.random.uniform(0, 1 / num_players, 1)) / num_players
        for i in range(num_players):
            temp = (i + 1) * step
            for n, p in enumerate(probability):
                if temp <= p:
                    res = self.clone_player(players[n])
                    result.append(res)
                    break
        return result


    def mutation(self, chromosome, mp = 0.5):
        random_number = np.random.uniform(0, 1, 1)
        if random_number <= mp:
            chromosome.nn.w1 += np.random.normal(0, 0.3, size=chromosome.nn.w1.shape)
            chromosome.nn.w2 += np.random.normal(0, 0.3, size=chromosome.nn.w2.shape)

            chromosome.nn.b1 += np.random.normal(0, 0.3, size=chromosome.nn.b1.shape)
            chromosome.nn.b2 += np.random.normal(0, 0.3, size=chromosome.nn.b2.shape)
        return chromosome


    def crossover(self, array1, array2, cp = 0.8):
        crossover_place = math.floor(array1.shape[0] / 2)

        random_number = np.random.uniform(0, 1, 1)
        if random_number > cp:
            return array1, array2

        else:
            child1 = np.concatenate((array1[:crossover_place], array2[crossover_place:]), axis=0)
            child2 = np.concatenate((array2[:crossover_place], array1[crossover_place:]), axis=0)
        return child1, child2

    def next_population_selection(self, players, num_players):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """
        # TODO (Implement top-k algorithm here)
        players.sort(key=lambda x: x.fitness, reverse=True)

        fitness_list = [player.fitness for player in players]
        best_fitness = float(np.max(fitness_list))
        average_fitness = float(np.mean(fitness_list))
        worst_fitness = float(np.min(fitness_list))
        self.save_fitness_results(best_fitness, worst_fitness, average_fitness)

        # TODO (Additional: Implement roulette wheel here)
        # TODO (Additional: Implement SUS here)

        # TODO (Additional: Learning curve)
        return players[: num_players]

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        first_generation = prev_players is None
        if first_generation:
            return [Player(self.game_mode) for _ in range(num_players)]
        else:
            # TODO ( Parent selection and child generation )
            children = []
            # new_players = self.q_tournament(num_players, prev_players, 10)
            # new_players = self.roulette_wheel(num_players, prev_players)
            new_players = self.sus(prev_players, num_players)
            for i in range(0, len(new_players), 2):
                parent1 = new_players[i]
                parent2 = new_players[i + 1]
                child1 = self.clone_player(parent1)
                child2 = self.clone_player(parent2)
                child1.nn.w1, child2.nn.w1 = self.crossover(parent1.nn.w1, parent2.nn.w1)
                child1.nn.w2, child2.nn.w2 = self.crossover(parent1.nn.w2, parent2.nn.w2)
                child1.nn.b1, child2.nn.b1 = self.crossover(parent1.nn.b1, parent2.nn.b1)
                child1.nn.b2, child2.nn.b2 = self.crossover(parent1.nn.b2, parent2.nn.b2)
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)
                children.append(child1)
                children.append(child2)
            return children

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player


    def save_fitness_results(self, max_fitness, min_fitness, average_fitness):
        if (self.generation_number == 0):
            generation_results = {
                'best_fitnesses': [max_fitness],
                'worst_fitnesses': [min_fitness],
                'average_fitnesses': [average_fitness]
            }
            with open('generation_results.json', 'w') as file:
                json.dump(generation_results, file)
            file.close()
        else:
            with open('generation_results.json', 'r') as file:
                generation_results = json.load(file)
            file.close()
            generation_results['best_fitnesses'].append(max_fitness)
            generation_results['worst_fitnesses'].append(min_fitness)
            generation_results['average_fitnesses'].append(average_fitness)

            with open('generation_results.json', 'w') as file:
                json.dump(generation_results, file)
            file.close()
        self.generation_number += 1