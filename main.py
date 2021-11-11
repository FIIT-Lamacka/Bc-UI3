import copy
import random
from tabulate import tabulate
from colorama import *
init()


class Orientation:
    def __init__(self):
        self.directions = ["N", "E", "S", "W"]
        self.current = 0

    def get_orientation(self):
        return self.directions[self.current]

    def change_direction(self, side):
        if side == "0":
            self.current = (self.current - 1) % 4
            return

        elif side == "1":
            self.current = (self.current + 1) % 4
            return


class Generation:
    def __init__(self, contained_monks, given_garden):
        self.monks = contained_monks
        self.garden = given_garden
        self.alfa_monk = None
        self.alfa_monk_solution = None


class Garden:
    def __init__(self, garden_map):
        self.garden_map = garden_map
        self.size_w = len(garden_map[0])
        self.size_h = len(garden_map)
        self.circ = 2*self.size_w + 2*self.size_h
        self.orientation = None

        k = 0
        for i in range(len(garden_map)):
            for j in range(len(garden_map[i])):
                if garden_map[i][j] == -1:
                    k += 1
        self.stones_n = k

    def print_garden(self):
        print(tabulate(self.garden_map, tablefmt="fancy_grid"))

    def find_entry(self, number):
        # print(number)
        # print(self.circ)
        if number < self.size_w:
            # print("ENTRY TOP")
            x_entry = number
            y_entry = 0
            return [x_entry, y_entry, 2]

        if number < self.size_w + self.size_h:
            # print("ENTRY RIGHT")
            x_entry = self.size_w - 1
            y_entry = number - self.size_w
            return [x_entry, y_entry, 3]

        if number < 2*self.size_w + self.size_h:
            # print("ENTRY BOTTOM")
            x_entry = self.size_w - (number - (self.size_w + self.size_h)) - 1
            y_entry = self.size_h - 1
            return [x_entry, y_entry, 0]

        if number < 2*self.size_w + 2*self.size_h:
            # print("ENTRY LEFT")
            x_entry = 0
            y_entry = self.size_h - (number - (2*self.size_w + self.size_h)) - 1
            return [x_entry, y_entry, 1]


class Monk:
    def __init__(self, genome, genome_size, garden_size):
        self.genome = format(genome, "0" + str(genome_size) + "b")
        self.genome_len = len(self.genome)
        self.move_genome = []
        self.orientation = Orientation()
        self.x_pos = None
        self.y_pos = None
        self.bump_count = 1
        self.plow_counter = 1
        self.fitness = 0
        self.solved_garden = None
        self.garden_size = garden_size

    def next_step_pos(self):
        if self.orientation.get_orientation() == "N":
            return [self.y_pos-1, self.x_pos]
        if self.orientation.get_orientation() == "E":
            return [self.y_pos, self.x_pos + 1]
        if self.orientation.get_orientation() == "S":
            return [self.y_pos + 1, self.x_pos]
        if self.orientation.get_orientation() == "W":
            return [self.y_pos, self.x_pos - 1]

    def step(self, garden_array):
        if self.orientation.get_orientation() == "N":
            # AK SME NA KONCI ZAHRADY
            if self.y_pos == 0:
                garden_array[self.y_pos][self.x_pos] = self.plow_counter
                return "Done"

            # AK MNICH NARAZI NA KAMEN
            if garden_array[self.y_pos - 1][self.x_pos] != 0:
                # print("STONE FUCKED ME")
                garden_array[self.y_pos][self.x_pos] = self.plow_counter
                self.orientation.change_direction(self.genome[-self.bump_count])
                next_s = self.next_step_pos()

                if next_s[1] == -1 or next_s[1] == len(garden_array[0]):
                    return "Done"

                if garden_array[next_s[0]][next_s[1]] != 0:
                    # print("OTOCKA")
                    # print(self.orientation.get_orientation())
                    self.orientation.change_direction("0")
                    self.orientation.change_direction("0")
                    # print(self.orientation.get_orientation())

                next_s = self.next_step_pos()
                if next_s[1] == -1 or next_s[1] == len(garden_array[0]):
                    return "Done"

                if garden_array[next_s[0]][next_s[1]] != 0:
                    # print("ANOTHER STONE FUCKED ME")
                    return "Broken"

                self.bump_count += 1
                return

            # AK JE VSETKO OK
            garden_array[self.y_pos][self.x_pos] = self.plow_counter
            self.y_pos -= 1

        if self.orientation.get_orientation() == "W":
            # AK SME NA KONCI ZAHRADY
            if self.x_pos == 0:
                garden_array[self.y_pos][self.x_pos] = self.plow_counter
                return "Done"

            # AK MNICH NARAZI NA KAMEN
            if garden_array[self.y_pos][self.x_pos - 1] != 0:
                # print("STONE FUCKED ME")
                garden_array[self.y_pos][self.x_pos] = self.plow_counter
                self.orientation.change_direction(self.genome[-self.bump_count])
                next_s = self.next_step_pos()

                if next_s[0] == -1 or next_s[0] == len(garden_array):
                    return "Done"
                if garden_array[next_s[0]][next_s[1]] != 0:
                    # print("OTOCKA")
                    # print(self.orientation.get_orientation())
                    self.orientation.change_direction("0")
                    self.orientation.change_direction("0")
                    # print(self.orientation.get_orientation())

                next_s = self.next_step_pos()
                if next_s[0] == -1 or next_s[0] == len(garden_array):
                    return "Done"
                if garden_array[next_s[0]][next_s[1]] != 0:
                    # print("ANOTHER STONE FUCKED ME")
                    return "Broken"

                self.bump_count += 1
                return

            # AK JE VSETKO OK
            garden_array[self.y_pos][self.x_pos] = self.plow_counter
            self.x_pos -= 1

        if self.orientation.get_orientation() == "E":
            # AK SME NA KONCI ZAHRADY
            if self.x_pos == garden.size_w-1:
                garden_array[self.y_pos][self.x_pos] = self.plow_counter
                return "Done"

            # AK MNICH NARAZI NA KAMEN
            if garden_array[self.y_pos][self.x_pos + 1] != 0:
                # print("STONE FUCKED ME")
                garden_array[self.y_pos][self.x_pos] = self.plow_counter
                self.orientation.change_direction(self.genome[-self.bump_count])
                next_s = self.next_step_pos()

                if next_s[0] == -1 or next_s[0] == len(garden_array):
                    return "Done"

                if garden_array[next_s[0]][next_s[1]] != 0:
                    # print("OTOCKA")
                    # print(self.orientation.get_orientation())
                    self.orientation.change_direction("0")
                    self.orientation.change_direction("0")
                    #  print(self.orientation.get_orientation())

                next_s = self.next_step_pos()

                if next_s[0] == -1 or next_s[0] == len(garden_array):
                    return "Done"
                if garden_array[next_s[0]][next_s[1]] != 0:
                    # print("ANOTHER STONE FUCKED ME")
                    return "Broken"

                self.bump_count += 1
                return

            # AK JE VSETKO OK
            garden_array[self.y_pos][self.x_pos] = self.plow_counter
            self.x_pos += 1

        if self.orientation.get_orientation() == "S":
            # AK SME NA KONCI ZAHRADY
            if self.y_pos == garden.size_h-1:
                garden_array[self.y_pos][self.x_pos] = self.plow_counter
                return "Done"

            # AK MNICH NARAZI NA KAMEN
            if garden_array[self.y_pos + 1][self.x_pos] != 0:
                # print("STONE FUCKED ME")
                garden_array[self.y_pos][self.x_pos] = self.plow_counter
                self.orientation.change_direction(self.genome[-self.bump_count])
                next_s = self.next_step_pos()

                if next_s[1] == -1 or next_s[1] == len(garden_array[0]):
                    return "Done"

                if garden_array[next_s[0]][next_s[1]] != 0:
                    # print("OTOCKA")
                    # print(self.orientation.get_orientation())
                    self.orientation.change_direction("0")
                    self.orientation.change_direction("0")
                    # print(self.orientation.get_orientation())

                next_s = self.next_step_pos()
                if next_s[1] == -1 or next_s[1] == len(garden_array[0]):
                    return "Done"
                if garden_array[next_s[0]][next_s[1]] != 0:
                    # print("ANOTHER STONE FUCKED ME")
                    return "Broken"

                self.bump_count += 1
                return

            # AK JE VSETKO OK
            garden_array[self.y_pos][self.x_pos] = self.plow_counter
            self.y_pos += 1

    def mutate_genome(self):
        gene_len = len(self.genome)
        move_len = len(self.move_genome)
        mutated_gene = copy.copy(self.genome)

        for i in range(move_len):
            if random.randint(0, move_len) == 0:
                self.move_genome[i] = random.randint(0, self.garden_size-1)

        for i in range(gene_len):
            if random.randint(0, gene_len) == 0:
                # print("MUTATED AT", i)
                mutated_gene = flip_bit(mutated_gene, i)

        self.genome = mutated_gene

    def move(self, given_garden, entry_x, entry_y, direction):

        self.x_pos = entry_x
        self.y_pos = entry_y

        if given_garden[entry_y][entry_x] != 0:
            return "Broken"
        self.orientation.current = direction
        answer = None

        while answer != "Done" and answer != "Broken":
            # print("STEP AT X:", self.x_pos, "Y:", self.y_pos, "ORIENTATION:", self.orientation.get_orientation())
            answer = self.step(given_garden)

        if answer == "Broken":
            return "Broken"
        self.plow_counter += 1

    def solve(self, given_garden):
        garden_copy = [row[:] for row in given_garden.garden_map]
        iterator = 0
        answer = None

        while answer != "Broken":
            # print(self.genome)
            entry_point = garden.find_entry(self.move_genome[iterator])
            # print(entry_point[0], entry_point[1], entry_point[2])

            answer = self.move(garden_copy, entry_point[0], entry_point[1], entry_point[2])
            iterator += 1

        self.solved_garden = garden_copy
        colormap(garden_copy)
        for j in range(len(garden_copy)):
            for i in range(len(garden_copy[0])):
                if garden_copy[j][i] > 0:
                    self.fitness += 1


def colormap(garden_orig):
    garden_copy = [row[:] for row in garden_orig]

    for j in range(len(garden_copy)):
        for i in range(len(garden_copy[0])):
            if garden_copy[j][i] != 0 and garden_copy[j][i] != -1:
                garden_copy[j][i] = Back.GREEN + Fore.BLACK + str(garden_copy[j][i]) + Style.RESET_ALL
            if garden_copy[j][i] == -1:
                garden_copy[j][i] = Back.RED + str(-1) + Style.RESET_ALL

    return garden_copy


def flip_bit(gene, bit_pos):
    left = gene[:bit_pos]
    bit = gene[bit_pos]
    right = gene[bit_pos+1:]
    if bit == "0":
        bit = "1"
    else:
        bit = "0"

    return left + bit + right


def random_genes(gene_number, size):
    generated_genes = []
    for i in range(gene_number):
        generated_genes.append(random.randint(0, pow(2, size) - 1))

    return generated_genes


def create_first_gen(size, garden_size):
    first_monks = []
    genes = random_genes(48, size)

    for gene in genes:
        new_monk = Monk(gene, size, garden_size)
        new_monk.move_genome = random.sample(range(0, garden_size), int(garden_size/2))
        print(new_monk.move_genome)
        first_monks.append(new_monk)

    return first_monks


def generate_garden(width, height, *args):
    new_garden = [([0] * width) for i in range(height)]
    for stone in args:
        new_garden[stone[1]][stone[0]] = -1

    return new_garden


def choice_roulette(monks):
    roulette = []
    lucky_monks = []

    for monk in monks:
        for i in range(monk.fitness):
            roulette.append(monk)
        # print(monk.fitness)

    for i in range(24):
        lucker = random.choice(roulette)
        while lucker in lucky_monks:
            lucker = random.choice(roulette)
        lucky_monks.append(lucker)

    return lucky_monks


def choice_elite(monks):
    sorted_monks = sorted(monks, key=lambda monk: monk.fitness, reverse=True)

    for i in range(24):
        sorted_monks.pop()

    return sorted_monks


def make_children(parent1: Monk, parent2: Monk):
    cutoff = random.randint(0, parent1.genome_len)
    cutoff2 = random.randint(0, int(parent1.garden_size/2))
    gene1 = parent1.genome[:cutoff] + parent2.genome[cutoff:]
    gene2 = parent2.genome[:cutoff] + parent1.genome[cutoff:]
    move_gene1 = parent1.move_genome[:cutoff2]
    move_gene2 = parent2.move_genome[:cutoff2]
    move_gene1.extend(parent2.move_genome[cutoff2:])
    move_gene2.extend(parent1.move_genome[cutoff2:])


    chilren = []
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)
    child1.genome = gene1
    child1.move_genome = move_gene1
    child2.genome = gene2
    child2.move_genome = move_gene2
    child1.mutate_genome()
    child2.mutate_genome()

    chilren.append(child1)
    chilren.append(child2)

    return chilren


def create_generation(parents):
    old_generation = copy.deepcopy(parents)
    new_generation = copy.deepcopy(parents)
    for monk in new_generation:
        monk.orientation = Orientation()
        monk.x_pos = None
        monk.y_pos = None
        monk.bump_count = 1
        monk.plow_counter = 1
        monk.fitness = 0
        monk.solved_garden = None

    for monk in old_generation:
        monk.orientation = Orientation()
        monk.x_pos = None
        monk.y_pos = None
        monk.bump_count = 1
        monk.plow_counter = 1
        monk.fitness = 0
        monk.solved_garden = None

    pairs = random.sample(range(0, 24), 24)

    for i in range(0, 24, 2):
        new_generation.extend(make_children(old_generation[pairs[i]], old_generation[pairs[i+1]]))

    return new_generation


def evolution_elite(gen: Generation):
    best_fitness = 0
    best_garden = None
    gen_count = 0

    while True:
        gen_count += 1
        for monk in gen.monks:
            monk.solve(garden)

        '''for monks in gen.monks:
            print(monks.fitness)'''
        for monk in gen.monks:
            if monk.fitness > best_fitness:
                best_fitness = monk.fitness
                best_garden = monk.solved_garden
                print("GENERATION:", gen_count)
                print("FITNESS:", best_fitness)
                print(tabulate(colormap(best_garden), tablefmt="fancy_grid"))

        alpha_monks = choice_elite(gen.monks)

        # print(tabulate(colormap(alpha_monks[len(alpha_monks)-1].solved_garden), tablefmt="fancy_grid"))

        new_monks = create_generation(alpha_monks)

        gen = Generation(new_monks, gen.garden)


def evolution_roulette(gen: Generation):
    best_fitness = 0
    best_garden = None
    gen_count = 0

    while True:
        gen_count += 1
        for monk in gen.monks:
            monk.solve(garden)



        for monk in gen.monks:
            if monk.fitness > best_fitness:
                best_fitness = monk.fitness
                best_garden = monk.solved_garden
                print("GENERATION:", gen_count)
                print("FITNESS:", best_fitness)
                print(tabulate(colormap(best_garden), tablefmt="fancy_grid"))

        lucky_monks = choice_roulette(gen.monks)
        new_monks = create_generation(lucky_monks)

        gen = Generation(new_monks, gen.garden)


if __name__ == '__main__':
    garden = Garden(generate_garden(12, 10, [1, 2], [5, 1], [4, 3], [2, 4], [8, 6], [9, 6]))
    monks = create_first_gen(32, garden.circ)
    garden.print_garden()

    gen = Generation(monks, garden)

    evolution_elite(gen)
    #evolution_roulette(gen)


    # print(tabulate(colormap(best_monk.solved_garden), tablefmt="fancy_grid"))






