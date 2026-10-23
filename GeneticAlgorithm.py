import random
from idlelib.editor import keynames
from operator import concat


class Product:
    def __init__(self, name, cost, preference_score, max_weekly_count):
        self.name = name
        self.cost = cost
        self.preference_score = preference_score
        self.max_weekly_count = max_weekly_count

class Chromosome:
    def __init__(self, gens):
        self.gens = gens

product = [
    {"name": "Молоко 1л", "cost": 90, "preference_score": 0.85, "max_weekly_count": 3},
    {"name": "Кефир 1л", "cost": 85, "preference_score": 0.60, "max_weekly_count": 2},
    {"name": "Творог 5% (200г)", "cost": 120, "preference_score": 0.75, "max_weekly_count": 4},
    {"name": "Сыр твердый (кг)", "cost": 800, "preference_score": 0.90, "max_weekly_count": 1},
    {"name": "Масло сливочное (180г)", "cost": 150, "preference_score": 0.70, "max_weekly_count": 2},
    {"name": "Яйца (10 шт)", "cost": 110, "preference_score": 0.95, "max_weekly_count": 3},
    {"name": "Йогурт питьевой", "cost": 70, "preference_score": 0.50, "max_weekly_count": 5},

    #Мясо
    {"name": "Куриное филе (кг)", "cost": 380, "preference_score": 0.80, "max_weekly_count": 2},
    {"name": "Фарш говяжий (кг)", "cost": 550, "preference_score": 0.75, "max_weekly_count": 2},
    {"name": "Сосиски (уп)", "cost": 250, "preference_score": 0.40, "max_weekly_count": 2},

    #крупы
    {"name": "Хлеб нарезной", "cost": 60, "preference_score": 0.90, "max_weekly_count": 3},
    {"name": "Макароны (450г)", "cost": 90, "preference_score": 0.65, "max_weekly_count": 2},
    {"name": "Рис (800г)", "cost": 130, "preference_score": 0.70, "max_weekly_count": 1},
    {"name": "Гречка (800г)", "cost": 110, "preference_score": 0.80, "max_weekly_count": 1},
    {"name": "Овсяные хлопья (400г)", "cost": 80, "preference_score": 0.60, "max_weekly_count": 1},
    {"name": "Масло подсолнечное (1л)", "cost": 140, "preference_score": 0.85, "max_weekly_count": 1},
    {"name": "Сахар (1кг)", "cost": 90, "preference_score": 0.50, "max_weekly_count": 1},
    {"name": "Соль (1кг)", "cost": 40, "preference_score": 0.30, "max_weekly_count": 1},

    #Овощи
    {"name": "Картофель (кг)", "cost": 50, "preference_score": 0.75, "max_weekly_count": 4},
    {"name": "Лук репчатый (кг)", "cost": 45, "preference_score": 0.60, "max_weekly_count": 2},
    {"name": "Морковь (кг)", "cost": 55, "preference_score": 0.65, "max_weekly_count": 2},
    {"name": "Помидоры (кг)", "cost": 250, "preference_score": 0.85, "max_weekly_count": 2},
    {"name": "Огурцы (кг)", "cost": 200, "preference_score": 0.80, "max_weekly_count": 2},
    {"name": "Капуста белокочанная (кг)", "cost": 40, "preference_score": 0.50, "max_weekly_count": 2},

    #Фрукты
    {"name": "Яблоки (кг)", "cost": 120, "preference_score": 0.80, "max_weekly_count": 3},
    {"name": "Бананы (кг)", "cost": 130, "preference_score": 0.90, "max_weekly_count": 2},
    {"name": "Апельсины (кг)", "cost": 180, "preference_score": 0.70, "max_weekly_count": 2},

    #Прочее
    {"name": "Чай черный (100 пак)", "cost": 150, "preference_score": 0.85, "max_weekly_count": 1},
    {"name": "Кофе молотый (250г)", "cost": 450, "preference_score": 0.95, "max_weekly_count": 1},
    {"name": "Сок апельсиновый (1л)", "cost": 160, "preference_score": 0.60, "max_weekly_count": 2},
    {"name": "Шоколад (90г)", "cost": 120, "preference_score": 0.45, "max_weekly_count": 3}
]

def initialize_population(pop_size,products):
    population = []
    for i in range(pop_size):
        population.append(generate_chromosome(products))

    return population

def get_elite(population, fitnesses, elite_pct):
    sorted_population = list(zip(population, fitnesses))
    sorted_population.sort(key=lambda x: x[1], reverse=True)
    elite_count = int(len(population) * elite_pct)
    elites = []
    for i in range(elite_count):
        elites.append(sorted_population[i][0])

    return elites

def generate_chromosome(products):
    gens = []
    for i, product_item in enumerate(products):
        gens.append({"name": product_item["name"], "count": random.randint(0,min(1,product_item["max_weekly_count"]))})

    return Chromosome(gens)

def calculate_target_function(chromosome, products, budget_constraints=None):
    total_cost = 0
    total_preference_score = 0

    for i, gen in enumerate(chromosome.gens):
        total_cost += products[i]["cost"] * gen["count"]
        total_preference_score += products[i]["preference_score"] * gen["count"]

    fitness = total_preference_score - total_cost / 100.0

    if total_cost > budget_constraints and budget_constraints:
        excess = total_cost - budget_constraints
        penalty = excess * 0.2
        fitness -= penalty


    return fitness

def crossover_one(chromosome1: Chromosome, chromosome2 : Chromosome):
    point = random.randint(1,len(chromosome1.gens) - 1)
    child1 = chromosome1.gens[:point] + chromosome2.gens[point:]
    child2 = chromosome2.gens[:point] + chromosome1.gens[point:]

    return Chromosome(child1), Chromosome(child2)

def crossover_multi(chromosome1: Chromosome, chromosome2: Chromosome):
    points = [random.randint(1,len(chromosome1.gens)//2), random.randint(len(chromosome1.gens)//2,len(chromosome1.gens))]
    child1 = chromosome1.gens[:points[0]] + chromosome2.gens[points[0]:points[1]] + chromosome1.gens[points[1]:]
    child2 = chromosome2.gens[:points[0]] + chromosome1.gens[points[0]:points[1]] + chromosome2.gens[points[1]:]

    return Chromosome(child1), Chromosome(child2)

def mutate_random(chromosome: Chromosome, mutation_rate, products):
    for i, gen in enumerate(chromosome.gens):
        value = random.randint(0,100)
        if value < mutation_rate * 100:
            new_value = random.randint(0,products[i]["max_weekly_count"])
            gen["count"] = new_value

def mutate_decrease(chromosome: Chromosome, mutation_rate, products):
    for i, gen in enumerate(chromosome.gens):
        value = random.randint(0,100)
        if value < mutation_rate * 100:
            new_value = gen['count'] - 1
            if new_value < 0:
                new_value = products[i]["max_weekly_count"]
            gen["count"] = new_value

def get_parent_tournament(population, fitnesses):
    participant_count = random.randint(2,5)
    keys = ["chromosome", "fitness"]
    combined = [dict(zip(keys, pair)) for pair in zip(population, fitnesses)]

    participants = random.sample(combined, participant_count)
    winner = max(participants, key=lambda x: x['fitness'])

    return winner[keys[0]]

def main_algorithm(products, pop_size, mutation_rate, generations_count, elite_pct, crossover_type='single_point', mutation_type='mutate_random', stop_condition='generations', budget=None):
    population = initialize_population(pop_size,products)

    population_fitness_history = []
    for generation in range(generations_count):
        fitnesses = []
        for chromosome in population:
            fitnesses.append(calculate_target_function(chromosome, products, budget))
        population_fitness_history.append(max(fitnesses))

        new_population = get_elite(population, fitnesses, elite_pct)
        empty = pop_size - len(new_population)
        while len(new_population) < pop_size:
            parent1 = get_parent_tournament(population, fitnesses)
            parent2 = get_parent_tournament(population, fitnesses)

            if crossover_type == 'single_point':
                child1, child2 = crossover_one(parent1, parent2)
            else:
                child1, child2 = crossover_multi(parent1, parent2)

            if mutation_type == 'mutate_random':
                mutate_random(child1, mutation_rate, products)
                mutate_random(child2, mutation_rate, products)
            else:
                mutate_decrease(child1, mutation_rate, products)
                mutate_decrease(child2, mutation_rate, products)

            new_population.append(child1)
            new_population.append(child2)

        population = new_population[:pop_size]

    return population, population_fitness_history

BUDGET = 5000

pop, his = main_algorithm(product,50,0.05,100,0.1,budget=BUDGET)

final_fitnesses = [calculate_target_function(chrom, product, BUDGET) for chrom in pop]
best_chrom, best_fit = max(zip(pop, final_fitnesses), key=lambda x: x[1])

print("=== ЛУЧШЕЕ РЕШЕНИЕ ===")
total_cost = 0
total_pref = 0
for i, gen in enumerate(best_chrom.gens):
    if gen["count"] > 0:
        print(f"{product[i]['name']}: {gen['count']} шт. (Цена: {product[i]['cost']})")
        total_cost += product[i]["cost"] * gen["count"]
        total_pref += product[i]["preference_score"] * gen["count"]

print(f"\nИтоговая стоимость: {total_cost} руб. (Бюджет: 5000)")
print(f"Итоговое предпочтение: {total_pref:.2f}")
print(f"Фитнес: {best_fit}")
total_cost = 0
for i in range(len(pop[0].gens)):
    total_cost += product[i]["cost"] * pop[0].gens[i]["count"]
print(total_cost)
print(his)






