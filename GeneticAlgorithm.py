import copy
import random
import time
from idlelib.editor import keynames
from operator import concat
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd

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
    pop = []
    for i in range(elite_count):
        elites.append(sorted_population[i][0])
    for i in range(elite_count, len(population)):
        pop.append(sorted_population[i][0])

    return elites, pop

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

    if total_cost == 0:
        return 0

    cost_ratio = total_cost / budget_constraints

    fitness = total_preference_score / cost_ratio

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

def main_algorithm(products, pop_size, mutation_rate, generations_count, elite_pct, crossover_type='single_point', mutation_type='mutate_random', stop_condition='generations', budget=None, target_fitness=None):
    population = initialize_population(pop_size,products)

    population_fitness_history = []
    best_fitness_ever = -float('inf')
    for generation in range(generations_count):
        fitnesses = []
        for chromosome in population:
            fitnesses.append(calculate_target_function(chromosome, products, budget))

        current_best = max(fitnesses)
        best_fitness_ever = max(best_fitness_ever, current_best)
        population_fitness_history.append(best_fitness_ever)

        new_population, population = get_elite(population, fitnesses, elite_pct)
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
        if stop_condition == 'target_function':
            if best_fitness_ever >= target_fitness:
                print(f"Достигнут целевой фитнес {target_fitness} на поколении {generation}")
                break

    return population, population_fitness_history


def run_experiment(override_params=None, num_runs=3):
    BASE_PARAMS = {
        'pop_size': 50,
        'generations_count': 150,
        'elite_pct': 0.1,
        'mutation_rate': 0.1,
        'crossover_type': 'single_point',
        'mutation_type': 'mutate_random',
        'stop_condition': 'generations',
        'budget': 5000,
        'target_fitness': 30
    }

    params = copy.deepcopy(BASE_PARAMS)

    if override_params:
        params.update(override_params)

    results_for_this_config = []

    for run in range(num_runs):
        random.seed(42 + run)

        pop, history = main_algorithm(product, **params)

        final_fitnesses = [calculate_target_function(chrom, product, params['budget']) for chrom in pop]
        best_chrom, best_fit = max(zip(pop, final_fitnesses), key=lambda x: x[1])

        total_cost = sum(product[i]["cost"] * gen["count"] for i, gen in enumerate(best_chrom.gens))
        total_pref = sum(product[i]["preference_score"] * gen["count"] for i, gen in enumerate(best_chrom.gens))

        result = {
            **params,
            'run_id': run,
            'best_fitness': best_fit,
            'final_cost': total_cost,
            'final_preference': total_pref,
            'fitness_history': history
        }
        results_for_this_config.append(result)

    return results_for_this_config

def show_graphs(parameter_name, parameters, df, base_generations=150):
    plt.figure(figsize=(10, 6))
    for p in parameters:
        subset = df[(df[parameter_name] == p) & (df['generations_count'] == base_generations)]

        if len(subset) == 0:
            print(f"⚠Нет данных для {parameter_name}={p}")
            continue

        avg_history = subset['fitness_history'].apply(pd.Series).mean().values
        plt.plot(avg_history, label=f'{parameter_name}={p}')

    plt.xlabel('Поколение')
    plt.ylabel('Лучший фитнес')
    plt.title(f'Влияние {parameter_name} на сходимость')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'convergence_{parameter_name}.png')
    plt.show()

def show_barcharts(parameter_name, df):
    stats = df.groupby(parameter_name)['best_fitness'].agg(['mean', 'std'])
    plt.figure(figsize=(8, 5))
    plt.bar(stats.index, stats['mean'], yerr=stats['std'], capsize=5)
    plt.ylabel('Средний лучший фитнес')
    plt.title(f'Сравнение {parameter_name}')
    plt.savefig(f'{parameter_name}_comparison.png')
    plt.show()


def show_bar_generations(df):
    plt.figure(figsize=(10, 6))

    generations_counts = [50, 100, 150, 200]
    avg_fitness = []
    std_fitness = []

    for gc in generations_counts:
        subset = df[df['generations_count'] == gc]
        if len(subset) > 0:
            final_fitnesses = [row['fitness_history'][-1] for _, row in subset.iterrows()]
            avg_fitness.append(np.mean(final_fitnesses))
            std_fitness.append(np.std(final_fitnesses))
        else:
            avg_fitness.append(0)
            std_fitness.append(0)

    plt.bar(generations_counts, avg_fitness, yerr=std_fitness,
            color=['blue', 'orange', 'green', 'red'])
    plt.xlabel('Количество поколений')
    plt.ylabel('Финальный лучший фитнес')
    plt.title('Влияние generations_count на качество решения')
    plt.grid(True, alpha=0.3)
    plt.savefig('bar_generations_count.png', dpi=150)
    plt.show()

pop_sizes = [30,50,70,100]
generations_counts = [50,100,150,200]
elite_pcts = [0.05,0.1,0.15,0.2]
mutation_rates = [0.05,0.1,0.15,0.2]
crossover_types = ['single_point','multi_points']
mutation_types = ['mutate_random','mutate_decrease']
stop_conditions = ['generations','target_function']

all_results = []

for pop in pop_sizes:
    results = run_experiment(override_params={'pop_size': pop}, num_runs=3)
    all_results.extend(results)
    print(f"pop_size={pop} завершён")

for mr in mutation_rates:
    results = run_experiment(override_params={'mutation_rate': mr}, num_runs=3)
    all_results.extend(results)
    print(f"mutation_rate={mr} завершён")

for ct in crossover_types:
    results = run_experiment(override_params={'crossover_type': ct}, num_runs=3)
    all_results.extend(results)
    print(f"crossover_type={ct} завершён")

for gc in generations_counts:
    results = run_experiment(override_params={'generations_count': gc}, num_runs=3)
    all_results.extend(results)
    print(f"generation_count={gc} завершён")

for ep in elite_pcts:
    results = run_experiment(override_params={'elite_pct': ep}, num_runs=3)
    all_results.extend(results)
    print(f"elite_pct={ep} завершён")

for mt in mutation_types:
    results = run_experiment(override_params={'mutation_type': mt}, num_runs=3)
    all_results.extend(results)
    print(f"mutation_type={mt} завершён")

for sc in stop_conditions:
    results = run_experiment(override_params={'stop_condition': sc}, num_runs=3)
    all_results.extend(results)
    print(f"stop_condition={sc} завершён")

df = pd.DataFrame(all_results)

show_graphs('pop_size',pop_sizes,df)
show_graphs('mutation_rate',mutation_rates,df)
show_graphs('elite_pct',elite_pcts,df)
show_bar_generations(df)
show_barcharts('crossover_type',df)
show_barcharts('mutation_type',df)
show_barcharts('stop_condition',df)