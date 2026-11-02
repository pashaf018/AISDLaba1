from math import sin, asin, radians, sqrt, cos
from sklearn.neighbors import BallTree
import numpy as np
from random import Random

import pandas as pd
import matplotlib.pyplot as plt
from numpy import ceil

def calculate_distance(latitude1, latitude2, longitude1, longitude2):
    R = 6371.0

    phi1, lam1 = radians(latitude1), radians(longitude1)
    phi2, lam2 = radians(latitude2), radians(longitude2)

    dphi = phi2 - phi1
    dlam = lam2 - lam1

    a = sin(dphi / 2) ** 2 + cos(phi1) * cos(phi2) * sin(dlam / 2) ** 2
    c = 2 * asin(sqrt(a))

    return R * c

def clusterize(index, distances, cluster_index):
    temp_cluster = []

    sum2, ind2 = 0, 0
    for i in range(len(distances)):
        if distances[i][1] > 2000:
            break
        sum2 += distances[i][2]
        if sum2 > 10000:
            sum2 -= distances[i][2]
            ind2 = i
            break

    for i in range(ind2):
        temp_cluster.append([distances[i][0], df.iloc[distances[i][0]]['house_uuid'], float(df.iloc[distances[i][0]]['latitude']), float(df.iloc[distances[i][0]]['longitude'])])

    return temp_cluster


def try_optimize(cluster_houses, df, max_devices, max_radius_m):
    best_center = None
    best_cluster = []
    best_devices_sum = 0
    best_radius = 0.0

    for candidate in cluster_houses:
        distances_from_candidate = []
        for house in cluster_houses:
            dist_m = int(calculate_distance(
                candidate[2], house[2],
                candidate[3], house[3]
            ) * 1000)
            devices = df.at[house[0], 'end_devices_count']
            distances_from_candidate.append([dist_m, house, devices])

        distances_from_candidate.sort(key=lambda x: x[0])

        current_cluster = []
        current_sum = 0
        current_max_radius = 0.0

        for dist, house, device_count in distances_from_candidate:
            if current_sum + device_count <= max_devices and dist <= max_radius_m:
                current_cluster.append(house)
                current_sum += device_count
                current_max_radius = max(current_max_radius, dist)
            else:
                break

        if current_sum > best_devices_sum:
            best_center = candidate
            best_cluster = current_cluster
            best_devices_sum = current_sum
            best_radius = current_max_radius

    return {
        'center': best_center,
        'cluster': best_cluster,
        'devices': best_devices_sum,
        'radius': best_radius
    }


def optimize_cluster_center(cluster_houses, df, cluster_index):

    result_bs1 = try_optimize(cluster_houses, df, max_devices=2000, max_radius_m=1000)
    result_bs2 = try_optimize(cluster_houses, df, max_devices=10000, max_radius_m=2000)

    if 0 < result_bs1['devices'] <= 2000:
        efficiency_bs1 = 50000 / result_bs1['devices']
    else:
        efficiency_bs1 = float('inf')

    if 0 < result_bs2['devices'] <= 10000:
        efficiency_bs2 = 180000 / result_bs2['devices']
    else:
        efficiency_bs2 = float('inf')

    if efficiency_bs1 <= efficiency_bs2:
        chosen = result_bs1
        bs_type = 1
    else:
        chosen = result_bs2
        bs_type = 2

    for house in chosen['cluster']:
        df.at[house[0], 'cluster'] = cluster_index
        df.at[house[0], 'is_visited'] = True

    return chosen, bs_type

def get_neighbors_in_radius(lat, lon, radius_km=2.0):
    point_rad = np.radians([[lat, lon]])
    radius_rad = radius_km / 6371.0

    indices, distances_rad = tree.query_radius(
        point_rad, r=radius_rad, return_distance=True
    )
    indices = indices[0]
    distances_km = distances_rad[0] * 6371.0
    return indices, distances_km


# def choose_best_start(df, tree):
#     unvisited = df[df['is_visited'] == False].index
#     if len(unvisited) == 0:
#         return None
#
#     best_idx = None
#     best_devices = -1  # ← Изменено с 0 на -1
#
#     for idx in unvisited:
#         neighbors, _ = get_neighbors_in_radius(
#             df.at[idx, 'latitude'],
#             df.at[idx, 'longitude'],
#             radius_km=2.0
#         )
#         devices_sum = sum(
#             df.at[n, 'end_devices_count']
#             for n in neighbors
#             if not df.at[n, 'is_visited']
#         )
#         if devices_sum > best_devices:
#             best_devices = devices_sum
#             best_idx = idx
#
#     return best_idx
#

def choose_best_start(df, tree):
    unvisited = df[df['is_visited'] == False].index
    if len(unvisited) == 0:
        return None

    best_idx = unvisited[0]
    max_devices = df.at[best_idx, 'end_devices_count']

    for idx in unvisited[1:]:
        devices = df.at[idx, 'end_devices_count']
        if devices > max_devices:
            max_devices = devices
            best_idx = idx

    return best_idx

# MainProgram
df = pd.read_csv('задание.csv', delimiter=";")
df['is_visited'] = False
df['cluster'] = -1

coords_rad = np.radians(df[['latitude', 'longitude']].values)
tree = BallTree(coords_rad, metric='haversine')

clusters = []
cluster_index = 0
total_devices = df['end_devices_count'].sum()

while True:
    covered_devices = df[df['cluster'] >= 0]['end_devices_count'].sum()
    print("total = " + str(total_devices))
    print("covered devices = " + str(covered_devices))
    if covered_devices >= 0.95 * total_devices:
        break
    if len(clusters) >= 1000:
        break

    index = choose_best_start(df, tree)
    if index is None:
        break

    neighbors_in_2km, distances_km = get_neighbors_in_radius(
        df.at[index, 'latitude'],
        df.at[index, 'longitude'],
        radius_km=2.0
    )

    unvisited_neighbors = [
        (idx, dist) for idx, dist in zip(neighbors_in_2km, distances_km)
        if not df.at[idx, 'is_visited']
    ]

    if len(unvisited_neighbors) < 3:
        df.at[index, 'is_visited'] = True
        continue

    cluster_houses = []
    for idx, dist in unvisited_neighbors:
        cluster_houses.append([
            idx,
            df.at[idx, 'house_uuid'],
            df.at[idx, 'latitude'],
            df.at[idx, 'longitude']
        ])

    res, bs_type = optimize_cluster_center(cluster_houses, df, cluster_index)
    if res['center'] is not None:
        center = [res['center'], bs_type]
        new_cluster = res['cluster']
        cluster_index += 1
        clusters.append([new_cluster, center])

result = []
for data in clusters:
    result.append([data[1][0][1], data[1][1]])
result_df = pd.DataFrame(result, columns=['house_uuid', 'type'])
result_df.to_csv('result.csv', sep=';', index=False)

# Визуализация
colors = df['cluster'].apply(lambda x: x if x >= 0 else -1).values
cmap = plt.get_cmap('tab20')

scatter = plt.scatter(
    df['longitude'], df['latitude'],
    c=colors, cmap=cmap,
    s=10, alpha=0.6, label='Дома'
)

for data in clusters:
    cluster_houses, center_info = data
    center_house, bs_type = center_info
    lat, lon = center_house[2], center_house[3]

    if bs_type == 1:
        plt.scatter(lon, lat, c='red', marker='*', s=200,
                    edgecolors='black', zorder=5,
                    label='БС-1' if 'БС-1' not in plt.gca().get_legend_handles_labels()[1] else "")
        circle = plt.Circle((lon, lat), 0.009, color='red', fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_patch(circle)
    else:
        plt.scatter(lon, lat, c='blue', marker='*', s=300,
                    edgecolors='black', zorder=5,
                    label='БС-2' if 'БС-2' not in plt.gca().get_legend_handles_labels()[1] else "")
        circle = plt.Circle((lon, lat), 0.018, color='blue', fill=False, linestyle='--', alpha=0.5)
        plt.gca().add_patch(circle)


def validate_solution(df, clusters):
    print("--- НАЧАЛО ПРОВЕРКИ РЕШЕНИЯ ---")
    is_valid = True

    if len(clusters) > 1000:
        print(f"❌ ОШИБКА: Превышен лимит БС! Установлено {len(clusters)}, максимум 1000.")
        is_valid = False
    else:
        print(f"✅ Лимит БС соблюден: {len(clusters)} из 1000.")

    covered_devices = df[df['cluster'] >= 0]['end_devices_count'].sum()
    total_devices = df['end_devices_count'].sum()
    coverage = covered_devices / total_devices * 100
    if coverage < 95.0:
        print(f"❌ ОШИБКА: Покрытие недостаточное! {coverage:.2f}% < 95.00%")
        is_valid = False
    else:
        print(f"✅ Покрытие достаточное: {coverage:.2f}% (нужно 95.00%)")

    bs1_count, bs2_count = 0, 0
    for i, data in enumerate(clusters):
        cluster_houses, center_info = data
        center_house, bs_type = center_info

        max_dev = 2000 if bs_type == 1 else 10000
        max_rad_km = 1.0 if bs_type == 1 else 2.0

        if bs_type == 1:
            bs1_count += 1
        else:
            bs2_count += 1

        center_lat, center_lon = center_house[2], center_house[3]

        sum_dev = 0
        for house in cluster_houses:
            dist = calculate_distance(center_lat, house[2], center_lon, house[3])
            if dist > max_rad_km + 0.001:
                print(
                    f"❌ ОШИБКА в БС #{i}: Дом {house[1]} находится в {dist:.3f} км, а радиус БС-{bs_type} всего {max_rad_km} км.")
                is_valid = False
            sum_dev += df.at[house[0], 'end_devices_count']

        if sum_dev > max_dev:
            print(f"❌ ОШИБКА в БС #{i}: Подключено {sum_dev} устройств, а лимит БС-{bs_type} всего {max_dev}.")
            is_valid = False

    if is_valid:
        print("✅ Все БС валидны (радиусы и лимиты устройств соблюдены).")

    total_cost = bs1_count * 50000 + bs2_count * 180000
    print(f"\n--- ИТОГОВАЯ СТАТИСТИКА ---")
    print(f"БС-1: {bs1_count} шт. (по {50000} руб.)")
    print(f"БС-2: {bs2_count} шт. (по {180000} руб.)")
    print(f"💰 Суммарные затраты: {total_cost:,} руб.".replace(",", " "))
    print("--- КОНЕЦ ПРОВЕРКИ ---")

    return is_valid

validate_solution(df, clusters)

plt.title('Кластеры и базовые станции')
plt.xlabel('longitude')
plt.ylabel('latitude')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
plt.savefig('result.png')
