import numpy as np
import csv
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt
import copy

def load_data(filename):
    pokemon = []
    with open(filename, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pokemon.append(
                {'HP': row['HP'], 'Attack': row['Attack'], 'Defense': row['Defense'],
                'Sp. Atk': row['Sp. Atk'], 'Sp. Def': row['Sp. Def'], 'Speed': row['Speed']
                 }
            )
    return pokemon

def calc_features(pokemon):
    pokemon_features = np.array([int(pokemon['HP']), int(pokemon['Attack']), int(pokemon['Defense']), int(pokemon['Sp. Atk']),int(pokemon['Sp. Def']), int(pokemon['Speed'])])
    return pokemon_features


def imshow(z):
    plt.figure()
    dn = hierarchy.dendrogram(z)
    plt.show()


def update_distance_matrix(clusters, distance_matrix):
    updated_distance_matrix = []
    print(distance_matrix)
    print((clusters))
    for i in range(0, len(clusters)):
        updated_distance_matrix.append([])
        print("i is:", i)
        for j in range(0, len(clusters)):
            print(updated_distance_matrix)
            print("j is:", j)
            if len(clusters[(list(clusters)[i])]) > 1 and len(clusters[(list(clusters)[j])]) > 1:
                max_dist = 0
                for k in range(0, len(clusters[(list(clusters)[i])])):
                    for l in range(0, len(clusters[(list(clusters)[j])])):
                        if any(isinstance(m, list) for m in clusters[(list(clusters)[i])]) and any(isinstance(m, list) for m in clusters[(list(clusters)[j])]):
                            if clusters[(list(clusters)[i])] == clusters[(list(clusters)[j])]:
                                max_dist = 0
                            else:

                                print(clusters[(list(clusters)[i])][k][0])
                                print(clusters[(list(clusters)[j])][l][0])
                                a = distance_matrix[clusters[(list(clusters)[i])][k][0]][clusters[(list(clusters)[j])][l][0]]
                                b = distance_matrix[clusters[(list(clusters)[i])][k][0]][clusters[(list(clusters)[j])][l][1]]
                                c = distance_matrix[clusters[(list(clusters)[i])][k][1]][clusters[(list(clusters)[j])][l][0]]
                                d = distance_matrix[clusters[(list(clusters)[i])][k][1]][clusters[(list(clusters)[j])][l][1]]
                                e = max(a, b, c, d)
                                if e > max_dist:max_dist = e
                        elif any(isinstance(m, list) for m in clusters[(list(clusters)[i])]) and not any(isinstance(m, list) for m in clusters[(list(clusters)[j])]):
                            if len(clusters[(list(clusters)[i])][k]) > 1:
                                a = distance_matrix[clusters[(list(clusters)[i])][k][0]][clusters[(list(clusters)[j])][l]]
                                b = distance_matrix[clusters[(list(clusters)[i])][k][1]][clusters[(list(clusters)[j])][l]]
                                c = max(a, b)
                                if c > max_dist:max_dist = c
                            else:
                                max_dist1 = distance_matrix[clusters[(list(clusters)[i])][k][0]][clusters[(list(clusters)[j])][l]]
                                if max_dist1 > max_dist: max_dist = max_dist1
                        elif not any(isinstance(m, list) for m in clusters[(list(clusters)[i])]) and any(isinstance(m, list) for m in clusters[(list(clusters)[j])]):
                            if len(clusters[(list(clusters)[j])][l]) > 1:
                                a = distance_matrix[clusters[(list(clusters)[i])][k]][clusters[(list(clusters)[j])][l][0]]
                                b = distance_matrix[clusters[(list(clusters)[i])][k]][clusters[(list(clusters)[j])][l][1]]
                                c = max(a, b)
                                if c > max_dist:max_dist = c
                            else:
                                max_dist1 = distance_matrix[clusters[(list(clusters)[i])][k]][clusters[(list(clusters)[j])][l][0]]
                                if max_dist1 > max_dist: max_dist = max_dist1
                        elif not any(isinstance(m, list) for m in clusters[(list(clusters)[i])]) and not any(isinstance(m, list) for m in clusters[(list(clusters)[j])]):
                            if clusters[(list(clusters)[i])] == clusters[(list(clusters)[j])]:
                                max_dist = 0
                            else:
                                max_dist1 = distance_matrix[clusters[(list(clusters)[i])][k]][clusters[(list(clusters)[j])][l]]
                                if max_dist1 > max_dist:
                                    max_dist = max_dist1
                updated_distance_matrix[i].append(max_dist)

            elif len(clusters[(list(clusters)[i])]) > 1 and len(clusters[(list(clusters)[j])]) == 1:
                max_dist = 0
                if any(isinstance(m, list) for m in clusters[(list(clusters)[i])]):
                    for k in range(0, len(clusters[(list(clusters)[i])])):
                        if len(clusters[(list(clusters)[i])][k]) > 1:
                            a = distance_matrix[clusters[(list(clusters)[i])][k][0]][clusters[(list(clusters)[j])][0]]
                            b = distance_matrix[clusters[(list(clusters)[i])][k][1]][clusters[(list(clusters)[j])][0]]
                            c = max(a, b)
                            if c > max_dist: max_dist = c
                        elif len(clusters[(list(clusters)[i])][k]) == 1:
                            max_dist1 = distance_matrix[clusters[(list(clusters)[i])][k][0]][clusters[(list(clusters)[j])][0]]
                            if max_dist1 > max_dist: max_dist = max_dist1
                else:
                    for k in range(0, len(clusters[(list(clusters)[i])])):
                        a = distance_matrix[clusters[(list(clusters)[i])][k]][clusters[(list(clusters)[j])][0]]
                        b = distance_matrix[clusters[(list(clusters)[i])][k]][clusters[(list(clusters)[j])][0]]
                        c = max(a, b)
                        if c > max_dist: max_dist = c
                updated_distance_matrix[i].append(max_dist)

            elif len(clusters[(list(clusters)[i])]) == 1 and len(clusters[(list(clusters)[j])]) > 1:
                max_dist = 0
                if any(isinstance(m, list) for m in clusters[(list(clusters)[j])]):
                    for k in range(0, len(clusters[(list(clusters)[j])])):
                        if len(clusters[(list(clusters)[j])][k]) > 1:
                            a = distance_matrix[clusters[(list(clusters)[i])][0]][clusters[(list(clusters)[j])][k][0]]
                            b = distance_matrix[clusters[(list(clusters)[i])][0]][clusters[(list(clusters)[j])][k][1]]
                            c = max(a, b)
                            if c > max_dist: max_dist = c
                        elif len(clusters[(list(clusters)[j])][k]) == 1:
                            max_dist1 = distance_matrix[clusters[(list(clusters)[i])][0]][clusters[(list(clusters)[j])][k][0]]
                            if max_dist1 > max_dist: max_dist = max_dist1
                else:
                    for k in range(0, len(clusters[(list(clusters)[j])])):
                        a = distance_matrix[clusters[(list(clusters)[i])][0]][clusters[(list(clusters)[j])][k]]
                        b = distance_matrix[clusters[(list(clusters)[i])][0]][clusters[(list(clusters)[j])][k]]
                        c = max(a, b)
                        if c > max_dist: max_dist = c
                updated_distance_matrix[i].append(max_dist)
            elif len(clusters[(list(clusters)[i])]) == 1 and len(clusters[(list(clusters)[j])]) == 1:
                print(distance_matrix[clusters[(list(clusters)[i])][0]][clusters[(list(clusters)[j])][0]])
                updated_distance_matrix[i].append(distance_matrix[clusters[(list(clusters)[i])][0]][clusters[(list(clusters)[j])][0]])
    print(np.array(updated_distance_matrix))
    return updated_distance_matrix
            #if len(list(clusters)[i]) > 1 and len(list(clusters)[j]) > 1:


    '''
    for i in range(0, len(feature_clusters)):
        updated_distance_matrix.append([])
        for j in range(0, len(feature_clusters)):
            if len(feature_clusters[i]) > 1 and len(feature_clusters[j]) > 1:
                max_dist = 0
                for k in range(0, len(feature_clusters[i])):
                    for l in range(0, len(feature_clusters[j])):
                        if len(feature_clusters[i][k]) > 1 and len(feature_clusters[j][l]) > 1:
                            a = np.linalg.norm(feature_clusters[i][k][0] - feature_clusters[j][l][0])
                            b = np.linalg.norm(feature_clusters[i][k][0] - feature_clusters[j][l][1])
                            c = np.linalg.norm(feature_clusters[i][k][1] - feature_clusters[j][l][0])
                            d = np.linalg.norm(feature_clusters[i][k][1] - feature_clusters[j][l][1])
                            e = max(a,b,c,d)
                            if e > max_dist:
                                max_dist = e
                        elif len(feature_clusters[i][k]) > 1 and len(feature_clusters[j][l]) == 1:
                            a = np.linalg.norm(feature_clusters[i][k][0] - feature_clusters[j][l])
                            b = np.linalg.norm(feature_clusters[i][k][1] - feature_clusters[j][l])
                            c = max(a, b)
                            if c > max_dist:
                                max_dist = c
                        elif len(feature_clusters[i][k]) == 1 and len(feature_clusters[j][l]) > 1:
                            a = np.linalg.norm(feature_clusters[i][k] - feature_clusters[j][l][0])
                            b = np.linalg.norm(feature_clusters[i][k] - feature_clusters[j][l][1])
                            c = max(a, b)
                            if c > max_dist:
                                max_dist = c
                        elif len(feature_clusters[i][k]) == 1 and len(feature_clusters[j][l]) == 1:
                            max_dist1 = np.linalg.norm(feature_clusters[i][k] - feature_clusters[j][l])
                            if max_dist1 > max_dist:
                                max_dist = max_dist1
                updated_distance_matrix[i].append(max_dist)
            elif len(feature_clusters[i]) > 1 and len(feature_clusters[j]) == 1:
                max_dist = 0
                for k in range(0, len(feature_clusters[i])):
                    if len(feature_clusters[i][k]) > 1:
                        a = np.linalg.norm(feature_clusters[i][k][0] - feature_clusters[j])
                        b = np.linalg.norm(feature_clusters[i][k][1] - feature_clusters[j])
                        c = max(a, b)
                        if c > max_dist: max_dist = c
                    elif len(feature_clusters[i][k]) == 1:
                        max_dist1 = np.linalg.norm(feature_clusters[i][k] - feature_clusters[j])
                        if max_dist1 > max_dist: max_dist = max_dist1
                updated_distance_matrix[i].append(max_dist)
            elif len(feature_clusters[i]) == 1 and len(feature_clusters[j]) > 1:
                max_dist = 0
                for k in range(0, len(feature_clusters[j])):
                    if len(feature_clusters[j][k]) > 1:
                        a = np.linalg.norm(feature_clusters[i] - feature_clusters[j][k][0])
                        b = np.linalg.norm(feature_clusters[i] - feature_clusters[j][k][1])
                        c = max(a, b)
                        if c > max_dist: max_dist = c
                    elif len(feature_clusters[j][k]) == 1:
                        max_dist1 = np.linalg.norm(feature_clusters[i]-feature_clusters[j][k])
                        if max_dist1 > max_dist: max_dist = max_dist1
                updated_distance_matrix[i].append(max_dist)
            elif len(feature_clusters[i]) == 1 and len(clusters[j]) == 1:
                updated_distance_matrix[i].append(np.linalg.norm(feature_clusters[i]-feature_clusters[j]))
        '''




def merge_points(clusters, point1, point2):
    print("point1:", point1)
    print("point2:", point2)
    print(clusters)
    print("Need to merge:", list(clusters)[point1], "and", list(clusters)[point2])
    print("That is:", clusters[list(clusters)[point1]], "and", clusters[list(clusters)[point2]])

    last_key = list(clusters)[-1]
    clusters[last_key+1] = []
    if len(clusters[list(clusters)[point1]]) > 1 or len(clusters[list(clusters)[point2]]) > 1:
        clusters[last_key+1].append(clusters[list(clusters)[point2]])
        clusters[last_key+1].append(clusters[list(clusters)[point1]])

    else:
        #print(clusters[list(clusters)[point1]][0])
        #print(clusters[list(clusters)[point2]][0])
        clusters[last_key+1].append(clusters[list(clusters)[point1]][0])
        clusters[last_key+1].append(clusters[list(clusters)[point2]][0])
    key_point1 = list(clusters)[point1]
    key_point2 = list(clusters)[point2]
    clusters.pop(key_point1)
    clusters.pop(key_point2)
    return clusters

def find_feature_clusters(clusters, pokemon_features):
    temp = []
    for key in clusters.keys():
        if len(clusters[key]) > 1:
            multi_features = []
            for j in clusters[key]:
                multi_features.append(pokemon_features[j])
            temp.append(multi_features)
        else:
            temp.append(pokemon_features[clusters[key][0]])
    return temp

def change_distance_matrix(old_cluster, clusters, distance_matrix):
    updated_distance_matrix = []
    old_cluster_list = list(old_cluster.keys())
    old_cluster_val_list = list(old_cluster.values())
    for i in range(0, len(clusters) - 1):
        updated_distance_matrix.append([])
        for j in range(0, len(clusters) - 1):
            updated_distance_matrix[i].append(distance_matrix[old_cluster_list.index(list(clusters)[i])][old_cluster_list.index(list(clusters)[j])])
    new_cluster = []
    for j in range(0, len(clusters) - 1):
        val1 = clusters[list(clusters)[-1]][0]
        val2 = clusters[list(clusters)[-1]][-1]
        if val1 not in old_cluster_val_list:
            temp = val1
            val1 = []
            val1.append(temp)
        if val2 not in old_cluster_val_list:
            temp = val2
            val2 = []
            val2.append(temp)
        new_cluster.append(max(distance_matrix[old_cluster_list.index(list(clusters)[j])][old_cluster_val_list.index(val1)],distance_matrix[old_cluster_list.index(list(clusters)[j])][old_cluster_val_list.index(val2)]))
        updated_distance_matrix[j].append(max(distance_matrix[old_cluster_list.index(list(clusters)[j])][old_cluster_val_list.index(val1)],distance_matrix[old_cluster_list.index(list(clusters)[j])][old_cluster_val_list.index(val2)]))
    new_cluster.append(0.0)
    updated_distance_matrix.append(new_cluster)
    return updated_distance_matrix

def hac(pokemon_features):
    distance_matrix = []
    for i in range(0, len(pokemon_features)):
        distance_matrix.append([])
        for j in range(0, len(pokemon_features)):
            distance_matrix[i].append(np.linalg.norm(pokemon_features[i]-pokemon_features[j]))
    z = []
    clusters = dict()
    npokemon = dict()
    for i in range(0, len(pokemon_features)):
        clusters[i] = []
        clusters[i].append(i)
        npokemon[i] = 1

    for i in range(0, len(pokemon_features) - 1):
        z.append([])
        min_vals = []
        #print("Distance matrix:", np.array(distance_matrix))
        for j in range(0, len(distance_matrix)):
            m = np.sort(distance_matrix[j])
            #print("m val:",m)
            for k in m:
                if k > 0:
                    min_val = k
                    min_vals.append(k)
                    break
                else:
                    continue
        smallest_val = min(min_vals)
        #print("Smallest val:", smallest_val)

        result = np.where(distance_matrix == smallest_val)
        #print("Result:",result)
        if result[0][0] == result[0][1]:
            ind1 = result[0][2]
            ind2 = result[0][3]
        else:
            ind1 = result[0][0]
            ind2 = result[0][1]
        z[i].append(list(clusters)[ind1])
        z[i].append(list(clusters)[ind2])
        z[i].append(smallest_val)
        old_value = copy.deepcopy(clusters)
        clusters = merge_points(clusters, ind1, ind2)
        #print("Clusters:", clusters)
        last_cluster = list(clusters.values())[-1]
        n = str(last_cluster).count(",")+1
        #print("No of pokemon:", n)
        distance_matrix = change_distance_matrix(old_value, clusters, distance_matrix)

        z[i].append(n)  # need to fix
        #print("Z:", z)

    z = np.array(z)
    #print(z)
    return z

def imshow_hac(z):
    plt.figure()
    dn = hierarchy.dendrogram(z)
    plt.show()


poke = load_data('/Users/mnijagal/Documents/University/Sem-1/AI_CS540/HW4/Pokemon.csv')
z1 = hac([calc_features(row) for row in load_data('Pokemon.csv')][:30])
imshow_hac(z1)
#z2 = hac_scipy([calc_features(row) for row in load_data('Pokemon.csv')][:30])
