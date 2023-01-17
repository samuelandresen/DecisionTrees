import numpy as np
import copy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from collections import deque


# -------------------------------------------------------------------------------------------
# PLOTTING TREES BONUS
fig, ax = plt.subplots(figsize=(15, 6))
plot = False

def plot_graph(tree, xmin, xmax, ymin, ymax, gap, num_of_branches=-1):
    queue = deque([(tree, xmin, xmax, ymin, ymax)])
    counter = 0
    while len(queue) > 0:
        i = queue.popleft()
        tree = i[0]
        xmin = i[1]
        xmax = i[2]
        ymin = i[3]
        ymax = i[4]
        if 'attribute' in tree:
            label = str(tree['value'])

        center = xmin + (xmax - xmin) / 2.0
        d = (center - xmin) / 2.0

        if 'l_branch' in tree:
            queue.append((tree['l_branch'], xmin, center, ymin, ymax - gap))
            ax.annotate(label, xy=(center - d, ymax - gap), xytext=(center, ymax), arrowprops=dict(arrowstyle="-"), )

        if 'r_branch' in tree:
            queue.append((tree['r_branch'], center, xmax, ymin, ymax - gap))
            ax.annotate(label, xy=(center + d, ymax - gap), xytext=(center, ymax), arrowprops=dict(arrowstyle="-"), )

        if not('l_branch' in tree) and not('r_branch' in tree):
            an1 = ax.annotate(tree['leaf'], xy=(center, ymax), xycoords="data", va="bottom", ha="center",
                              bbox=dict(boxstyle="round", fc="w"))
        counter += 1

        if counter == num_of_branches:
            break

# loading dataset
def load_dataset():
    return np.loadtxt('./wifi_db/clean_dataset.txt')


# entropy of each leaf gives the avg amount of information
def H(training_dataset):
    dataset_size = len(training_dataset)
    value, counts = np.unique(training_dataset, return_counts=True)
    # probability of selecting data with a specific leaf
    probability = counts / dataset_size
    # entropy formula
    entropy = -sum(probability * np.log2(probability))
    return entropy


# remainder calculates the (weighted) avg entropy of the produced subsets
def Remainder(l_dataset, r_dataset):
    # weightings are represented by the proportion of each subset
    l_weight = len(l_dataset) / (len(l_dataset) + len(r_dataset))
    r_weight = len(r_dataset) / (len(l_dataset) + len(r_dataset))
    # multiply the entropy of each subset by their respective weightings and sum them together
    remainder = (l_weight * H(l_dataset)) + (r_weight * H(r_dataset))
    return remainder


# information gain is the difference between the initial entropy and the avg entropy of the produced subsets
def Gain(training_dataset, l_dataset, r_dataset):
    return H(training_dataset) - Remainder(l_dataset, r_dataset)


# finding the value with the highest gain within each attribute
def OPTIMAL_VALUE(training_dataset, index):
    values = np.unique(training_dataset[:, index])
    sorted = training_dataset[training_dataset[:, index].argsort()]

    # highest_gain = (value, value_gain, l_dataset, r_dataset)
    highest_gain = (0, -float('inf'), [], [])

    # consider split points between two values
    for value in range(len(values) - 1):
        median_value = (values[value] + values[value + 1]) / 2
        l_dataset = sorted[np.where(sorted[:, index] < median_value)]
        r_dataset = sorted[np.where(sorted[:, index] >= median_value)]

        # calculate gain for each split point
        value_gain = Gain(sorted, l_dataset, r_dataset)

        if value_gain > highest_gain[1]:
            highest_gain = (median_value, value_gain, l_dataset, r_dataset)

    return highest_gain


# choose attribute and value that results in the highest information gain
def FIND_SPLIT(training_dataset):
    highest_gain = -float('inf')

    # optimal_split = (attribute, value, l_dataset, r_dataset)
    optimal_split = (0, 0, [], [])

    # find best attribute using OPTIMAL_VALUE() for each attribute
    for attribute in range(len(training_dataset[0]) - 1):
        value, value_gain, l_dataset, r_dataset = OPTIMAL_VALUE(training_dataset, attribute)
        if value_gain > highest_gain:
            highest_gain = value_gain
            optimal_split = (attribute, value, l_dataset, r_dataset)

    return optimal_split


def decision_tree_learning(training_dataset, depth=0):
    # if all samples have the same leaf then
    # return (a leaf node with this value, depth)
    if len(np.unique(training_dataset[:, -1])) == 1:
        leaf_node = {'leaf': int(training_dataset[:, -1][0])}
        return leaf_node, depth

    else:
        attribute, value, l_dataset, r_dataset = FIND_SPLIT(training_dataset)

        # recursively call decision_tree_learning() on l_branch and r_branch branches of tree
        l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
        r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)

        # node = new decision tree with root as split value
        node = {
            'attribute': attribute,
            'value': value,
            'l_branch': l_branch,
            'r_branch': r_branch
        }
        return (node, max(l_depth, r_depth))


# confusion matrix shows number of occurrences for tp, tn, fp, fn
def confusion_matrix(predicted, true):
    confusion_matrix = np.zeros((4, 4), dtype=int)
    for i in range(len(predicted)):
        if int(predicted[i]) == 1 and int(true[i]) == 1:
            confusion_matrix[0, 0] += 1
        elif int(predicted[i]) == 1 and int(true[i]) == 2:
            confusion_matrix[0, 1] += 1
        elif int(predicted[i]) == 1 and int(true[i]) == 3:
            confusion_matrix[0, 2] += 1
        elif int(predicted[i]) == 1 and int(true[i]) == 4:
            confusion_matrix[0, 3] += 1
        elif int(predicted[i]) == 2 and int(true[i]) == 1:
            confusion_matrix[1, 0] += 1
        elif int(predicted[i]) == 2 and int(true[i]) == 2:
            confusion_matrix[1, 1] += 1
        elif int(predicted[i]) == 2 and int(true[i]) == 3:
            confusion_matrix[1, 2] += 1
        elif int(predicted[i]) == 2 and int(true[i]) == 4:
            confusion_matrix[1, 3] += 1
        elif int(predicted[i]) == 3 and int(true[i]) == 1:
            confusion_matrix[2, 0] += 1
        elif int(predicted[i]) == 3 and int(true[i]) == 2:
            confusion_matrix[2, 1] += 1
        elif int(predicted[i]) == 3 and int(true[i]) == 3:
            confusion_matrix[2, 2] += 1
        elif int(predicted[i]) == 3 and int(true[i]) == 4:
            confusion_matrix[2, 3] += 1
        elif int(predicted[i]) == 4 and int(true[i]) == 1:
            confusion_matrix[3, 0] += 1
        elif int(predicted[i]) == 4 and int(true[i]) == 2:
            confusion_matrix[3, 1] += 1
        elif int(predicted[i]) == 4 and int(true[i]) == 3:
            confusion_matrix[3, 2] += 1
        elif int(predicted[i]) == 4 and int(true[i]) == 4:
            confusion_matrix[3, 3] += 1

    return confusion_matrix


# accuracy = (tp + tn) / (tp + tn + fp + fn)
def get_accuracy(matrix):
    return np.sum(np.diagonal(matrix)) / np.sum(matrix)


# evaluation returns the confusion matrix and accuracy of the tree
def evaluation(data, tree):
    true_leafs = []
    predicted_leafs = []

    # storing the true labels
    for value in data:
        node = tree
        attribute_values = value[:-1]
        true_leaf = int(value[-1])
        true_leafs.append(true_leaf)

        while True:
            # store predicted labels for leaf nodes
            if 'leaf' in node:
                predicted_leaf = node['leaf']
                predicted_leafs.append(predicted_leaf)
                break
            # travel down l_branch of tree
            elif attribute_values[node['attribute']] < node['value']:
                node = node['l_branch']
            # travel down r_branch of tree
            else:
                node = node['r_branch']

    matrix = confusion_matrix(predicted_leafs, true_leafs)

    return matrix, get_accuracy(matrix)


# divide dataset into k folds
def fold_data(dataset, k):
    data = np.copy(dataset)
    np.random.shuffle(data)
    data_split = np.split(data, k)
    return data_split


# splitting folds into training data and test data
def get_training_and_test_data(folds, index):
    test_data = folds[index]

    if index == 0:
        training_data = folds[1]
        i = 2
    else:
        training_data = folds[0]
        i = 1

    while i < len(folds):
        if i != index:
            training_data = np.vstack((training_data, folds[i]))
        i += 1

    return training_data, test_data


# finding leaf nodes
def predict(tree, x):
    if 'leaf' in tree:
        return tree['leaf']
    else:
        if x[tree['attribute']] < tree['value']:
            return predict(tree['l_branch'], x)
        else:
            return predict(tree['r_branch'], x)


# eval_acc() finds the number of times a predicted label appears in the dataset
def eval_acc(dataset, tree):
    data_no_attributes = dataset[:, :-1]
    expected_values = dataset[:, -1]
    predicted_values = []

    for i in range(len(data_no_attributes)):
        predicted_val = predict(tree, data_no_attributes[i])
        predicted_values.append(predicted_val)

    return np.count_nonzero(predicted_values == expected_values)


# splitting dataset
def split_dataset(dataset, split):
    x = dataset[dataset[:, split[0]].argsort()]
    l_dataset = x[x[:, split[0]] <= split[1]]
    r_dataset = x[~(x[:, split[0]] <= split[1])]
    return l_dataset, r_dataset


# inner_pruning() prunes a node if it is connected to 2 leaf nodes
def inner_pruning(dataset, tree, check=0):
    if 'leaf' in tree['l_branch'] and 'leaf' in tree['r_branch']:

        original_acc = eval_acc(dataset, tree)
        # number of times the label for the left leaf appears in the dataset
        left_acc = eval_acc(dataset, tree['l_branch'])
        # number of times the label for the right leaf appears in the dataset
        right_acc = eval_acc(dataset, tree['r_branch'])

        # if the left leaf is the majority class label assign the left leaf label to the parent node
        if left_acc >= right_acc and left_acc >= original_acc:
            tree = tree['l_branch']
            check += 1
        # if the right leaf is the majority class label assign the right leaf label to the parent node
        elif right_acc > left_acc and right_acc > original_acc:
            tree = tree['r_branch']
            check += 1
    # recursively call inner_pruning() until you get to a node connected to two leaf nodes
    else:
        l_dataset, r_dataset = split_dataset(dataset, [tree['attribute'], tree['value']])
        if 'l_branch' in tree:
            if not ('leaf' in tree['l_branch']):
                tree['l_branch'], check = inner_pruning(l_dataset, tree['l_branch'])
        if 'r_branch' in tree:
            if not ('leaf' in tree['r_branch']):
                tree['r_branch'], check = inner_pruning(r_dataset, tree['r_branch'])

    return tree, check


# finds the maximum depth of a tree
def get_depth(tree, depth=0):
    if 'leaf' in tree:
        return depth
    return max(get_depth(tree['l_branch'], depth + 1), get_depth(tree['r_branch'], depth + 1))


def stack_arrays_without_index(folds, discount):
    start = 0
    if 0 in discount:
        start = 1
    elif 1 in discount:
        start = 0

    stack = folds[start]

    start += 1

    while start < len(folds):
        if not (start in discount):
            stack = np.vstack((stack, folds[start]))
        start += 1
    return stack


# use this to only consider training and validation folds
def get_folds_excluding_test(folds, index):
    new_folds = []
    for i in range(len(folds)):
        if i != index:
            new_folds.append(folds[i])
    return new_folds


def cross_validation(dataset, k=10):
    # divide dataset into 10 folds
    folds = fold_data(dataset, k)

    total_accuracy_unpruned = []
    total_accuracy_pruned = []
    total_matrix_unpruned = []
    total_matrix_pruned = []

    # iterate 10 times each time testing on a different portion of the data
    for i in range(k):
        # initially split data into 9 folds for training and 1 fold for testing
        training_data, test_data = get_training_and_test_data(folds, i)
        tree, depth = decision_tree_learning(training_data, 0)
        # evaluate the performance of the unpruned tree
        matrix_unpruned, accuracy_unpruned = evaluation(test_data, tree)
        if plot:
            plot_graph(tree, 0.0, 1.0, 0.0, 1.0, 1.0/depth, 50)

            plt.show()

        # now split data into 8 folds for training, 1 fold for validation and 1 fold for testing
        for j in range(k - 1):
            folds_excluding_test = get_folds_excluding_test(folds, i)
            new_training_set = stack_arrays_without_index(folds_excluding_test, [j])
            validation = folds_excluding_test[j]
            u_tree, depth1 = decision_tree_learning(new_training_set, 0)

            tree1 = copy.deepcopy(u_tree)
            check = []

            # prune tree and check accuracy, repeat until accuracy no longer improves
            while check != 0:
                pruned, check = inner_pruning(validation, tree1)
                matrix_pruned, accuracy_pruned = evaluation(test_data, pruned)
                tree1 = copy.deepcopy(pruned)

            total_matrix_pruned.append(matrix_pruned)
            total_accuracy_pruned.append(accuracy_pruned)
        total_matrix_unpruned.append(matrix_unpruned)
        total_accuracy_unpruned.append(accuracy_unpruned)

    confusion_matrix_unpruned = np.array(total_matrix_unpruned).mean(axis=0)
    avg_matrix_unpruned = np.around(confusion_matrix_unpruned)
    avg_matrix_unpruned = avg_matrix_unpruned.astype(int)

    avg_acc_u = np.mean(total_accuracy_unpruned)
    avg_acc_p = np.mean(total_accuracy_pruned)

    confusion_matrix_pruned = np.array(total_matrix_pruned).mean(axis=0)
    avg_matrix_pruned = np.around(confusion_matrix_pruned)
    avg_matrix_pruned = avg_matrix_pruned.astype(int)

    return avg_matrix_unpruned, avg_matrix_pruned, avg_acc_u, avg_acc_p


acc_u, acc_p, avg_acc_u, avg_acc_p = cross_validation(load_dataset())

print("unpruned confusion matrix")
print(acc_u, "\n")
print("pruned confusion matrix")
print(acc_p, "\n")
print("unpruned accuracy = ", avg_acc_u)
print("pruned accuracy = ", avg_acc_p)




