# // ===============================
# // AUTHOR     : Ali Raza
# // CREATE DATE     : Dec 22, 2019
# // PURPOSE     : main function. a wrapper of charge perdiction system for testing and evaluation
# // SPECIAL NOTES: Uses charge_prediction_system that uses data_handling.py and model.py
# // ===============================
# // Change History: 1.0: initial code: wrote and tested.
# // Change History: 2.0: updated code: added mini batches
# //
# //==================================
__author__ = "Ali Raza"
__copyright__ = "Copyright 2019"
__credits__ = []
__license__ = ""
__version__ = "2.0"
__maintainer__ = "ali raza"
__email__ = "razaa@oregonstate.edu"
__status__ = "done"

from data_handling import *
from model import *
from charge_prediction_system import *
from torch_geometric.loader import DataLoader
import os
import numpy as np
import random
import copy
import csv
import torch

print("----------------------------------------------")
print(">>> loading parameters")

GRAPHS_LOCATION = "input"
ONE_HOT_ENCODING_CSV = "../atom_to_int.csv"
TRAINING_SET_CUT = 70  # percentage
VALIDATION_SET_CUT = 10  # percentage
# remaining is test set
MAX_EPOCHS = 3000
BATCH_SIZE = 32
MAX_ITERATIONS = 1
random.seed(42)

GNN_LAYERS = 4
EMBEDDING_SIZE = 10
HIDDEN_FEATURES_SIZE = 30

device = torch.device('cuda')
# crit = torch.nn.MSELoss()
crit = torch.nn.L1Loss()

if not (os.path.exists("results/")):
    os.mkdir('results/')
if not (os.path.exists('results/graphs')):
    os.mkdir('results/graphs')
if not (os.path.exists('results/embedding')):
    os.mkdir('results/embedding')

print("...done")
print("----------------------------------------------")

print("----------------------------------------------")
print(">>> reading graphs and generating data_list")
data_list = data_handling(GRAPHS_LOCATION, READ_LABELS = True)
print("...done")
print("----------------------------------------------")
print()
NUM_NODE_FEATURES = data_list[0]['x'].shape[1]

# dividing data into testing and training
print(">>> reading one-hot encoding")
with open(ONE_HOT_ENCODING_CSV) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    element_types = []
    one_hot_encoding = []
    next(readCSV)
    for row in readCSV:
        element_types.append(row[0])
        one_hot_encoding.append(int(row[1]))

    # sorting them
    indices_sorted_elements = np.argsort(one_hot_encoding)
    element_types = np.array(element_types)[indices_sorted_elements]
    one_hot_encoding = np.array(one_hot_encoding)[indices_sorted_elements]
print("...done")
print("-----------------------------------------------------------------------------------------")

print("----------------------------------------------")
print(">>> shuffling data for different training, validation, and testing sets each run")
data_size = len(data_list)
cut_training = int(data_size * (TRAINING_SET_CUT / 100))
cut_validation = int(data_size * (TRAINING_SET_CUT + VALIDATION_SET_CUT) / 100)
# iteration = 0
data_list_shuffled = copy.deepcopy(data_list)
# making sure training_dataset has all the elements after shuffling
unique_flag = False
while unique_flag == False:
    unique_flag = True
    random.shuffle(data_list_shuffled)
    train_dataset = data_list_shuffled[:cut_training]
    train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))
    for data in train_loader:
        data = data.to(device)
        label = data.y.to(device)
        features = data.x.to(device)
        elements_number = len(features[0])
        for element_index in range(elements_number):
            indices = (features[:, element_index] == 1)
            if len((label[indices])) == 0:  # number of atoms in datasets
                print('{} is not in training set, trying again...'.format(element_types[element_index]), end="\r",
                      flush=True)
                unique_flag = False
                break

print('shuffling datasets is done.............!!')

# ------------------------
valid_dataset = data_list_shuffled[cut_training:cut_validation]
test_dataset = data_list_shuffled[cut_validation:]
train_data_size = len(train_dataset)
valid_data_size = len(valid_dataset)
test_data_size = len(test_dataset)

print("training crystals: {}".format(train_data_size))
print("validation crystals: {}".format(valid_data_size))
print("testing crystals: {}".format(test_data_size))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
# valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

dataa = data_list
loader = DataLoader(dataa, batch_size=len(dataa))
print("total MOFs: {}".format(len(dataa)))

for data in loader:
    data = data.to(device)
    label = data.y.to(device)
    features = data.x.to(device)
    print("total nodes: {}".format(len(label)))

    elements_number = len(features[0])
    total_instances_all = np.zeros(elements_number)
    total_instances_mof_all = np.zeros(elements_number)

    for element_index in range(elements_number):
        indices = (features[:, element_index] == 1)
        label_element = label[indices].cpu().numpy()
        total_instances_all[element_index] = len((label[indices]))  # number of atoms in datasets
        total_instances_mof_all[element_index] = len(
            set(data.batch[indices].cpu().numpy()))  # number of mofs containing that element

    # indices of sorted element
    indices_sorted_elements = np.argsort(total_instances_all)
    indices_sorted_elements = np.flipud(indices_sorted_elements)
    # %-----------------------------------------------------------------------

systems = ['gaussian_cor', 'gaussian_cor_with_erf_loss']
models = []
iteration=0
for system in systems:
    model = charge_prediction_system(train_loader, valid_loader, NUM_NODE_FEATURES, EMBEDDING_SIZE, GNN_LAYERS, HIDDEN_FEATURES_SIZE, train_data_size, valid_data_size, MAX_EPOCHS, iteration, system, crit)
    models.append(model)

torch.save(models, 'models.pt')
torch.save(test_dataset, 'test_dataset.pt')

# models = torch.load('models.pt')

iteration = 0

dataa = test_dataset
loader = DataLoader(dataa, batch_size=len(dataa))

loss_all = np.zeros((len(systems), MAX_ITERATIONS))
charge_sum_all = np.zeros((len(systems), MAX_ITERATIONS))
mad_all = np.zeros((len(systems), MAX_ITERATIONS))
print("Total MOFs: {}".format(len(dataa)))
# module for evaluating
print()

def multivariate_mean_variance(means, sigmas):
    n = len(sigmas)

    A = torch.inverse(torch.diag(torch.pow(sigmas[:-1], -1)))
    B = torch.ones(n-1, n-1).to(device) * torch.pow(sigmas[-1], -1)

    covariance_matrix = A - 1/(1 + torch.trace(torch.matmul(B, A))) * torch.matmul(A ,torch.matmul(B, A))

    c = (k - means[-1])/sigmas[-1]
    reduced_mean = torch.matmul(covariance_matrix, torch.ones(n-1).to(device)*c + torch.div(means[:-1], sigmas[:-1]))

    return reduced_mean

predictions = []
variance_charge = []
sigma_all = []
with torch.no_grad():
    for data in loader:
        data = data.to(device)
        label = data.y.to(device)
        features = data.x.to(device)
        print("Total Nodes: {}".format(len(label)))
        for index, system in enumerate(systems):
            model = models[index]
            model.eval()
            if system == 'gaussian_cor':
                pred, _, _, _ = model(data)
            elif system == 'gaussian_cor_with_sampling':
                pred = model(data, False)
            else:
                mu_bar, sigma_bar = model(data)
                pred = torch.empty_like(mu_bar)
                for i in range(0, data.num_graphs):
                    mu_sample = mu_bar[data.batch == i]
                    sigma_sample = sigma_bar[data.batch == i]

                    reduced_mean = multivariate_mean_variance(mu_sample, sigma_sample)
                    pred[data.batch == i] = torch.cat((reduced_mean, torch.tensor([0 - torch.sum(reduced_mean)]).to(device)), dim=0)
                # pred, _ = model(data)

            predictions.append(pred)

            loss = crit(pred, label)
            sum_charge = ts.scatter_add(pred, data.batch, dim=0)
            mean_tt = ts.scatter_mean(pred, data.batch, dim=0)
            #             variance = E[X^2] - (E[X])^2
            variance_charge.append(
                ts.scatter_mean(torch.mul(pred, pred), data.batch, dim=0) - torch.mul(mean_tt, mean_tt))
            print("Method \t\t\t MAD \t avg_abs_sum_charge \t max_charge_sum \t min_charge_sum")
            print("---------------------------------------------------------------------------------------")
            print(system, "\t\t {:.6f} \t {:.6f} \t {:.6f} \t\t {:.6f}".format(loss.item(), np.average(
                np.absolute(sum_charge.cpu().numpy())), np.max(sum_charge.cpu().numpy()),
                                                                               np.min(sum_charge.cpu().numpy())))

            charge_sum_all[index, iteration] = np.average(np.absolute(sum_charge.cpu().numpy()))

            error_all = pred.cpu().numpy() - label.cpu().numpy()
            mad_all[index, iteration] = np.mean(np.absolute(error_all))
            loss_all[index, iteration] = loss.item()