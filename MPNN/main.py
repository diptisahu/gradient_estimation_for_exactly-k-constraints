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

MAX_EPOCHS = 3000
BATCH_SIZE = 64
MAX_ITERATIONS = 10
random.seed(42)

GNN_LAYERS = 4
EMBEDDING_SIZE = 20
HIDDEN_FEATURES_SIZE = 40
PATIENCE_THRESHOLD = 150
LEARNING_RATE = 0.005

device = torch.device('cuda:2')
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

loss_all = np.zeros(MAX_ITERATIONS)
charge_sum_all = np.zeros(MAX_ITERATIONS)
mad_all = np.zeros(MAX_ITERATIONS)
print("Total MOFs: {}".format(len(dataa)))
# module for evaluating
print()

def multivariate_mean_variance(means, sigmas):
    n = len(sigmas)

    A = torch.inverse(torch.diag(torch.pow(sigmas[:-1], -1)))
    B = torch.ones(n-1, n-1).double().to(device) * torch.pow(sigmas[-1], -1)

    covariance_matrix = A - 1/(1 + torch.trace(torch.matmul(B, A))) * torch.matmul(A ,torch.matmul(B, A))

    c = (k - means[-1])/sigmas[-1]
    reduced_mean = torch.matmul(covariance_matrix, torch.ones(n-1).to(device)*c + torch.div(means[:-1], sigmas[:-1]))

    return reduced_mean

print("BATCH_SIZE = ", BATCH_SIZE)
print("GNN_LAYERS = ", GNN_LAYERS)
print("HIDDEN_FEATURES_SIZE =", HIDDEN_FEATURES_SIZE)
print("EMBEDDING_SIZE = ", EMBEDDING_SIZE)
print("PATIENCE_THRESHOLD = ", PATIENCE_THRESHOLD)

train_losses = {
    'gaussian_cor' : [],
    'gaussian_with_erf_loss' : []
}
valid_losses = {
    'gaussian_cor' : [],
    'gaussian_with_erf_loss' : []
}
test_losses = {
    'gaussian_cor' : [],
    'gaussian_with_erf_loss' : []
}
final_loss = {
    'gaussian_cor' : [],
    'gaussian_with_erf_loss' : [],
    'gaussian_with_erf_loss_model_1': []
}

for iteration in range(MAX_ITERATIONS):
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
                    # print('{} is not in training set, trying again...'.format(element_types[element_index]), end="\r",
                    #     flush=True)
                    unique_flag = False
                    break

    # print('shuffling datasets is done.............!!')

    # ------------------------
    valid_dataset = data_list_shuffled[cut_training:cut_validation]
    test_dataset = data_list_shuffled[cut_validation:]
    train_data_size = len(train_dataset)
    valid_data_size = len(valid_dataset)
    test_data_size = len(test_dataset)

    # print("training crystals: {}".format(train_data_size))
    # print("validation crystals: {}".format(valid_data_size))
    # print("testing crystals: {}".format(test_data_size))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
    # valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    valid_loader = DataLoader(valid_dataset, batch_size=len(valid_dataset))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    systems = ['gaussian_with_erf_loss']
    # system = 'gaussian_with_erf_loss'
    # models = []
    # iteration=0
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

    for system in systems:
    # for i in range(3):
        if system == 'gaussian_cor':
            model, train_loss, valid_loss, test_loss = charge_prediction_system(train_loader, valid_loader, test_loader, NUM_NODE_FEATURES, EMBEDDING_SIZE, GNN_LAYERS, HIDDEN_FEATURES_SIZE, train_data_size, valid_data_size, test_data_size, MAX_EPOCHS, iteration, system, PATIENCE_THRESHOLD, LEARNING_RATE, crit)
            # models.append(model)
        elif system == 'gaussian_with_erf_loss':
            m = len(train_loader)
            model1, train_loss, valid_loss, test_loss = charge_prediction_system(train_loader, valid_loader, test_loader, NUM_NODE_FEATURES, EMBEDDING_SIZE, GNN_LAYERS, HIDDEN_FEATURES_SIZE, train_data_size, valid_data_size, test_data_size, MAX_EPOCHS, iteration, system, PATIENCE_THRESHOLD, LEARNING_RATE, crit)
            model2, train_loss, valid_loss, test_loss = charge_prediction_system(train_loader, valid_loader, test_loader, NUM_NODE_FEATURES, EMBEDDING_SIZE, GNN_LAYERS, HIDDEN_FEATURES_SIZE, train_data_size, valid_data_size, test_data_size, MAX_EPOCHS, iteration, system, PATIENCE_THRESHOLD, LEARNING_RATE, crit)
        # train_losses[system].append(train_loss)
        # valid_losses[system].append(valid_loss)
        # test_losses[system].append(test_loss)
    
    torch.save([model1, model2], 'models.pt')
    # torch.save(test_dataset, 'test_dataset.pt')
    # models = torch.load('models.pt')

    # iteration = 0

    dataa = test_dataset
    # dataa = valid_dataset
    loader = DataLoader(dataa, batch_size=len(dataa))

    variance_charge = []
    sigma_all = []
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            label = data.y.to(device)
            features = data.x.to(device)
            # print("Total Nodes: {}".format(len(label)))

            print("Method \t\t\t MAD")

            for index, system in enumerate(systems):
            # for index in range(1):
                # model = models[index]
                # model.eval()
                if system == 'gaussian_cor':
                    model.eval()
                    pred, _, _, _ = model(data)
                elif system == 'gaussian_cor_with_sampling':
                    pred = model(data, False)
                else:
                    model1.eval()
                    model2.eval()

                    mu_bar, sigma_bar = model1(data)
                    pred1 = torch.empty_like(mu_bar)
                    for i in range(0, data.num_graphs):
                        mu_sample = mu_bar[data.batch == i]
                        sigma_sample = sigma_bar[data.batch == i]

                        reduced_mean = multivariate_mean_variance(mu_sample, sigma_sample)
                        pred1[data.batch == i] = torch.cat((reduced_mean, torch.tensor([0 - torch.sum(reduced_mean)]).to(device)), dim=0)

                    mu_bar, sigma_bar = model2(data)
                    pred2 = torch.empty_like(mu_bar)
                    for i in range(0, data.num_graphs):
                        mu_sample = mu_bar[data.batch == i]
                        sigma_sample = sigma_bar[data.batch == i]

                        reduced_mean = multivariate_mean_variance(mu_sample, sigma_sample)
                        pred2[data.batch == i] = torch.cat((reduced_mean, torch.tensor([0 - torch.sum(reduced_mean)]).to(device)), dim=0)
                    # pred, _ = model(data)

                    pred = (pred1 + pred2)/2

                    loss = crit(pred1, label)
                    final_loss["gaussian_with_erf_loss_model_1"].append(loss.item())
                    print(system+"_model1", "\t\t {:.6f}".format(loss.item()))

                    # loss = crit(pred2, label)
                    # print(system+"_model2", "\t\t {:.6f}".format(loss.item()))

                loss = crit(pred, label)
                print(system, "\t\t {:.6f}".format(loss.item()))
                
                final_loss[system].append(loss.item())
    
for system in systems:
    # train_losses[system] = np.vstack(train_losses[system])
    # train_losses[system] = np.mean(train_losses[system], axis=0)

    # valid_losses[system] = np.vstack(valid_losses[system])
    # valid_losses[system] = np.mean(valid_losses[system], axis=0)

    # test_losses[system] = np.vstack(test_losses[system])
    # test_losses[system] = np.mean(test_losses[system], axis=0)

    mean = np.mean(final_loss[system])
    sum = 0
    for item in final_loss[system]:
        dev = np.absolute(item - mean)
        sum += dev
    
    print(system, mean, sum/len(final_loss[system]))

system = "gaussian_with_erf_loss_model_1"
mean = np.mean(final_loss[system])
sum = 0
for item in final_loss[system]:
    dev = np.absolute(item - mean)
    sum += dev

print(system, mean, sum/len(final_loss[system]))

# hfont = {'fontname':'DejaVu Sans'}
# fontsize_label_legend = 24
# plt.figure(figsize=(8,8), dpi= 80)
# plt.plot(train_losses['gaussian_cor'], label="Baseline train loss", color='blue', linewidth = 1)
# plt.plot(valid_losses['gaussian_cor'], label="Baseline valid loss", color='red', linewidth = 1)
# plt.plot(train_losses['gaussian_with_erf_loss'], label="Sampling train loss", color='black', linewidth = 1)
# plt.plot(valid_losses['gaussian_with_erf_loss'], label="Sampling valid loss", color='green', linewidth = 1)
# plt.plot(test_losses['gaussian_cor'], label="Baseline test loss", color='cyan', linewidth = 1)
# plt.plot(test_losses['gaussian_with_erf_loss'], label="Sampling test loss", color='maroon', linewidth = 1)
# plt.legend(frameon=False, prop={'size': 22})
# plt.xlabel('Epochs', fontsize=fontsize_label_legend, **hfont)
# plt.ylabel('Loss', fontsize=fontsize_label_legend, **hfont)
# plt.legend(frameon=False, prop={"family":"DejaVu Sans", 'size': fontsize_label_legend})
# plt.tick_params(axis='both', which='major', labelsize=17)
# plt.savefig('./results/convergence_curve-erf-loss-300-epochs.png')
# plt.show()
