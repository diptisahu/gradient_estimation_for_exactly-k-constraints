import torch
from torch.distributions import MultivariateNormal, Normal

k = 0
def find_covariance(sigmas):
    reduced_covariance_matrix = torch.zeros(len(sigmas)-1, len(sigmas)-1)
    squared_sigmas = torch.sum(sigmas)
    for i in range(len(reduced_covariance_matrix)):
        for j in range(len(reduced_covariance_matrix)):
            if i == j:
                reduced_covariance_matrix[i][j] = sigmas[i] - torch.square(sigmas[i])/squared_sigmas
            else:
                reduced_covariance_matrix[i][j] = - sigmas[i]*sigmas[j]/squared_sigmas

    return reduced_covariance_matrix

def find_mean(means, sigmas, covariance_matrix):
    reduced_mean = torch.zeros(len(means)-1)
    c = (k - means[-1])/sigmas[-1]
    for i in range(len(reduced_mean)):
        reduced_mean[i] = torch.sum(covariance_matrix[i]*torch.add(means[i]/sigmas[i], c))

    return reduced_mean

def find_marginal_conditional_distribution(i, means, sigmas):
    conditional_mean = means[i] + sigmas[i]*(k - torch.sum(means))/torch.sum(sigmas)
    conditional_var = sigmas[i] - torch.square(sigmas[i])/torch.sum(sigmas)
    return torch.normal(mean = conditional_mean, std = torch.sqrt(conditional_var))

mu_all = torch.tensor([1, -1, 1, -1, 1, -1], dtype=torch.double)
sigma_all = torch.tensor([0.01, 0.01, 0.01, 0.01, 0.01, 0.02], dtype=torch.double)

constraint_covariance = find_covariance(sigma_all)
# print(constraint_covariance)

constraint_mu = find_mean(mu_all, sigma_all, constraint_covariance)
# print(constraint_mu)

multivariate_normal = MultivariateNormal(constraint_mu, constraint_covariance)
# print(multivariate_normal.sample())

for i in range(len(mu_all)):
    print(find_marginal_conditional_distribution(i, mu_all, sigma_all))