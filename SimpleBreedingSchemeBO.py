# ### Adapt the qNoisy tutorial code for my purposes
import os
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using {} device".format(device))
dtype = torch.double
torch.set_default_dtype(dtype)

N_STAGES = 3
# ### Objective function setup
# The objective function calls R by sourcing this runWithBudget script
import rpy2.robjects as ro
from rpy2.robjects import numpy2ri
numpy2ri.activate()

# X should be a one-dimensional torch tensor
def objective_func(X):
    ro.globalenv['percentages'] = X.numpy()
    ro.r.source('SourceToRunWithBudget.R')
    return torch.tensor(ro.globalenv['percentages']).unsqueeze(-2), torch.tensor(ro.globalenv['gain']).unsqueeze(-2)

# ### Set up the R side of the optimization
ro.r.source('BreedSimCostSetup.R')
budget_constraints = torch.tensor(ro.globalenv['budget_constraints'])
# Try to set this up so that only N_STAGES parameters are being captured
# So, they have to sum to less than 1.0
mb = 1.0 - budget_constraints[0] - budget_constraints[1]
bounds = torch.tensor(
    [[budget_constraints[0], 0.0, 0.0], [mb] * N_STAGES],
    device=device, dtype=dtype)

lr = budget_constraints[4]
inequality_constraints = [
    (torch.tensor([i for i in range(N_STAGES)]), torch.tensor([-1.0] * (N_STAGES)), -(1.0 - budget_constraints[1])), 
    (torch.tensor([0, 1]), torch.tensor([1.0, -budget_constraints[2]]), 0.0),
    (torch.tensor([1, 2]), torch.tensor([1.0, -budget_constraints[3]]), 0.0),
    (torch.tensor([i for i in range(N_STAGES)]), torch.tensor([lr, lr, 1+lr]), lr),
    ]
print(budget_constraints)
print(bounds)
print(inequality_constraints)

# ### Model initialization
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
    
def initialize_model(train_x, train_obj):
    # define model for objective
    surrogate = SingleTaskGP(train_x, train_obj, outcome_transform=Standardize(m=1))
    mll = ExactMarginalLogLikelihood(surrogate.likelihood, surrogate)
    # fit the models
    fit_gpytorch_model(mll)
    return surrogate

# ### Start the overall loop
from botorch import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.optim import optimize_acqf
from botorch.sampling.samplers import SobolQMCNormalSampler
from botorch.exceptions import BadInitialCandidatesWarning

import time
import warnings

warnings.filterwarnings('ignore', category=BadInitialCandidatesWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

N_OPTIMIZATIONS = 5
N_ITER = 64
MC_SAMPLES = 256
NUM_RESTARTS = 30 # 10 * input_dim
RAW_SAMPLES = 600 # 200 * input_dim

TESTING = True

# Storage for the optimizations
opt_traces = []

for optimization in range(N_OPTIMIZATIONS):
    print(f"\nOptimization {optimization:>2} of {N_OPTIMIZATIONS} ", end="")
    # Initial data
    ro.globalenv['testing'] = TESTING
    ro.r.source('GenerateInitialData.R')
    train_x = torch.tensor(ro.globalenv['budgets'])
    train_obj = torch.tensor(ro.globalenv['gains']).unsqueeze(-1)
    best_observed_obj = train_obj.max().item()
    best_observed_vec = [best_observed_obj]

    # run N_ITER rounds of BayesOpt after the initial random batch
    for iteration in range(N_ITER):
        print(f"\nIteration {iteration:>2} of {N_ITER} ", end="")
        surrogate = initialize_model(train_x, train_obj)
    
        # define the qEI using a QMC sampler [I don't understand what this does]
        qmc_sampler = SobolQMCNormalSampler(num_samples=MC_SAMPLES)
            
        # for best_f, use the best observed noisy values as an approximation
        # What does passing the objective function on to the acquisition function do?
        qEI = qExpectedImprovement(
            model=surrogate, 
            best_f=best_observed_obj,
            sampler=qmc_sampler, 
        )

        # optimize the acquisition function
        candidates, _ = optimize_acqf(
            acq_function=qEI,
            bounds=bounds,
            inequality_constraints=inequality_constraints,
            q=1,
            num_restarts=NUM_RESTARTS,
            raw_samples=RAW_SAMPLES,  # used for intialization heuristic
            options={"batch_limit": 5, "maxiter": 200},
        )

        print(candidates)

        # get new observation
        new_perc, new_obj = objective_func(candidates)
                    
        # update training points
        train_x = torch.cat([train_x, new_perc])
        train_obj = torch.cat([train_obj, new_obj])
        best_observed_obj = train_obj.max().item()
        best_observed_vec.append(best_observed_obj)

    opt_traces.append(best_observed_vec)

ro.globalenv['train_x'] = train_x.numpy()
ro.globalenv['train_obj'] = train_obj.numpy()
ro.globalenv['opt_traces'] = opt_traces
ro.r.source('AnalyzeOptData.R')
