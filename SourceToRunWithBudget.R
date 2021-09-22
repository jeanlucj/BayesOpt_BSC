# Function to source to get results needed for Bayesian Optimization
# Python rpy2 should have made the percentages vector
# burnedInBSD should exist from BreedSimCostSetup
# runWithBudget returns 
# c(percentages, 
# start_breedPopMean, start_breedPopSD, start_varCandMean, 
# end_breedPopMean, end_breedPopSD, end_varCandMean, 
# realizedBudget)

percentages <- c(percentages, 1 - sum(percentages))
idx <- length(percentages)+1
rwbOut <- runWithBudget(percentages, bsd=burnedInBSD)
percentages <- rwbOut[1:3]
gain <- rwbOut[idx+5] - rwbOut[idx]
