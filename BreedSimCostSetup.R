# 8 Sept 2021
# For the moment, the decision is to have the same burn-in for 
# all iterations of the breeding scheme
# Future effort will have to go to seeing what variation is 
# caused by different burn-ins

## ----load packages----------------------------
ip <- installed.packages()
packages_used <- c("AlphaSimR", "tidyverse",
                   "workflowr", "here", "devtools")
for (package in packages_used){
  if (!(package %in% ip[,"Package"])) install.packages(package)
}
library(tidyverse)

packages_devel <- c("BreedSimCost")
for (package in packages_devel){
  if (!(package %in% ip[,"Package"])){
    devtools::install_github(paste0("jeanlucj/", package), ref="main",
                             build_vignettes=F)
  }
}
library(BreedSimCost)

here::i_am("BreedSimCostSetup.R")

random_seed <- 567890
set.seed(random_seed)


## ----Initialize program-----------------------------------------------
bsd <- initializeProgram(
         here::here("data", "FounderCtrlFile.txt"),
         here::here("data", "SchemeCtrlFile.txt"),
         here::here("data", "CostsCtrlFile.txt"),
         here::here("data", "OptimizationCtrlFile.txt")
       )


## ----Fill variety development pipeline------------------
# Year 1
bsd$year <- bsd$year+1
bsd <- makeVarietyCandidates(bsd)

bsd$entries <- bsd$varietyCandidates@id
bsd <- runVDPtrial(bsd, "SDN")

parents <- selectParentsBurnIn(bsd)
bsd <- makeCrossesBurnIn(bsd, parents)

# Year 2
bsd$year <- bsd$year+1
bsd <- makeVarietyCandidates(bsd)

bsd <- chooseTrialEntries(bsd, toTrial="SDN")
bsd <- runVDPtrial(bsd, "SDN")
bsd <- chooseTrialEntries(bsd, fromTrial="SDN", toTrial="CET")
bsd <- runVDPtrial(bsd, "CET")

parents <- selectParentsBurnIn(bsd)
bsd <- makeCrossesBurnIn(bsd, parents)

# Year 3 and onward
for (burnIn in 1:bsd$nBurnInCycles){
  bsd$year <- bsd$year+1
  bsd <- makeVarietyCandidates(bsd)

  bsd <- chooseTrialEntries(bsd, toTrial="SDN")
  bsd <- runVDPtrial(bsd, "SDN")
  bsd <- chooseTrialEntries(bsd, fromTrial="SDN", toTrial="CET")
  bsd <- runVDPtrial(bsd, "CET")
  bsd <- chooseTrialEntries(bsd, fromTrial="CET", toTrial="PYT")
  bsd <- runVDPtrial(bsd, "PYT")

  parents <- selectParentsBurnIn(bsd)
  bsd <- makeCrossesBurnIn(bsd, parents)
}

burnedInBSD <- bsd

budget_constraints <- bsd$initBudget[c("minPICbudget", "minLastStgBudget")]
budget_constraints <- c(budget_constraints, bsd$initBudget[grep("ratio", names(bsd$initBudget))])
