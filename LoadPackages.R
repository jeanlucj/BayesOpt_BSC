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

here::i_am("LoadPackages.R")

random_seed <- 567890
set.seed(random_seed)
