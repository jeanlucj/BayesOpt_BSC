
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
