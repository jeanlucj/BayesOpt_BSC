# Source this to generate initial data

here::i_am("GenerateInitialData.R")

if (testing){
  initDat <- readRDS(here::here("data", "bg.rds"))
  budgets <- initDat[[1]]
  gains <- initDat[[2]]
} else{
  batch <- makeGrid(bsd)
  newBatchOut <- runBatch(batch, bsd)
  budgets <- newBatchOut %>% dplyr::select(contains("perc")) %>% as.matrix
  gains <- newBatchOut %>% dplyr::pull(response)
  saveRDS(list(budgets, gains), file=here::here("data", "bg.rds"))
}
budgets <- as.matrix(budgets)
budgets <- budgets[,-4]

