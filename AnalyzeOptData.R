
here::i_am("AnalyzeOptData.R")

btOut <- list(train_x, train_obj, traces)
saveRDS(btOut, file=here::here("output", "BoTorchOut.rds"))

btOut <- readRDS(file=here::here("output", "BoTorchOut.rds"))
train_x <- btOut[[1]]
train_obj <- btOut[[2]]
traces <- btOut[[3]]

# Figure out how to combobulate all this