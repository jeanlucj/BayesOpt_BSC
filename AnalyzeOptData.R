
here::i_am("AnalyzeOptData.R")

btOut <- list(train_x, train_obj, opt_traces)
saveRDS(btOut, file=here::here("data", "BoTorchOut.rds"))

btOut <- readRDS(file=here::here("data", "BoTorchOut.rds"))
train_x <- btOut[[1]]
train_obj <- btOut[[2]]
opt_traces <- btOut[[3]]

plot(train_obj)
plot(unlist(opt_traces[[1]]))
plot(train_x[,1])
