
here::i_am("AnalyzeOptData.R")

btOut <- list(train_x, train_obj, traces)
saveRDS(btOut, file=here::here("output", "BoTorchOut.rds"))

if (FALSE){
btOut <- readRDS(file=here::here("output", "BoTorchOut.rds"))
train_x <- btOut[[1]]
train_obj <- btOut[[2]]
traces <- btOut[[3]]

# Figure out how to recombobulate all this
picBud <- NULL
for (i in 1:length(train_x)){
  tst <- train_x[[i]][[1]]
  picBud <- cbind(picBud, tst[,1])
}
picBud <- cbind(1:536, picBud)
plot(picBud[,1:2], pch=16)
}