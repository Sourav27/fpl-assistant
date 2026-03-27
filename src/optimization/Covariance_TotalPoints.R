# fix the working directory
setwd("C:\\Users\\debna\\OneDrive - Indian Institute of Management\\Term 4\\CCS\\")

dftp <- read.csv("datasets\\TotalPoints.csv")
dftp[is.na(dftp)] <- 0
for (i in 1:ncol(dftp)){
  colnames(dftp)[i] <- substring(colnames(dftp)[i],2)
}

S <- cov(dftp)
