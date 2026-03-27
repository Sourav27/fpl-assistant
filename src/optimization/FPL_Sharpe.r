########################### FPL Team Optimization using Sharpe Ratio optimization

############################ DATA READING AND CLEANING

# load libraries as required
library(ggplot2)
library(tidyverse)
library(lpSolve)
#install.packages("caret")
library(caret)

setwd("C:\\Users\\debna\\OneDrive - Indian Institute of Management\\Term 4\\CCS\\")
# read Gameweek 1 data for Season 2022-23
df <- read.csv("datasets\\gws\\gw1.csv")

df1 <- df
df1$SelectionPool <- sample(c(0, 1), size = nrow(df1), replace = TRUE)
df1$Playing11 <- sample(c(0, 1), size = nrow(df1), replace = TRUE)
df1$IsCaptain <- sample(c(0, 1), size = nrow(df1), replace = TRUE)

columns_to_encode <- c("position", "team")

# Perform dummy encoding using dummyVars()
encoded_data <- model.matrix(~.-1, data = df1[, columns_to_encode])
df1 <- cbind(df1, encoded_data)


# Define constants and constraints
Playing <- 11
num_positions <- 4
max_players_per_team <- 3
max_total_players <- 15
Sel_GK<- 2
Sel_def <- 5
min_def <- 3
Sel_mid <- 5
min_mid <- 2
Sel_fwd <- 3
min_fwd <- 1
min_GK <- 1

#create covariance matrix using total points data from Season 2021-22
dftp <- read.csv("datasets\\TotalPoints.csv")
dftp[is.na(dftp)] <- 0
for (i in 1:ncol(dftp)){
  colnames(dftp)[i] <- substring(colnames(dftp)[i],2)
}

S <- cov(dftp)

# Linear ------------------------------------------------------------------



# Define the objective function: maximize total expected points

# Define the constraint matrix
constraint_matrix <- matrix(0, nrow = 26, ncol = nrow(df1))
rownames(constraint_matrix) <- c("SelectionPool","Sel_GK", "Sel_DEF", "Sel_MID", "Sel_FWD","Southampton","Bournemouth","Chelsea","Newcastle","Leicester","Nott'm Forest","Crystal Palace","Wolves","Brentford","Spurs","West Ham","Liverpool","Leeds","Fulham","Brighton","Man City","Man Utd","Everton","Arsenal","Aston Villa","Price")
#"SelectionPool","Playing11","Sel_GK", "Sel_DEF", "Sel_MID", "Sel_FWD","Team_gk","Team_def","Team_mid","Team_fwd","Southampton","Bournemouth","Chelsea","Newcastle","Leicester","Nott'm Forest","Crystal Palace","Wolves","Brentford","Spurs","West Ham","Liverpool","Leeds","Fulham","Brighton","Man City","Man Utd","Everton","Arsenal","Aston Villa","Price","Captain"
colnames(constraint_matrix) <- t(df1$element)


# Constraint: Maximum number of players per position
#SelectionPool
for (i in 1:ncol(constraint_matrix)) {
  constraint_matrix[1, i] <- 1
}

#Constraint: Playing 11
#for (i in 1:ncol(constraint_matrix)) {
  #constraint_matrix[2, i] <- 1
#}

#Selection GK
for (i in 1:ncol(constraint_matrix)) {
  constraint_matrix[2, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix)[i])),46]
}

#Selection DEF
for (i in 1:ncol(constraint_matrix)) {
  constraint_matrix[3, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix)[i])),44]
}

#Selection MID
for (i in 1:ncol(constraint_matrix)) {
  constraint_matrix[4, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix)[i])),47]
}

#Selection FWD
for (i in 1:ncol(constraint_matrix)) {
  constraint_matrix[5, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix)[i])),45]
}

#Same team
for (i in 1:ncol(constraint_matrix)){
  for (j in 6:25){
    if(rownames(constraint_matrix)[j] == df1[which(df1$element == as.numeric(colnames(constraint_matrix)[i])),3]){
      constraint_matrix[j,i] <- 1
    }else{
      constraint_matrix[j,i] <- 0
    }
  }
}

#Value
for (i in 1:ncol(constraint_matrix)) {
  constraint_matrix[26, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix)[i])),which(colnames(df1) == "value")]
}

# #Captain
# for (i in 1:ncol(constraint_matrix)) {
#   constraint_matrix[32, i] <- 1
# }

obj <- t(df1$xP)

dir <- c(rep("==",5),rep("<=",20),"<=")

rhs <- c(15,2,5,5,3,rep(3,20),1000)

lp <- lp("max", obj, constraint_matrix, dir, rhs)

lp_model <- lp(direction = "max",
               objective.in = obj,  # Objective function coefficients
               const.mat = constraint_matrix,  # Constraint coefficients
               const.dir = dir,  # Constraint directions
               const.rhs = rhs,  # Constraint right-hand sides
               all.bin = TRUE)  # Specify variables as binary

obj_value <- lp_model$objval

#extracting player information in 15 Member selection Squad
Rnum <- matrix(0,nrow=1,ncol=15)

SelTeam <- matrix(0, nrow = 15, ncol = ncol(df))
colnames(SelTeam) <- colnames(df)

j <- 1
for(i in 1:length(lp_model$solution)){
  if(lp_model$solution[i]==1){
    Rnum[1,j] <- i
    j <- j+1
  }
}

for(a in 1:15){
  for(b in 1:ncol(SelTeam)){
    SelTeam[a,b] <- df[Rnum[1,a],b]
  }
}

#11 member team
constraint_matrix1 <- matrix(0, nrow = 8, ncol = nrow(SelTeam))
rownames(constraint_matrix1) <- c("Playing11","Team_gk","Team_deflow","Team_defhigh","Team_midlow","Team_midhigh","Team_fwdlow","Team_fwdhigh")
#"SelectionPool","Playing11","Sel_GK", "Sel_DEF", "Sel_MID", "Sel_FWD","Team_gk","Team_def","Team_mid","Team_fwd","Southampton","Bournemouth","Chelsea","Newcastle","Leicester","Nott'm Forest","Crystal Palace","Wolves","Brentford","Spurs","West Ham","Liverpool","Leeds","Fulham","Brighton","Man City","Man Utd","Everton","Arsenal","Aston Villa","Price","Captain"
colnames(constraint_matrix1) <- t(SelTeam[,10])

#Constraint: Playing 11
for (i in 1:ncol(constraint_matrix1)) {
constraint_matrix1[1, i] <- 1
}

#Team gk
for (i in 1:ncol(constraint_matrix1)) {
constraint_matrix1[2, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix1)[i])),46]
}

#Team deflow
for (i in 1:ncol(constraint_matrix1)) {
constraint_matrix1[3, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix1)[i])),44]
}

#Team defhigh
for (i in 1:ncol(constraint_matrix1)) {
  constraint_matrix1[4, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix1)[i])),44]
}

#Team midlow
for (i in 1:ncol(constraint_matrix1)) {
constraint_matrix1[5, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix1)[i])),47]
}

#Team midhigh
for (i in 1:ncol(constraint_matrix1)) {
  constraint_matrix1[6, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix1)[i])),47]
}

#Team fwdlow
for (i in 1:ncol(constraint_matrix1)) {
constraint_matrix1[7, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix1)[i])),45]
}

#Team fwdhigh
for (i in 1:ncol(constraint_matrix1)) {
  constraint_matrix1[8, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix1)[i])),45]
}

obj1 <- as.numeric(t(SelTeam[,4]))

dir1 <- c("==","==",">=","<=",">=","<=",">=","<=")

rhs1 <- c(11,1,3,5,2,5,1,3)


lp_model1 <- lp(direction = "max",
               objective.in = obj1,  # Objective function coefficients
               const.mat = constraint_matrix1,  # Constraint coefficients
               const.dir = dir1,  # Constraint directions
               const.rhs = rhs1,  # Constraint right-hand sides
               all.bin = TRUE)  # Specify variables as binary

obj_value1 <- lp_model1$objval

#extracting player information in 11 Member team
Rnum1 <- matrix(0,nrow=1,ncol=11)

Team <- matrix(0, nrow = 11, ncol = ncol(df))
colnames(Team) <- colnames(df)

j <- 1
for(i in 1:length(lp_model1$solution)){
  if(lp_model1$solution[i]==1){
    Rnum1[1,j] <- i
    j <- j+1
  }
}

for(a in 1:11){
  for(b in 1:ncol(Team)){
    Team[a,b] <- SelTeam[Rnum1[1,a],b]
  }
}

#CAPTAIN
Captain <- Team[which.max(as.numeric(Team[,4])),1]
#ViceCaptain <- Team[,1]

TotalxP <- sum(as.numeric(Team[,4]))+as.numeric(Team[which.max(Team[,4]),4])

##Calculating Sharpe Ratio

TeamxPVar <- 0

for (i in 1:11){
  for (j in 1:11){
    TeamxPVar <- TeamxPVar + S[as.numeric(Team[i,10]),as.numeric(Team[j,10])]
  }
}

SRatio <- TotalxP/sqrt(TeamxPVar)

##Greedy Algo

df2 <- df[df$position=="MID" & df$xP>=2.3,]
df3 <- df[df$position=="FWD" & df$xP>=2,]
df4 <- df[df$position=="DEF" & df$xP>=3,]
df5 <- df[df$position=="GK" & df$xP>=3.6,]

dfnew <- rbind(df2,df3,df4,df5)

Possible15 <- data.frame(matrix(0, nrow = 500, ncol = 15))
def <- data.frame(matrix(0, nrow = 1, ncol = 5))
mid <- data.frame(matrix(0, nrow = 1, ncol = 5))
fwd <- data.frame(matrix(0, nrow = 1, ncol = 3))
gk <- data.frame(matrix(0, nrow = 1, ncol = 2))

for (i in 1:500){
  set.seed(i)
  def <- sample(dfnew$element[dfnew$position=="DEF"],5,replace = FALSE)
  mid <- sample(dfnew$element[dfnew$position=="MID"],5,replace = FALSE)
  fwd <- sample(dfnew$element[dfnew$position=="FWD"],3,replace = FALSE)
  gk <- sample(dfnew$element[dfnew$position=="GK"],2,replace = FALSE)
  for (j in 1:5){
    Possible15[i,j] <- def[j]
  }
  for (j in 6:10){
    Possible15[i,j] <- mid[j-5]
  }
  for (j in 11:13){
    Possible15[i,j] <- fwd[j-10]
  }
  for (j in 14:15){
    Possible15[i,j] <- gk[j-13]
  }
}
Possible15$Price <- c(rep(0,500))
Possible15$Team <- c(rep(1,500))

for (i in 1:500){
  for (j in 1:15){
    Possible15[i,16] <- Possible15[i,16] + dfnew[which(dfnew$element == as.numeric(Possible15[i,j])),38]
  }
}

SameTeam <- data.frame(matrix(0, ncol = 15, nrow =length(unique(df$team))))
rownames(SameTeam) <- c("Southampton","Bournemouth","Chelsea","Newcastle","Leicester","Nott'm Forest","Crystal Palace","Wolves","Brentford","Spurs","West Ham","Liverpool","Leeds","Fulham","Brighton","Man City","Man Utd","Everton","Arsenal","Aston Villa")
SameTeam$Total <- rowSums(SameTeam)

#Same team

for (i in 1:500){
  SameTeam <- data.frame(matrix(0, ncol = 15, nrow =length(unique(df$team))))
  rownames(SameTeam) <- c("Southampton","Bournemouth","Chelsea","Newcastle","Leicester","Nott'm Forest","Crystal Palace","Wolves","Brentford","Spurs","West Ham","Liverpool","Leeds","Fulham","Brighton","Man City","Man Utd","Everton","Arsenal","Aston Villa")
  SameTeam$Total <- rowSums(SameTeam)
  for (j in 1:15){
    for (k in 1:nrow(SameTeam)){
      if(rownames(SameTeam)[k] == dfnew[which(dfnew$element == as.numeric(Possible15[i,j])),3]){
        SameTeam[k,j] <- SameTeam[k,j] + 1
      }
    }
  }
  if(any(apply(SameTeam, 1, function(row) sum(row) > 3))==TRUE){
    Possible15[i,17] <- 0
  }
}


Feasible15 <- Possible15[Possible15$Price<=1000 & Possible15$Team==1,]
Feasible15 <- Feasible15[,-16]
Feasible15 <- Feasible15[,-16]

####Generate Team with 11 players

colnames(Feasible15) <- c("D1","D2","D3","D4","D5","M1","M2","M3","M4","M5","F1","F2","F3","G1","G2")

# Install and load the necessary package
install.packages("gtools")
library(gtools)
# Function to combine two sets of combinations
combine_combinations <- function(comb1, comb2) {
  combined_combinations <- c()
  for (i in 1:nrow(comb1)) {
    for (j in 1:nrow(comb2)) {
      combined_combinations <- rbind(combined_combinations, c(comb1[i, ], comb2[j, ]))
    }
  }
  return(combined_combinations)
}



Final11 <- c()

# 1-4-4-2 Formation

for(i in 1:nrow(Feasible15)){
  def_comb <- combinations(5,4,v=t(Feasible15[i,1:5]))
  mid_comb <- combinations(5,4,v=t(Feasible15[i,6:10]))
  fwd_comb <- combinations(3,2,v=t(Feasible15[i,11:13]))
  gk_comb <- combinations(2,1,v=t(Feasible15[i,14:15]))
  Combs <- combine_combinations(combine_combinations(def_comb,mid_comb),combine_combinations(fwd_comb,gk_comb))
  Final11 <- rbind(Final11,Combs)
}

# 1-4-5-1 Formation

for(i in 1:nrow(Feasible15)){
  def_comb <- combinations(5,4,v=t(Feasible15[i,1:5]))
  mid_comb <- combinations(5,5,v=t(Feasible15[i,6:10]))
  fwd_comb <- combinations(3,1,v=t(Feasible15[i,11:13]))
  gk_comb <- combinations(2,1,v=t(Feasible15[i,14:15]))
  Combs <- combine_combinations(combine_combinations(def_comb,mid_comb),combine_combinations(fwd_comb,gk_comb))
  Final11 <- rbind(Final11,Combs)
}

# 1-4-3-3 Formation

for(i in 1:nrow(Feasible15)){
  def_comb <- combinations(5,4,v=t(Feasible15[i,1:5]))
  mid_comb <- combinations(5,3,v=t(Feasible15[i,6:10]))
  fwd_comb <- combinations(3,3,v=t(Feasible15[i,11:13]))
  gk_comb <- combinations(2,1,v=t(Feasible15[i,14:15]))
  Combs <- combine_combinations(combine_combinations(def_comb,mid_comb),combine_combinations(fwd_comb,gk_comb))
  Final11 <- rbind(Final11,Combs)
}

# 1-5-4-1 Formation

for(i in 1:nrow(Feasible15)){
  def_comb <- combinations(5,5,v=t(Feasible15[i,1:5]))
  mid_comb <- combinations(5,4,v=t(Feasible15[i,6:10]))
  fwd_comb <- combinations(3,1,v=t(Feasible15[i,11:13]))
  gk_comb <- combinations(2,1,v=t(Feasible15[i,14:15]))
  Combs <- combine_combinations(combine_combinations(def_comb,mid_comb),combine_combinations(fwd_comb,gk_comb))
  Final11 <- rbind(Final11,Combs)
}

# 1-5-3-2 Formation

for(i in 1:nrow(Feasible15)){
  def_comb <- combinations(5,5,v=t(Feasible15[i,1:5]))
  mid_comb <- combinations(5,3,v=t(Feasible15[i,6:10]))
  fwd_comb <- combinations(3,2,v=t(Feasible15[i,11:13]))
  gk_comb <- combinations(2,1,v=t(Feasible15[i,14:15]))
  Combs <- combine_combinations(combine_combinations(def_comb,mid_comb),combine_combinations(fwd_comb,gk_comb))
  Final11 <- rbind(Final11,Combs)
}

# 1-5-2-3 Formation

for(i in 1:nrow(Feasible15)){
  def_comb <- combinations(5,5,v=t(Feasible15[i,1:5]))
  mid_comb <- combinations(5,2,v=t(Feasible15[i,6:10]))
  fwd_comb <- combinations(3,3,v=t(Feasible15[i,11:13]))
  gk_comb <- combinations(2,1,v=t(Feasible15[i,14:15]))
  Combs <- combine_combinations(combine_combinations(def_comb,mid_comb),combine_combinations(fwd_comb,gk_comb))
  Final11 <- rbind(Final11,Combs)
}

# 1-3-5-2 Formation

for(i in 1:nrow(Feasible15)){
  def_comb <- combinations(5,3,v=t(Feasible15[i,1:5]))
  mid_comb <- combinations(5,5,v=t(Feasible15[i,6:10]))
  fwd_comb <- combinations(3,2,v=t(Feasible15[i,11:13]))
  gk_comb <- combinations(2,1,v=t(Feasible15[i,14:15]))
  Combs <- combine_combinations(combine_combinations(def_comb,mid_comb),combine_combinations(fwd_comb,gk_comb))
  Final11 <- rbind(Final11,Combs)
}

# 1-3-4-3 Formation

for(i in 1:nrow(Feasible15)){
  def_comb <- combinations(5,3,v=t(Feasible15[i,1:5]))
  mid_comb <- combinations(5,4,v=t(Feasible15[i,6:10]))
  fwd_comb <- combinations(3,3,v=t(Feasible15[i,11:13]))
  gk_comb <- combinations(2,1,v=t(Feasible15[i,14:15]))
  Combs <- combine_combinations(combine_combinations(def_comb,mid_comb),combine_combinations(fwd_comb,gk_comb))
  Final11 <- rbind(Final11,Combs)
}

Final11 <- as.data.frame(Final11)
Final11$xP <- c(rep(0,nrow(Final11)))

for (i in 1:nrow(Final11)){
  for (j in 1:11){
    Final11[i,12] <- Final11[i,12]+dfnew[which(dfnew$element==as.numeric(Final11[i,j])),4]
  }
}

#Calculating SD of xP

Final11$xPSD <- c(rep(0,nrow(Final11)))

for (i in 1:nrow(Final11)){
  TeamxPV <- 0
  for (j in 1:11){
    for (k in 1:11) {
      TeamxPV <- TeamxPV + S[as.numeric(Final11[i,j]),as.numeric(Final11[i,k])]
    }
  }
  Final11[i,13] <- sqrt(TeamxPV)
}

Final11$SharpeRatio <- Final11$xP/Final11$xPSD

plot(Final11$xPSD,Final11$xP)

MaxS <- which.max(Final11$SharpeRatio)
V <- Final11[MaxS,1:11]

BestTeam <- df[df$element %in% V, ]
###Running the optimization keeping minimizing Variance as obj function



