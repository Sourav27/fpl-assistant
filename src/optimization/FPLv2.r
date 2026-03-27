############################ DATA READING AND CLEANING

# load libraries as required
library(ggplot2)
library(tidyverse)
library(lpSolve)
library(caret)

# fix the working directory
setwd("C:\\Users\\debna\\OneDrive - Indian Institute of Management\\Term 4\\CCS\\")

# read the match-level data from T20I matches
df <- read.csv("datasets\\gw1.csv")

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


# Linear ------------------------------------------------------------------



# Define the objective function: maximize total expected points

# Define the constraint matrix
constraint_matrix <- matrix(0, nrow = 32, ncol = nrow(df1))
rownames(constraint_matrix) <- c("SelectionPool","Playing11","Sel_GK", "Sel_DEF", "Sel_MID", "Sel_FWD","Team_gk","Team_def","Team_mid","Team_fwd","Southampton","Bournemouth","Chelsea","Newcastle","Leicester","Nott'm Forest","Crystal Palace","Wolves","Brentford","Spurs","West Ham","Liverpool","Leeds","Fulham","Brighton","Man City","Man Utd","Everton","Arsenal","Aston Villa","Price","Captain")
colnames(constraint_matrix) <- t(df1$element)
# Constraint: Maximum number of players per position
#SelectionPool
for (i in 1:ncol(constraint_matrix)) {
  constraint_matrix[1, i] <- 1
}

#Constraint: Playing 11
for (i in 1:ncol(constraint_matrix)) {
  constraint_matrix[2, i] <- 1
}

#Selection GK
for (i in 1:ncol(constraint_matrix)) {
  constraint_matrix[3, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix)[i])),46]
}

#Selection DEF
for (i in 1:ncol(constraint_matrix)) {
  constraint_matrix[4, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix)[i])),44]
}

#Selection MID
for (i in 1:ncol(constraint_matrix)) {
  constraint_matrix[5, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix)[i])),47]
}

#Selection FWD
for (i in 1:ncol(constraint_matrix)) {
  constraint_matrix[6, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix)[i])),45]
}

#Team gk
for (i in 1:ncol(constraint_matrix)) {
  constraint_matrix[7, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix)[i])),46]
}

#Team def
for (i in 1:ncol(constraint_matrix)) {
  constraint_matrix[8, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix)[i])),44]
}

#Team mid
for (i in 1:ncol(constraint_matrix)) {
  constraint_matrix[9, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix)[i])),47]
}

#Team fwd
for (i in 1:ncol(constraint_matrix)) {
  constraint_matrix[10, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix)[i])),45]
}

#Same team
for (i in 1:ncol(constraint_matrix)){
  for (j in 11:30){
    if(rownames(constraint_matrix)[j] == df1[which(df1$element == as.numeric(colnames(constraint_matrix)[i])),3]){
      constraint_matrix[j,i] <- 1
    }else{
      constraint_matrix[j,i] <- 0
    }
  }
}

#Value
for (i in 1:ncol(constraint_matrix)) {
  constraint_matrix[31, i] <- df1[which(df1$element == as.numeric(colnames(constraint_matrix)[i])),which(colnames(df1) == "value")]
}

#Captain
for (i in 1:ncol(constraint_matrix)) {
  constraint_matrix[32, i] <- 1
}


##Non-Linear
install.packages("nloptr")
library(nloptr)


# NonLinear ------------------------------------------------------------------

install.packages("nloptr")
library(nloptr)

x <- matrix(data = 0, nrow = 3, ncol = nrow(df1))
#x2 <- matrix(data = 0, nrow = 1, ncol = nrow(df1))
#x3 <- matrix(data = 0, nrow = 1, ncol = nrow(df1))

objective <- function(x) {
  return(-((x[1,]*x[2,]) %*% df1[,4] +(x[1,]*x[2,]*x[3,]) %*% df1[,4]))  # Objective function to maximize (-1 * (x * y))
}

# initial_values <- c(rep(1,nrow(df1))*3)
# ub <- c(rep(1,nrow(df1)*3))
# lb <- c(rep(0,nrow(df1)*3))
initial_values = matrix(data = 1, nrow = 3, ncol = nrow(df1)) #1 for players with maximum XP and zero for rest
ub = matrix(data = 1, nrow = 3, ncol = nrow(df1))
lb = matrix(data = 0, nrow = 3, ncol = nrow(df1))

#Constraints

equality_constraint <- function(x) {
  constraint1 <- sum(x[1,])-15  
  constraint2 <- sum(x[2,])-11
  constraint3 <- x[1,] %*% df1[,46]-2
  constraint4 <- x[1,] %*% df1[,44]-5
  constraint5 <- x[1,] %*% df1[,47]-5
  constraint6 <- x[1,] %*% df1[,45]-3
  constraint7 <- x[2,] %*% df1[,46]-1
  constraint8 <- sum(x[3,])-1
  return(c(constraint1,constraint2,constraint3,constraint4,constraint5,constraint6,constraint7,constraint8))
}

inequality_constraint <- function(x) {
  constraint9 <- x[1,] %*% df1[,38]-1000
  constraint10 <- x[2,] %*% df1[,44]-5
  constraint11 <- -(x[2,] %*% df1[,44]+3)
  constraint12 <- x[2,] %*% df1[,47]-5
  constraint13 <- -(x[2,] %*% df1[,47]+2)
  constraint14 <- x[2,] %*% df1[,45]-3
  constraint15 <- -(x[2,] %*% df1[,45]+1)
  return(c(constraint9,constraint10,constraint11,constraint12,constraint13,constraint14,constraint15))
}


#optim_result <- optim(initial_values, objective, method = "L-BFGS-B", control = list(fnscale = -1), constraints = constraint_prob)

is_integer <- rep(TRUE,nrow(df1)*3)

problem <- nloptr(x0 = initial_values,  # Initial values for x and y
                  eval_f = objective,  # Objective function
                  lb = lb,  # Lower bounds
                  ub = ub,  # Upper bounds
                  eval_g_eq = equality_constraint,  # Equality constraint function
                  eval_g_ineq = inequality_constraint,  # Inequality constraint function
                  opts = list("algorithm" = "NLOPT_GN_ISRES", "integer_data" = is_integer,"xtol_rel" = 1e-8, "maxeval" = 2000))


# Solve the optimization problem
result <- nloptr::nloptr(problem)


# Previous Code -----------------------------------------------------------


# Define the constraint type (≤)
constraint_types <- c(rep("==", 7),rep("<=",24),"==")

# Define the right-hand side of constraints
rhs <- c(max_total_players,Playing,Sel_GK,Sel_def,Sel_mid,Sel_fwd,min_GK,min_def,min_mid,min_fwd,rep(max_players_per_team,20),1000,1)

# Set up the linear programming problem
lp <- lp("max", objective_function, constraint_matrix, constraint_types, rhs)

# Solve the linear programming problem
solution <- lp["solution"]

# Extract the selected players
selected_players <- player_names[solution == 1]

# Print the optimal squad
cat("Optimal Squad (15 Players):\n")
cat(selected_players, "\n")

# Print the first 11 players
cat("\nFirst 11 Players:\n")
cat(selected_players[1:11], "\n")

# Print the captain and vice-captain
cat("\nCaptain: ", selected_players[1], "\n")
cat("Vice-Captain: ", selected_players[2], "\n")



#------------------



library(nloptr)

# Assuming df1 is your dataset with 500+ players

x <- matrix(data = 0, nrow = 3, ncol = nrow(df1))

objective <- function(x) {
  # -(sum(x[1, ] * x[2, ] * df1[, 4]) + sum(x[1, ] * x[2, ] * x[3, ] * df1[, 4]))
  a=-(sum(x[1, ]  * df1[, 4])) # 3 step optimization
  return(as.numeric(a))
}

initial_values <- matrix(data = 1, nrow = 3, ncol = nrow(df1))
ub <- matrix(data = 1, nrow = 3, ncol = nrow(df1))
lb <- matrix(data = 0, nrow = 3, ncol = nrow(df1))

# Constraints

equality_constraint <- function(x) {
  constraint1 <- sum(x[1, ]) - 15
  constraint2 <- sum(x[2, ]) - 11
  constraint3 <- sum(x[1, ] * df1[, 46]) - 2
  constraint4 <- sum(x[1, ] * df1[, 44]) - 5
  constraint5 <- sum(x[1, ] * df1[, 47]) - 5
  constraint6 <- sum(x[1, ] * df1[, 45]) - 3
  constraint7 <- sum(x[2, ] * df1[, 46]) - 1
  constraint8 <- sum(x[3, ]) - 1
  return(c(constraint1, constraint2, constraint3, constraint4, constraint5, constraint6, constraint7, constraint8))
}

inequality_constraint <- function(x) {
  constraint9 <- sum(x[1, ] * df1[, 38]) - 1000
  constraint10 <- sum(x[2, ] * df1[, 44]) - 5
  constraint11 <- -(sum(x[2, ] * df1[, 44]) + 3)
  constraint12 <- sum(x[2, ] * df1[, 47]) - 5
  constraint13 <- -(sum(x[2, ] * df1[, 47]) + 2)
  constraint14 <- sum(x[2, ] * df1[, 45]) - 3
  constraint15 <- -(sum(x[2, ] * df1[, 45]) + 1)
  return(c(constraint9, constraint10, constraint11, constraint12, constraint13, constraint14, constraint15))
}

is_integer <- matrix(data = TRUE, nrow = 3, ncol = nrow(df1))

problem <- nloptr(x0 = initial_values,
                  eval_f = objective,
                  lb = lb,
                  ub = ub,
                  eval_g_eq = equality_constraint,
                  eval_g_ineq = inequality_constraint,
                  opts = list("algorithm" = "NLOPT_GN_ISRES",
                              "integer_data" = is_integer,
                              "xtol_rel" = 1e-8,
                              "maxeval" = 2000))

# Solve the optimization problem
result <- nloptr::nloptr(problem)

# Extract the selected players
selected_players <- df1[result$solution[,] == 1, "player_name"]

# Print the optimal squad
cat("Optimal Squad (15 Players):\n")
cat(selected_players, "\n")

# Print the first 11 players
cat("\nFirst 11 Players:\n")
cat(selected_players[1:11], "\n")

# Print the captain and vice-captain
cat("\nCaptain: ", selected_players[1], "\n")
cat("Vice-Captain: ", selected_players[2], "\n")