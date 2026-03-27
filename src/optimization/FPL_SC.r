############################ DATA READING AND CLEANING

# load libraries as required
library(ggplot2)
library(tidyverse)
library(lpSolve)
# install.packages("caret")
library(caret)

# fix the working directory
setwd("C:\\Users\\debna\\OneDrive - Indian Institute of Management\\Term 4\\CCS\\")
df_merged <- read.csv("datasets\\merged_gw2223.csv")

result_summary <- data.frame(
  gw = numeric(length(unique(df_merged$round))),
  total_xP = numeric(length(unique(df_merged$round))),
  actual_points = numeric(length(unique(df_merged$round))),
  player1 = character(length(unique(df_merged$round))),
  player2 = character(length(unique(df_merged$round))),
  player3 = character(length(unique(df_merged$round))),
  player4 = character(length(unique(df_merged$round))),
  player5 = character(length(unique(df_merged$round))),
  player6 = character(length(unique(df_merged$round))),
  player7 = character(length(unique(df_merged$round))),
  player8 = character(length(unique(df_merged$round))),
  player9 = character(length(unique(df_merged$round))),
  player10 = character(length(unique(df_merged$round))),
  player11 = character(length(unique(df_merged$round))),
  mae = numeric(length(unique(df_merged$round))),
  total_players = numeric(length(unique(df_merged$round)))
)

for(k in 1:length(unique(df_merged$round))){
  
  
  # read the Gameweek "k" data
  df <- read.csv(paste0("datasets\\gws\\gw",unique(df_merged$round)[k],".csv"))
  # df_xMin <- read.csv("datasets\\xMin.csv")
  # 
  # df$xMin <- df_xMin$xMin[match(df$element,df_xMin$id)]
  # df$xMin[is.na(df$xMin)] = 30 #replaced na values with 30min for all new players
  # df$xPMin <- df$xP * df$xMin/90
  
  
  df1 <- df
  
  for(z in df1$element[duplicated(df1$element)]){
    new_row = c(name = df1$name[df1$element==z][1], position=df1$position[df1$element==z][1], team=df1$team[df1$element==z][1], xP=df1$xP[df1$element==z][1],assists=sum(as.numeric(df1$assists[df1$element==z])),bonus=sum(as.numeric(df1$bonus[df1$element==z])), bps=sum(as.numeric(df1$bps[df1$element==z])), clean_sheets=sum(as.numeric(df1$clean_sheets[df1$element==z])),creativity=sum(as.numeric(df1$creativity[df1$element==z])), 
                element=z, expected_assists=sum(as.numeric(df1$expected_assists[df1$element==z])),expected_goal_involvements=sum(as.numeric(df1$expected_goal_involvements[df1$element==z])),expected_goals=sum(as.numeric(df1$expected_goals[df1$element==z])),expected_goals_conceded=sum(as.numeric(df1$expected_goals_conceded[df1$element==z])),fixture=paste(df1$fixture[df1$element==z][1],df1$fixture[df1$element==z][2],sep=","),
                goals_conceded=sum(as.numeric(df1$goals_conceded[df1$element==z])),goals_scored=sum(as.numeric(df1$goals_scored[df1$element==z])),ict_index=sum(as.numeric(df1$ict_index[df1$element==z])),influence=sum(as.numeric(df1$influence[df1$element==z])),kickoff_time=paste(df1$kickoff_time[df1$element==z][1],df1$kickoff_time[df1$element==z][2],sep=","),minutes=sum(as.numeric(df1$minutes[df1$element==z])),
                opponent_team=paste(df1$opponent_team[df1$element==z][1],df1$opponent_team[df1$element==z][2],sep=","),own_goals=sum(as.numeric(df1$own_goals[df1$element==z])),penalties_missed=sum(as.numeric(df1$penalties_missed[df1$element==z])),penalties_saved=sum(as.numeric(df1$penalties_saved[df1$element==z])),red_cards=sum(as.numeric(df1$red_cards[df1$element==z])),round=df1$round[df1$element==z][1],
                saves=sum(as.numeric(df1$saves[df1$element==z])),selected=df1$selected[df1$element==z][1],starts=sum(as.numeric(df1$starts[df1$element==z])),team_a_score=paste(df1$team_a_score[df1$element==z][1],df1$team_a_score[df1$element==z][2],sep=","), team_h_score=paste(df1$team_h_score[df1$element==z][1],df1$team_h_score[df1$element==z][2],sep=","),threat=sum(as.numeric(df1$threat[df1$element==z])),
                total_points=sum(as.numeric(df1$total_points[df1$element==z])),transfers_balance=df1$transfers_balance[df1$element==z][1],transfers_in=df1$transfers_in[df1$element==z][1],transfers_out=df1$transfers_out[df1$element==z][1],value=df1$value[df1$element==z][1],was_home=paste(df1$was_home[df1$element==z][1],df1$was_home[df1$element==z][2],sep=","),yellow_cards=sum(as.numeric(df1$yellow_cards[df1$element==z])))
    df1 = rbind(df1,new_row)
    df1 <- df1[!(row.names(df1) %in% row.names(df1[df1$element==z,])),]
  }
  
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
  # #for (i in 1:ncol(constraint_matrix)) {
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
  
  SelTeam <- matrix(0, nrow = 15, ncol = ncol(df1))
  colnames(SelTeam) <- colnames(df1)
  
  j <- 1
  for(i in 1:length(lp_model$solution)){
    if(lp_model$solution[i]==1){
      Rnum[1,j] <- i
      j <- j+1
    }
  }
  
  for(a in 1:15){
    for(b in 1:ncol(SelTeam)){
      SelTeam[a,b] <- df1[Rnum[1,a],b]
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
  
  Team <- matrix(0, nrow = 11, ncol = ncol(df1))
  colnames(Team) <- colnames(df1)
  
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
  ViceCaptain <- Team[order(Team[,4],decreasing=TRUE),1][2]
  
  TotalxP <- sum(as.numeric(Team[,4]))+as.numeric(Team[which.max(Team[,4]),4])
  ActualP <- sum(as.numeric(Team[,34]))+as.numeric(Team[which.max(Team[,4]),34])

  result_summary$gw[k] = unique(df_merged$round)[k]
  result_summary$total_xP[k] = TotalxP
  result_summary$actual_points[k] = ActualP
  result_summary$player1[k] = Team[order(as.numeric(Team[,4]),decreasing=TRUE),1][1]
  result_summary$player2[k] = Team[order(as.numeric(Team[,4]),decreasing=TRUE),1][2]
  result_summary$player3[k] = Team[order(as.numeric(Team[,4]),decreasing=TRUE),1][3]
  result_summary$player4[k] = Team[order(as.numeric(Team[,4]),decreasing=TRUE),1][4]
  result_summary$player5[k] = Team[order(as.numeric(Team[,4]),decreasing=TRUE),1][5]
  result_summary$player6[k] = Team[order(as.numeric(Team[,4]),decreasing=TRUE),1][6]
  result_summary$player7[k] = Team[order(as.numeric(Team[,4]),decreasing=TRUE),1][7]
  result_summary$player8[k] = Team[order(as.numeric(Team[,4]),decreasing=TRUE),1][8]
  result_summary$player9[k] = Team[order(as.numeric(Team[,4]),decreasing=TRUE),1][9]
  result_summary$player10[k] = Team[order(as.numeric(Team[,4]),decreasing=TRUE),1][10]
  result_summary$player11[k] = Team[order(as.numeric(Team[,4]),decreasing=TRUE),1][11]
  mae = df1 %>% summarise(
    ae = sum(abs(as.numeric(xP) - as.numeric(total_points))),
    n = n(),
    mae = ae/n
  )
  
  result_summary$mae[k] = mae$mae
  result_summary$total_players[k] = mae$n
}

write.csv(result_summary,"results_summary_xP.csv")