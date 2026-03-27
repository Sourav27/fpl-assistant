############################ Historical expected minutes per player and position

############################ DATA READING AND CLEANING

# load libraries as required
library(ggplot2)
library(tidyverse)
library(lpSolve)
# install.packages("caret")
library(caret)

# fix the working directory
setwd("C:\\Users\\debna\\OneDrive - Indian Institute of Management\\Term 4\\CCS\\")

# read the merged gameweek data for 2021-22 season
merged_df <- read.csv("datasets\\merged_gw2122.csv")
merged_df_latest <- read.csv("datasets\\merged_gw2223.csv")
player_ids <- read.csv("datasets\\player_ids.csv")
merged_df$position[merged_df$position=="GKP"]="GK"
gameweek_summary <- merged_df %>%
  group_by(round, element, name, position) %>%
  summarize(
    total_minutes = sum(minutes),
    total_matches = n()
  )
avg_points_per_min = sum(merged_df$total_points)/sum(merged_df$minutes)
avg_xP_per_min = sum(merged_df$xP)/sum(merged_df$minutes)
position_summary <- merged_df %>%
  group_by(position) %>%
  summarise(
    total_minutes = sum(minutes),
    total_matches = n(),
    avgMin = total_minutes/total_matches,
    total_points = sum(total_points),
    total_xP = sum(xP),
    points_per_min = total_points/total_minutes,
    xP_per_min = total_xP/total_minutes,
    weight = points_per_min/avg_points_per_min,
    weight_xP = xP_per_min/avg_xP_per_min
  )

position_summary1 <- merged_df_latest %>%
  group_by(position) %>%
  summarise(
    total_minutes = sum(minutes),
    total_matches = n(),
    avgMin = total_minutes/total_matches,
    total_points = sum(total_points),
    points_per_min = total_points/total_minutes
  )

min_summary <- gameweek_summary %>%
  group_by(element, name, position) %>%
  summarize(
    minutes_played = sum(total_minutes),
    matches_played = sum(total_matches),
    xMin = minutes_played/matches_played
  ) %>%
  mutate(
    weight = position_summary$weight[position == position_summary$position],
    avgMin = position_summary$avgMin[position == position_summary$position]
  )

gameweek_summary1 <- merged_df_latest %>%
  group_by(round, element, name, position) %>%
  summarize(
    total_minutes = sum(minutes),
    total_matches = n()
  )

min_summary1 <- gameweek_summary1 %>%
  group_by(element, name, position) %>%
  summarize(
    minutes_played = sum(total_minutes),
    matches_played = sum(total_matches),
    xMin = minutes_played/matches_played
  )

df2 <- merge(x=player_ids,y=min_summary1, 
             by.x=c("id"),by.y=c("element"), all.y=TRUE)
df2 <- df2 %>% dplyr::rename('name' = 'name.y','avgMin2223' = 'xMin')
df2 <- subset(df2, select=-c(name.x,minutes_played,matches_played))
df2$xMin <- min_summary$xMin[match(df2$match_id, min_summary$element)]
df2$pos_weight <- min_summary$weight[match(df2$match_id, min_summary$element)]
df2$avgMin <- min_summary$avgMin[match(df2$match_id, min_summary$element)]
df2 <- subset(df2, select=-c(match_id))
write.csv(df2, "datasets\\xMin.csv")
