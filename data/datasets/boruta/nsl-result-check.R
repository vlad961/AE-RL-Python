# install the following packets if not present yet.
#install.packages("Boruta")
#install.packages("readr")
#install.packages("dplyr")
#install.packages("caret")
# load packets
library(Boruta)
library(readr)
library(dplyr)

setwd("/Users/vsehtman/PycharmProjects/AE-RL-Pure-Python/data/datasets/boruta")

boruta_df <- read.csv("rand-vs-strat-new/rand_10_percent/01_boruta_results_before_tentative_decision.csv", row.names = 1)

confirmed_features <- rownames(boruta_df[boruta_df$decision == "Confirmed", ])
tentative_features <- rownames(boruta_df[boruta_df$decision == "Tentative", ])
rejected_features <- rownames(boruta_df[boruta_df$decision == "Rejected", ])