# load packages
library(dplyr)
library(readr)

setwd("/Users/vsehtman/PycharmProjects/AE-RL-Pure-Python/data/datasets/boruta")
balanced_stratified_sample_results <- read_csv("./stratified_balanced/boruta_results_balanced_strat.csv")
colnames(balanced_stratified_sample_results)[1] <- "Feature"

stratified_sample_results <- read_csv("./rand_vs_stratified_10_percent/stratified_no_fix_seed/boruta_results.csv")
colnames(stratified_sample_results)[1] <- "Feature"

# Split features by decision
balanced_strat_confirmed <- balanced_stratified_sample_results %>%
  filter(decision == "Confirmed") %>%
  pull(Feature)  # "Feature" is the name of the feature-column

balanced_strat_rejected <- balanced_stratified_sample_results %>%
  filter(decision == "Rejected") %>%
  pull(Feature)

stratified_confirmed <- stratified_sample_results %>%
  filter(decision == "Confirmed") %>%
  pull(Feature)

stratified_rejected <- stratified_sample_results %>%
  filter(decision == "Rejected") %>%
  pull(Feature)

confirmed_both <- intersect(balanced_strat_confirmed, stratified_confirmed)
confirmed_only_balanced_strat <- setdiff(balanced_strat_confirmed, stratified_confirmed)
confirmed_only_stratified <- setdiff(stratified_confirmed, balanced_strat_confirmed)

rejected_both <- intersect(balanced_strat_rejected, stratified_rejected)
rejected_only_balanced_strat <- setdiff(balanced_strat_rejected, stratified_rejected)
rejected_only_stratified <- setdiff(stratified_rejected, balanced_strat_rejected)

print("Common confirmed features:")
print(confirmed_both)

print("Only in balanced stratified test confirmed features:")
print(confirmed_only_balanced_strat)

print("Only in stratified test confirmed features:")
print(confirmed_only_stratified)

print("Common rejected features:")
print(rejected_both)

print("Only in balanced stratified test rejected features:")
print(rejected_only_balanced_strat)

print("Only in stratified test rejected features:")
print(rejected_only_stratified)

# save results
write.csv(data.frame(confirmed_both), "confirmed_both.csv", row.names = FALSE)
write.csv(data.frame(confirmed_only_balanced_strat), "confirmed_only_balanced_strat.csv", row.names = FALSE)
write.csv(data.frame(confirmed_only_stratified), "confirmed_only_stratified.csv", row.names = FALSE)

write.csv(data.frame(rejected_both), "rejected_both.csv", row.names = FALSE)
write.csv(data.frame(rejected_only_balanced_strat), "rejected_only_balanced_strat.csv", row.names = FALSE)
write.csv(data.frame(rejected_only_stratified), "rejected_only_stratified.csv", row.names = FALSE)
