# load packages
library(dplyr)
library(readr)

setwd("/Users/vsehtman/PycharmProjects/AE-RL-Pure-Python/data/datasets/boruta")
balanced_sample_results <- read_csv("./strat-vs-balanced-new/balanced/02_boruta_results_after_tentative_decision.csv")
colnames(balanced_sample_results)[1] <- "Feature"

stratified_sample_results <- read_csv("./rand-vs-strat-new/strat_10_percent/02_boruta_results_after_tentative_decision.csv")
colnames(stratified_sample_results)[1] <- "Feature"

# Split features by decision
balanced_confirmed <- balanced_sample_results %>%
  filter(decision == "Confirmed") %>%
  pull(Feature)  # "Feature" is the name of the feature-column

balanced_rejected <- balanced_sample_results %>%
  filter(decision == "Rejected") %>%
  pull(Feature)

stratified_confirmed <- stratified_sample_results %>%
  filter(decision == "Confirmed") %>%
  pull(Feature)

stratified_rejected <- stratified_sample_results %>%
  filter(decision == "Rejected") %>%
  pull(Feature)

confirmed_both <- intersect(balanced_confirmed, stratified_confirmed)
confirmed_only_balanced <- setdiff(balanced_confirmed, stratified_confirmed)
confirmed_only_stratified <- setdiff(stratified_confirmed, balanced_confirmed)

rejected_both <- intersect(balanced_rejected, stratified_rejected)
rejected_only_balanced <- setdiff(balanced_rejected, stratified_rejected)
rejected_only_stratified <- setdiff(stratified_rejected, balanced_rejected)

print("Common confirmed features:")
print(confirmed_both)

print("Only in balanced test confirmed features:")
print(confirmed_only_balanced)

print("Only in stratified test confirmed features:")
print(confirmed_only_stratified)

print("Common rejected features:")
print(rejected_both)

print("Only in balanced test rejected features:")
print(rejected_only_balanced)

print("Only in stratified test rejected features:")
print(rejected_only_stratified)

# save results
write.csv(data.frame(confirmed_both), "confirmed_both.csv", row.names = FALSE)
write.csv(data.frame(confirmed_only_balanced), "confirmed_only_balanced.csv", row.names = FALSE)
write.csv(data.frame(confirmed_only_stratified), "confirmed_only_stratified.csv", row.names = FALSE)

write.csv(data.frame(rejected_both), "rejected_both.csv", row.names = FALSE)
write.csv(data.frame(rejected_only_balanced), "rejected_only_balanced.csv", row.names = FALSE)
write.csv(data.frame(rejected_only_stratified), "rejected_only_stratified.csv", row.names = FALSE)
