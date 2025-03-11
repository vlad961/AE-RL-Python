# load packages
library(dplyr)
library(readr)

setwd("/Users/vsehtman/PycharmProjects/AE-RL-Pure-Python/data/datasets/boruta")
random_sample_results <- read_csv("./rand-vs-strat-new/rand_10_percent/02_boruta_results_after_tentative_decision.csv")
colnames(random_sample_results)[1] <- "Feature"

stratified_sample_results <- read_csv("./rand-vs-strat-new/strat_10_percent/02_boruta_results_after_tentative_decision.csv")
colnames(stratified_sample_results)[1] <- "Feature"

# Split features by decision
random_confirmed <- random_sample_results %>%
  filter(decision == "Confirmed") %>%
  pull(Feature)  # ""Feature" is the name of the feature-column

random_rejected <- random_sample_results %>%
  filter(decision == "Rejected") %>%
  pull(Feature)

stratified_confirmed <- stratified_sample_results %>%
  filter(decision == "Confirmed") %>%
  pull(Feature)

stratified_rejected <- stratified_sample_results %>%
  filter(decision == "Rejected") %>%
  pull(Feature)

confirmed_both <- intersect(random_confirmed, stratified_confirmed)
confirmed_only_random <- setdiff(random_confirmed, stratified_confirmed)
confirmed_only_stratified <- setdiff(stratified_confirmed, random_confirmed)

rejected_both <- intersect(random_rejected, stratified_rejected)
rejected_only_random <- setdiff(random_rejected, stratified_rejected)
rejected_only_stratified <- setdiff(stratified_rejected, random_rejected)

print("Common confirmed features:")
print(confirmed_both)

print("Only in random sample confirmed features:")
print(confirmed_only_random)

print("Only in stratified sample confirmed features:")
print(confirmed_only_stratified)

print("Common rejected features:")
print(rejected_both)

print("Only in random sample rejected features:")
print(rejected_only_random)

print("Only in stratified sample rejected features:")
print(rejected_only_stratified)

# save results
write.csv(data.frame(confirmed_both), "confirmed_both.csv", row.names = FALSE)
write.csv(data.frame(confirmed_only_random), "confirmed_only_random.csv", row.names = FALSE)
write.csv(data.frame(confirmed_only_stratified), "confirmed_only_stratified.csv", row.names = FALSE)

write.csv(data.frame(rejected_both), "rejected_both.csv", row.names = FALSE)
write.csv(data.frame(rejected_only_random), "rejected_only_random.csv", row.names = FALSE)
write.csv(data.frame(rejected_only_stratified), "rejected_only_stratified.csv", row.names = FALSE)
