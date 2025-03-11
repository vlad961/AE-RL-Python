# Load packages
library(dplyr)
library(readr)

setwd("/Users/vsehtman/PycharmProjects/AE-RL-Pure-Python/data/datasets/boruta")

# Load Boruta results for balanced, stratified, and random sampling
balanced_sample_results <- read_csv("./strat-vs-balanced/balanced/02_boruta_results_after_tentative_decision.csv")
colnames(balanced_sample_results)[1] <- "Feature"

stratified_sample_results <- read_csv("./rand-vs-strat/strat-no-seed/boruta_results.csv")
colnames(stratified_sample_results)[1] <- "Feature"

random_sample_results <- read_csv("./rand-vs-strat/rand-no-seed/boruta_results.csv")
colnames(random_sample_results)[1] <- "Feature"

# Split features by decision
balanced_confirmed <- balanced_sample_results %>%
  filter(decision == "Confirmed") %>%
  pull(Feature)

balanced_rejected <- balanced_sample_results %>%
  filter(decision == "Rejected") %>%
  pull(Feature)

stratified_confirmed <- stratified_sample_results %>%
  filter(decision == "Confirmed") %>%
  pull(Feature)

stratified_rejected <- stratified_sample_results %>%
  filter(decision == "Rejected") %>%
  pull(Feature)

random_confirmed <- random_sample_results %>%
  filter(decision == "Confirmed") %>%
  pull(Feature)

random_rejected <- random_sample_results %>%
  filter(decision == "Rejected") %>%
  pull(Feature)

# Find common and unique confirmed features
confirmed_all_three <- Reduce(intersect, list(balanced_confirmed, stratified_confirmed, random_confirmed))
confirmed_only_balanced <- setdiff(balanced_confirmed, union(stratified_confirmed, random_confirmed))
confirmed_only_stratified <- setdiff(stratified_confirmed, union(balanced_confirmed, random_confirmed))
confirmed_only_random <- setdiff(random_confirmed, union(balanced_confirmed, stratified_confirmed))

# Find common and unique rejected features
rejected_all_three <- Reduce(intersect, list(balanced_rejected, stratified_rejected, random_rejected))
rejected_only_balanced <- setdiff(balanced_rejected, union(stratified_rejected, random_rejected))
rejected_only_stratified <- setdiff(stratified_rejected, union(balanced_rejected, random_rejected))
rejected_only_random <- setdiff(random_rejected, union(balanced_rejected, stratified_rejected))

# Print results
print("Confirmed in all three datasets:")
print(confirmed_all_three)

print("Only in balanced test confirmed features:")
print(confirmed_only_balanced)

print("Only in stratified test confirmed features:")
print(confirmed_only_stratified)

print("Only in random test confirmed features:")
print(confirmed_only_random)

print("Rejected in all three datasets:")
print(rejected_all_three)

print("Only in balanced test rejected features:")
print(rejected_only_balanced)

print("Only in stratified test rejected features:")
print(rejected_only_stratified)

print("Only in random test rejected features:")
print(rejected_only_random)

# Save results
write.csv(data.frame(confirmed_all_three), "confirmed_all_three.csv", row.names = FALSE)
write.csv(data.frame(confirmed_only_balanced), "confirmed_only_balanced.csv", row.names = FALSE)
write.csv(data.frame(confirmed_only_stratified), "confirmed_only_stratified.csv", row.names = FALSE)
write.csv(data.frame(confirmed_only_random), "confirmed_only_random.csv", row.names = FALSE)

write.csv(data.frame(rejected_all_three), "rejected_all_three.csv", row.names = FALSE)
write.csv(data.frame(rejected_only_balanced), "rejected_only_balanced.csv", row.names = FALSE)
write.csv(data.frame(rejected_only_stratified), "rejected_only_stratified.csv", row.names = FALSE)
write.csv(data.frame(rejected_only_random), "rejected_only_random.csv", row.names = FALSE)
