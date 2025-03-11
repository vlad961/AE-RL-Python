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

# Load data
nsl_kdd <- read_csv("formated_train.csv")

attack_map <- list(
  normal = c("normal"),
  DoS = c("back", "land", "neptune", "pod", "smurf", "teardrop", "mailbomb", "apache2", "processtable", "udpstorm"),
  Probe = c("ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"),
  R2L = c("ftp_write", "guess_passwd", "imap", "multihop", "phf", "spy", "warezclient", "warezmaster", "sendmail",
          "named", "snmpgetattack", "snmpguess", "xlock", "xsnoop", "worm"),
  U2R = c("buffer_overflow", "loadmodule", "perl", "rootkit", "httptunnel", "ps", "sqlattack", "xterm")
)

# Function to map attacks to attack types
get_attack_category <- function(row) {
  for (category in names(attack_map)) {
    if (sum(row[attack_map[[category]]]) > 0) {
      return(category)
    }
  }
  return("unknown")  # in case something unexpected happens
}

# Apply the Mapping mapping on each row
nsl_kdd$attack_category <- apply(nsl_kdd[123:162], 1, get_attack_category)

# convert attack types into numerical values (1-5)
nsl_kdd$attack_category <- as.numeric(as.factor(nsl_kdd$attack_category))
#levels(as.factor(apply(final_sample[123:162], 1, get_attack_category))) # check the mapping of attack types to numerical values

# Use all samples of attack_category == 4 (R2L) and == 5 (U2R)
sample_r2l <- nsl_kdd %>% filter(attack_category == 4)
sample_u2r <- nsl_kdd %>% filter(attack_category == 5)

remaining_target <- 11655 / 3
sample_dos <- nsl_kdd %>% filter(attack_category == 1) %>% sample_n(remaining_target)
sample_normal <- nsl_kdd %>% filter(attack_category == 2) %>% sample_n(remaining_target)
sample_probe <- nsl_kdd %>% filter(attack_category == 3) %>% sample_n(remaining_target)

# combine attack type samples into one set.
final_sample <- bind_rows(sample_r2l, sample_u2r, sample_dos, sample_normal, sample_probe)

# check sample distribution
#table(final_sample$attack_category)
#print(nrow(final_sample))  # Should be 12702 samples in case of the original nsl-kdd test dataset.

# select only features without labels
features <- final_sample %>% select(1:122)
# run Boruta algorithm to determine all relevant features
boruta_result <- Boruta(x = features, y = final_sample$attack_category, doTrace = 1)
# In case tentative features are present, decide automatically.
boruta_tentative_decided <- TentativeRoughFix(boruta_result)

# save results
print(boruta_result)
pdf("00_boruta_plots.pdf")
plot(boruta_result, las = 2, cex.axis = 0.37)  # Plot feature importance.
plotImpHistory(boruta_result) # Shows the development of Z-Scores over the iterations
dev.off()  # finish writing into pdf file.

# extract important features
confirmed_features_and_tentative <- getSelectedAttributes(boruta_result, withTentative = TRUE)

write.csv(attStats(boruta_result), "01_boruta_results_before_tentative_decision.csv", row.names = TRUE) # result before TentativeRoughFix
write.csv(attStats(boruta_tentative_decided), "02_boruta_results_after_tentative_decision.csv", row.names = TRUE) # result after TentativeRoughFix
write.csv(data.frame(Feature = confirmed_features_and_tentative), 
          "03_boruta_confirmed_and_tentative_features.csv", row.names = FALSE) # confirmed features + tentative Features before Tentative Decision

# extract important features after Tentative Decision
confirmed_features_final <- getSelectedAttributes(boruta_tentative_decided, withTentative = FALSE)
# final confirmed features after Tentative Decision
write.csv(data.frame(Feature = confirmed_features_final),
          "03_boruta_confirmed_features_final.csv", row.names = FALSE)

