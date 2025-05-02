# install the following packets if not present yet.
#install.packages("Boruta")
#install.packages("readr")
#install.packages("dplyr")
#install.packages("caret")
# load packets
library(Boruta)
library(readr)
library(dplyr)
library(caret) # in case stratified tests are executed

setwd("/Users/vsehtman/PycharmProjects/AE-RL-Pure-Python/data/datasets/boruta")

# Load data
nsl_kdd <- read_csv("formated_train.csv")
# Beneatch uncommented features are highly correlated.
nsl_kdd <- nsl_kdd %>% select(-num_outbound_cmds) # only 0 values
nsl_kdd <- nsl_kdd %>% select (-srv_serror_rate) # serror_rate
nsl_kdd <- nsl_kdd %>% select (-srv_rerror_rate) # rerror_rate
nsl_kdd <- nsl_kdd %>% select (-S0) # srv_serror_rate, s_error_rate (2nd run)
nsl_kdd <- nsl_kdd %>% select (-dst_host_srv_serror_rate) # srv_serror_rate,  
nsl_kdd <- nsl_kdd %>% select (-dst_host_srv_rerror_rate) # srv_rerror_rate,  
nsl_kdd <- nsl_kdd %>% select (-dst_host_serror_rate) # serror_rate (2nd run)
nsl_kdd <- nsl_kdd %>% select (-dst_host_rerror_rate) # rerror_rate (2nd run)
nsl_kdd <- nsl_kdd %>% select (-num_compromised) # num_root

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
#nsl_kdd$attack_category <- apply(nsl_kdd[114:154], 1, get_attack_category)
nsl_kdd$attack_category <- apply(nsl_kdd[113:153], 1, get_attack_category) # manually filtered high correlated features

# ==========================
#  Use only X samples
# ==========================
#set.seed(42) #uncomment to create reproducible results
#nsl_kdd <- nsl_kdd %>% sample_n(12598) # (10% of random chosen data)

# convert attack types into numerical values (1-5)
nsl_kdd$attack_category <- as.numeric(as.factor(nsl_kdd$attack_category))
#levels(as.factor(apply(final_sample[123:162], 1, get_attack_category))) # check the mapping of attack types to numerical values


# ==========================
#  Use 10% of data stratified by `attack_category`
# ==========================
sample_index <- createDataPartition(nsl_kdd$attack_category, p = 0.1, list = FALSE)
# select only those rows from original DataFrame
nsl_kdd_sample <- nsl_kdd[sample_index, ]

# select only features without labels
#features <- nsl_kdd %>% select(1:122)
#features <- nsl_kdd_sample %>% select(1:122)  # in case the stratified data shall be used.
features <- nsl_kdd_sample %>% select(1:113)  # in case the manually filtered highly correlated featuers.
# run Boruta algorithm to determine all relevant features
boruta_result <- Boruta(x = features, y = nsl_kdd_sample$attack_category, doTrace = 1)
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
