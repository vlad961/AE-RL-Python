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

# ==========================
#  Use only X samples
# ==========================
#set.seed(42) #uncomment to create reproducible results
nsl_kdd <- nsl_kdd %>% sample_n(12598) # (10% of random chosen data)

# convert attack types into numerical values (1-5)
nsl_kdd$attack_category <- as.numeric(as.factor(nsl_kdd$attack_category))
#levels(as.factor(apply(final_sample[123:162], 1, get_attack_category))) # check the mapping of attack types to numerical values


# ==========================
#  Use 10% of data stratified by `attack_category`
# ==========================
#sample_index <- createDataPartition(nsl_kdd$attack_category, p = 0.1, list = FALSE)
# select only those rows from original DataFrame
#nsl_kdd_sample <- nsl_kdd[sample_index, ]

# select only features without labels
features <- nsl_kdd %>% select(1:122)
#features <- nsl_kdd_sample %>% select(1:122)  # in case the stratified data shall be used.
# run Boruta algorithm to determine all relevant features
boruta_result <- Boruta(nsl_kdd$attack_category ~ ., data = features, doTrace = 1)

# analyse results
print(boruta_result)
pdf("boruta_plots.pdf")
plot(boruta_result, las = 2)  # Plot feature importance.
plotImpHistory(boruta_result) # Shows the development of Z-Scores over the iterations
dev.off()  # finish writing into pdf file.

# extract important features
confirmed_features <- getConfirmedFormula(boruta_result)
print(confirmed_features)

# In case tentative features are present, decide automatically.
boruta_result <- TentativeRoughFix(boruta_result)
print(getConfirmedFormula(boruta_result))

# save results
write.csv(attStats(boruta_result), "boruta_results_10percent_rand.csv", row.names = TRUE)
