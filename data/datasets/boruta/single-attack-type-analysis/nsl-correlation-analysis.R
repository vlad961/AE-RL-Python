# install the following packets if not present yet.
#install.packages("Boruta")
#install.packages("readr")
#install.packages("dplyr")
#install.packages("caret")
# load packets
library(Boruta)
library(readr)
library(dplyr)
library(caret)

setwd("/Users/vsehtman/PycharmProjects/AE-RL-Pure-Python/data/datasets/boruta")

# Load data
nsl_kdd <- read_csv("formated_train.csv")
nsl_kdd_test <- read_csv("formated_test.csv")

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

zero_sum_cols <- sapply(nsl_kdd, function(col) sum(col) == 0)
zero_sum_features <- names(zero_sum_cols[zero_sum_cols])

zero_sum_cols_test <- sapply(nsl_kdd_test, function(col) sum(col) == 0)
zero_sum_features_test <- names(zero_sum_cols_test[zero_sum_cols_test])

if (length(zero_sum_features) > 0 || length(zero_sum_features_test) > 0) {
  print("Folgende Spalten enthalten nur Nullen und werden entfernt:")
  print(zero_sum_features)
  print("Folgende Testspalten enthalten nur Nullen")
  print(zero_sum_features_test)
}

nsl_kdd$attack_category <- apply(nsl_kdd[123:162], 1, get_attack_category) # Apply the Mapping mapping on each row
nsl_kdd$attack_category <- as.numeric(as.factor(nsl_kdd$attack_category)) # convert attack types into numerical values (1-5)
#levels(as.factor(apply(final_sample[123:162], 1, get_attack_category))) # check the mapping of attack types to numerical values
nsl_kdd <- nsl_kdd %>% select(1:122)
nsl_kdd <- nsl_kdd %>% select(-num_outbound_cmds) # only 0 values
nsl_kdd <- nsl_kdd %>% select (-srv_serror_rate) # serror_rate
nsl_kdd <- nsl_kdd %>% select (-srv_rerror_rate) # rerror_rate
nsl_kdd <- nsl_kdd %>% select (-S0) # srv_serror_rate, s_error_rate (2nd run)
nsl_kdd <- nsl_kdd %>% select (-dst_host_srv_serror_rate) # srv_serror_rate,  
nsl_kdd <- nsl_kdd %>% select (-dst_host_srv_rerror_rate) # srv_rerror_rate,  
nsl_kdd <- nsl_kdd %>% select (-dst_host_serror_rate) # serror_rate (2nd run)
nsl_kdd <- nsl_kdd %>% select (-dst_host_rerror_rate) # rerror_rate (2nd run)
nsl_kdd <- nsl_kdd %>% select (-num_compromised) # num_root


cor_matrix <- cor(nsl_kdd[,1:113])
highCorr <- findCorrelation(cor_matrix, cutoff = 0.9)
highCorr_features <- colnames(nsl_kdd)[highCorr]

correlation_results <- list()

# Schleife durch alle hoch korrelierten Merkmale
for (feature in highCorr_features) {
  # Extrahiere die Spalte oder Zeile der Korrelationsmatrix
  cor_values <- cor_matrix[feature, ]
  
  # Entferne die Korrelation mit sich selbst
  cor_values <- cor_values[names(cor_values) != feature]
  
  # Höchste positive Korrelation
  max_corr <- max(cor_values)
  max_corr_feature <- names(cor_values)[which.max(cor_values)]
  
  # Niedrigste (negativste) Korrelation
  min_corr <- min(cor_values)
  min_corr_feature <- names(cor_values)[which.min(cor_values)]
  
  # Speichere die Ergebnisse in der Liste
  correlation_results[[feature]] <- list(
    highest_positive = list(feature = max_corr_feature, correlation = max_corr),
    lowest_negative = list(feature = min_corr_feature, correlation = min_corr)
  )
  
  # Ergebnisse ausgeben
  #cat("Feature:", feature, "\n")
  #cat("  Highest positive correlation:", max_corr_feature, "(", max_corr, ")\n")
  #cat("  Lowest negative correlation:", min_corr_feature, "(", min_corr, ")\n")
}


results_df <- do.call(rbind, lapply(names(correlation_results), function(feature) {
  data.frame(
    Feature = feature,
    Highest_Positive_Feature = correlation_results[[feature]]$highest_positive$feature,
    Highest_Positive_Correlation = correlation_results[[feature]]$highest_positive$correlation,
    Lowest_Negative_Feature = correlation_results[[feature]]$lowest_negative$feature,
    Lowest_Negative_Correlation = correlation_results[[feature]]$lowest_negative$correlation
  )
}))

# Ergebnisse anzeigen
print(results_df)

# Optional: Ergebnisse in eine CSV-Datei exportieren
write.csv(results_df, "04_high_corr_analysis_results_rm_num_compromised.csv", row.names = FALSE)