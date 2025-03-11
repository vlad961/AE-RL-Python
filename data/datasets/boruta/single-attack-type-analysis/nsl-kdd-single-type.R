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

nsl_kdd$attack_category <- apply(nsl_kdd[122:162], 1, get_attack_category) # Apply the Mapping mapping on each row
nsl_kdd$attack_category <- as.numeric(as.factor(nsl_kdd$attack_category)) # convert attack types into numerical values (1-5)
#levels(as.factor(apply(final_sample[123:162], 1, get_attack_category))) # check the mapping of attack types to numerical values

# Use all samples of attack_category == 4 (R2L) and == 5 (U2R)
sample_r2l <- nsl_kdd %>% filter(attack_category == 4)
sample_u2r <- nsl_kdd %>% filter(attack_category == 5)
sample_dos <- nsl_kdd %>% filter(attack_category == 1) %>% sample_n(2000) # 5000
sample_normal <- nsl_kdd %>% filter(attack_category == 2) %>% sample_n(2000) # 5000
sample_probe <- nsl_kdd %>% filter(attack_category == 3) %>% sample_n(2000) # 11656

run_boruta_analysis <- function(sample_data, category_name, normal_samples) {
  # Kombiniere die Angriffstypen mit Normal-Traffic
  sample_data <- bind_rows(sample_data, normal_samples)
  
  features <- sample_data %>% select(1:122)
  boruta_result <- Boruta(x = features, y = sample_data$attack_category, doTrace = 1)
  boruta_tentative_decided <- TentativeRoughFix(boruta_result)
  
  # Speichere die Ergebnisse
  print(boruta_result)
  
  timestamp <- format(Sys.time(), "%Y%m%d_%H%M%S")
  pdf(paste0("00_boruta_plots_", category_name, "_vs_Normal_", timestamp, ".pdf"))
  plot(boruta_result, las = 2, cex.axis = 0.37)
  plotImpHistory(boruta_result)
  dev.off()
  
  # Exportiere die wichtigen Features
  write.csv(attStats(boruta_result), paste0("01_boruta_result_", category_name, "_vs_Normal_before_tentative.csv"), row.names = TRUE)
  write.csv(attStats(boruta_tentative_decided), paste0("02_boruta_result_", category_name, "_vs_Normal_after_tentative.csv"), row.names = TRUE)
  
  confirmed_features <- getSelectedAttributes(boruta_tentative_decided, withTentative = FALSE)
  write.csv(data.frame(Feature = confirmed_features), paste0("03_boruta_confirmed_features_final_", category_name, "_vs_Normal.csv"), row.names = FALSE)
}

# Rufe die Funktion für jede Angriffskategorie auf, jetzt mit "normal" als Vergleichsgruppe
run_boruta_analysis(sample_r2l, "R2L", sample_normal)
run_boruta_analysis(sample_u2r, "U2R", sample_normal)
run_boruta_analysis(sample_dos, "DoS", sample_normal)
run_boruta_analysis(sample_probe, "Probe", sample_normal)