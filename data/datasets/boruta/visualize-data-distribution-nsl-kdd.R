
library(ggplot2)

setwd("/Users/vsehtman/PycharmProjects/AE-RL-Pure-Python/data/datasets/boruta")

# Load data
nsl_kdd <- read_csv("formated_train.csv")

attack_map <- list(
  Normal = c("normal"),
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

nsl_kdd$attack_category <- factor(nsl_kdd$attack_category, 
                                  levels = c(1, 2, 3, 4, 5),
                                  labels = c("DoS", "Normal", "Probe", "R2L", "U2R"))

attack_counts <- table(nsl_kdd$attack_category)
modified_order <- c("Normal", "DoS", "Probe", "R2L", "U2R")
attack_counts <- attack_counts[modified_order]

attack_distribution <- nsl_kdd %>%
  group_by(attack_category) %>%
  summarise(count = n()) %>%
  mutate(percentage = count / sum(count))

# Sortiere die Angriffskategorien nach Häufigkeit (höchste zuerst)
attack_distribution$attack_category <- factor(attack_distribution$attack_category, 
                                              levels = attack_distribution$attack_category[order(-attack_distribution$count)])


# Erstelle Labels mit den Angriffskategorien + Anzahl der Proben
attack_labels <- paste0(names(attack_counts), " (", attack_counts, ")")


ggplot(attack_distribution, aes(x = attack_category, y = percentage, fill = attack_category)) +
  geom_bar(stat = "identity") +
  geom_text(aes(label = scales::label_number(accuracy = 0.0001)(percentage)), vjust = -0.5, size = 5) +  # Prozentwerte über Balken
  labs(title = "Relative Distribution of Attack types in the NSL-KDD Trainset",
       x = "Attacktype",
       y = "Relative Frequency") +
  scale_fill_manual(values = c("#4CAF50", "#E74C3C", "#E74C3C", "#E74C3C", "#E74C3C"),
                    name = "Attacktypes and count",
                    labels = attack_labels) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave("attack_category_distribution.pdf", width = 10, height = 6)

