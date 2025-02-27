library(ggplot2)
library(dplyr)
library(scales)

setwd("/Users/vsehtman/PycharmProjects/AE-RL-Pure-Python/data/datasets/boruta")

# Load data
nsl_kdd <- read_csv("formated_train.csv")

# Liste der spezifischen Attacken, die visualisiert werden sollen
selected_attacks <- c("normal", "neptune", "satan", "ipsweep", "portsweep", "smurf", 
                      "nmap", "back", "teardrop", "warezclient", "pod", "guess_passwd", 
                      "buffer_overflow", "warezmaster", "land", "imap", "rootkit", 
                      "loadmodule", "ftp_write", "multihop", "phf", "perl", "spy")

# 🔹 Berechnung der Häufigkeit jeder Attacke (Summe der True-Werte pro Spalte)
attack_counts <- colSums(nsl_kdd[selected_attacks])

# 🔹 Erstelle DataFrame für ggplot2
attack_distribution <- data.frame(
  attack_name = names(attack_counts),
  count = attack_counts
)

# 🔹 Berechnung der relativen Häufigkeiten & Sortierung nach Häufigkeit
attack_distribution <- attack_distribution %>%
  mutate(percentage = count / sum(count)) %>%
  arrange(desc(count))

# 🔹 Farben zuweisen: "normal" grün, alle anderen Attacken rot
attack_distribution$color <- ifelse(attack_distribution$attack_name == "normal", "#4CAF50", "#E74C3C")

# 🔹 Erstelle Labels mit Anzahl der Proben für die Legende
attack_labels <- paste0(attack_distribution$attack_name, " (", attack_distribution$count, ")")

ggplot(attack_distribution, aes(x = reorder(attack_name, -count), y = percentage, fill = attack_name)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  geom_text(aes(
    label = sprintf("%.6f", percentage),  # Immer genau 4 Dezimalstellen
    vjust = ifelse(percentage > 0.3, 0.5, 0.5),  # Höhere Werte nach innen, kleine leicht gesenkt
    y = ifelse(percentage > 0.4, percentage - 0.148, percentage + 0.06),
    hjust = 0.5,  # Exakte Zentrierung auf Balken
    color = ifelse(percentage > 0.4, "white", "black")  # Weiß im Balken, schwarz außerhalb
  ), angle = 90, size = 5, show.legend = FALSE) +
  labs(title = "Relative Distribution of Specific Attacks in the NSL-KDD Trainset",
       x = "Attack Name",
       y = "Relative Frequency") +
  scale_fill_manual(values = setNames(attack_distribution$color, attack_distribution$attack_name)) +
  scale_color_identity() +  # Farbsteuerung der Zahlen
  scale_y_continuous(labels = scales::percent, expand = expansion(mult = c(0.05, 0.1))) +  # Mehr Platz für kleine Labels
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# 🔹 Speichern des Plots als PDF
ggsave("attack_distribution.pdf", width = 12, height = 6)

