# Load necessary libraries
library(lme4)
library(ggplot2)
library(dplyr)
library(lmerTest)
library(MASS)

# Load data

data <- read.csv(
    paste0(
        'Processed/neurokit2/collated/reprocessed_23_05/TDA_EV_Collated_Symptoms_with_Baseline_HRV.csv'
    )
)

savefolder <- "Analysis/H2a/EV/All/"
#create the folder if it doesn't exist
if (!dir.exists(savefolder)) {
  dir.create(savefolder, recursive = TRUE)
}


#filter out the Include#filter out the Include
data <- data %>% filter(Include %in% c("TRUE", "REVIEW - TRUE"))
data <- data %>% filter(Trial_No != 0)

# Filter out the lowest quartile of average_quality
# q1 <- quantile(data$average_quality, 0.25, na.rm = TRUE)
# data <- data %>% filter(average_quality > q1)

colnames(data)
data$TrialArmCode <- as.factor(data$TrialArmCode)
data$TrialArmCode <- relevel(data$TrialArmCode, ref = "1")

data$Timepoint <- as.factor(data$Timepoint)
data$Timepoint <- relevel(data$Timepoint, ref = "T1")

data$Trial_Type <- droplevels(data$Trial_Type)  # Remove unused levels
data <- data %>% filter(!Trial_Type %in% c("", NA)) %>%  # Remove empty strings and NA
               filter(Trial_Type %in% c("Own", "Control"))
data$Trial_Type <- factor(data$Trial_Type)  # Set the order of levels
data$Trial_Type <- relevel(data$Trial_Type, ref = "Control")


data$Run_Number <- ifelse(data$Series == "Rest_Pre", 1,
                            ifelse(data$Series == "Run1", 2,
                                   ifelse(data$Series == "Run2", 3,
                                          ifelse(data$Series == "Run3", 4, 5))))


# Ensure RMSSD is positive and non-zero for BoxCox
data$HRV_RMSSD <- as.numeric(data$HRV_RMSSD)
min_rmssd <- min(data$HRV_RMSSD, na.rm = TRUE)
if (min_rmssd <= 0) {
  data$HRV_RMSSD_bc <- data$HRV_RMSSD + abs(min_rmssd) + 1e-6
} else {
  data$HRV_RMSSD_bc <- data$HRV_RMSSD
}

# Find optimal lambda for BoxCox
bc <- boxcox(HRV_RMSSD_bc ~ 1, data = data, plotit = FALSE)
lambda <- bc$x[which.max(bc$y)]

# Apply BoxCox transformation
if (abs(lambda) < 1e-6) {
  data$BoxCox_HRV_RMSSD <- log(data$HRV_RMSSD_bc)
} else {
  data$BoxCox_HRV_RMSSD <- (data$HRV_RMSSD_bc^lambda - 1) / lambda
}

data$BoxCox_HRV_RMSSD_z <- as.numeric(scale(data$BoxCox_HRV_RMSSD))
#remove outliers
data <- data %>% filter(BoxCox_HRV_RMSSD_z < 3 & BoxCox_HRV_RMSSD_z > -3)

#apply the same Boxcox to the Baseline_HRV_RMSSD
data$Baseline_HRV_RMSSD <- as.numeric(data$Baseline_HRV_RMSSD)
min_baseline_rmssd <- min(data$Baseline_HRV_RMSSD, na.rm = TRUE)
if (min_baseline_rmssd <= 0) {
  data$Baseline_HRV_RMSSD_bc <- data$Baseline_HRV_RMSSD + abs(min_baseline_rmssd) + 1e-6
} else {
  data$Baseline_HRV_RMSSD_bc <- data$Baseline_HRV_RMSSD
}

if (abs(lambda) < 1e-6) {
  data$BoxCox_Baseline_HRV_RMSSD <- log(data$Baseline_HRV_RMSSD_bc)
} else {
  data$BoxCox_Baseline_HRV_RMSSD <- (data$Baseline_HRV_RMSSD_bc^lambda - 1) / lambda
}

# Scale the BoxCox transformed HRV_RMSSD using the mean and standard deviation of the BoxCox_HRV_RMSSD
mean_bc <- mean(data$BoxCox_HRV_RMSSD, na.rm = TRUE)
sd_bc <- sd(data$BoxCox_HRV_RMSSD, na.rm = TRUE)
data$BoxCox_Baseline_HRV_RMSSD_z <- (data$BoxCox_Baseline_HRV_RMSSD - mean_bc) / sd_bc

# Fit the linear mixed model
model_formula <- BoxCox_HRV_RMSSD_z ~ Trial_Type * TrialArmCode * Timepoint + BoxCox_Baseline_HRV_RMSSD_z + (1 | Participant) + (1 | Run_Number)

model_with_pvalues <- lmer(model_formula, data = data)

levels(data$Timepoint)

summary(model_with_pvalues)
# Save model output as txt file
sink(paste0(
  savefolder, "lmm_model_summary.txt"
))
summary(model_with_pvalues)
sink()

saveRDS(model_with_pvalues, file = paste0(savefolder, "lmm_model.rds"))


vif_values <- car::vif(model_with_pvalues)

sink(paste0(savefolder, "lmm_model_diagnostics.txt"))
cat("AIC:", AIC(model_with_pvalues), "\n")
cat("BIC:", BIC(model_with_pvalues), "\n")
cat("Log-Likelihood:", logLik(model_with_pvalues), "\n")

# Calculate VIF for fixed effects in the mixed model
cat("VIF for fixed effects in the mixed model:\n")
print(vif_values)

sink()

sink(paste0(savefolder, "sample_info.txt"))
cat("BoxCox lambda:", lambda, "\n")
cat("Mean (before scaling):", mean(data$BoxCox_HRV_RMSSD, na.rm = TRUE), "\n")
cat("Std (before scaling):", sd(data$BoxCox_HRV_RMSSD, na.rm = TRUE), "\n")

cat("Participants in sample:\n")
cat(unique(as.character(data$Participant)), sep = "\n")

sink()