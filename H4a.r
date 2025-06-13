# Load necessary libraries
library(lme4)
library(ggplot2)
library(dplyr)
library(lmerTest)
library(sandwich)
library(car)
library(MASS)

# Load data
data <- read.csv(
    paste0(
        "Processed/Neurokit2/Collated/Reprocessed_23_05/TDA_EV_Collated_Symptoms.csv"
    )
)

savefolder <- "Analysis/H2b/PH2/Sensitivity/"

# Create the folder if it doesn't exist
if (!dir.exists(savefolder)) {
    dir.create(savefolder, recursive = TRUE)
}

#filter out the Include
data <- data %>% filter(Include %in% c("TRUE", "REVIEW - TRUE"))
data <- data %>% filter(Trial_No != 0)
data <- data %>% filter(TrialArmCode != "1")  # Exclude TrialArmCode 0


#filter out rows with missing values in the variables of interest
data <- data %>% filter(!is.na(HRV_RMSSD))
data <- data %>% filter(!is.na(caps_total_diff))
data <- data %>% filter(!is.na(caps_total_baseline))
# data <- data %>% filter(!is.na(psd_nominal_diff))
# data <- data %>% filter(!is.na(psd_nominal_baseline))
data <- data %>% filter(!is.na(ph2_nominal_diff))
data <- data %>% filter(!is.na(ph2_nominal_baseline))
data <- data %>% filter(!is.na(Trial_Type))
#data <- data %>% filter((TrialArmCode =))

# Filter out the lowest quartile of average_quality
q1 <- quantile(data$average_quality, 0.25, na.rm = TRUE)
data <- data %>% filter(average_quality > q1)

# 1. Get one value per participant (e.g., first occurrence)
questionnaire_unique <- data %>%
  group_by(Participant) %>%
  slice(1) %>%
  ungroup()

# 2. Compute mean and sd from these unique values
caps_total_diff_mean <- mean(questionnaire_unique$caps_total_diff, na.rm = TRUE)
caps_total_diff_sd <- sd(questionnaire_unique$caps_total_diff, na.rm = TRUE)

caps_total_baseline_mean <- mean(questionnaire_unique$caps_total_baseline, na.rm = TRUE)
caps_total_baseline_sd <- sd(questionnaire_unique$caps_total_baseline, na.rm = TRUE)

# psd_nominal_diff_mean <- mean(questionnaire_unique$psd_nominal_diff, na.rm = TRUE)
# psd_nominal_diff_sd <- sd(questionnaire_unique$psd_nominal_diff, na.rm = TRUE)

# psd_nominal_baseline_mean <- mean(questionnaire_unique$psd_nominal_baseline, na.rm = TRUE)
# psd_nominal_baseline_sd <- sd(questionnaire_unique$psd_nominal_baseline, na.rm = TRUE)

ph2_nominal_diff_mean <- mean(questionnaire_unique$ph2_nominal_diff, na.rm = TRUE)
ph2_nominal_diff_sd <- sd(questionnaire_unique$ph2_nominal_diff, na.rm = TRUE)

ph2_nominal_baseline_mean <- mean(questionnaire_unique$ph2_nominal_baseline, na.rm = TRUE)
ph2_nominal_baseline_sd <- sd(questionnaire_unique$ph2_nominal_baseline, na.rm = TRUE)




# 3. Standardize the variables
data$caps_total_diff_z <- (data$caps_total_diff - caps_total_diff_mean) / caps_total_diff_sd
data$caps_total_baseline_z <- (data$caps_total_baseline - caps_total_baseline_mean) / caps_total_baseline_sd

# data$psd_nominal_diff_z <- (data$psd_nominal_diff - psd_nominal_diff_mean) / psd_nominal_diff_sd
# data$psd_nominal_baseline_z <- (data$psd_nominal_baseline - psd_nominal_baseline_mean) / psd_nominal_baseline_sd

data$ph2_nominal_diff_z <- (data$ph2_nominal_diff - ph2_nominal_diff_mean) / ph2_nominal_diff_sd
data$ph2_nominal_baseline_z <- (data$ph2_nominal_baseline - ph2_nominal_baseline_mean) / ph2_nominal_baseline_sd


# 4. Create a new variable for Trial_Type
data$Trial_Type <- droplevels(data$Trial_Type)  # Remove unused levels
data <- data %>% filter(!Trial_Type %in% c("", NA)) %>%  # Remove empty strings and NA
               filter(Trial_Type %in% c("Own", "Control"))
data$Trial_Type <- factor(data$Trial_Type, levels = c("Own", "Control"))  # Set the order of levels

data$Timepoint <- as.factor(data$Timepoint)
data$Timepoint <- relevel(data$Timepoint, ref = "T1")

# data$TrialArmCode <- as.factor(data$TrialArmCode)
# data$TrialArmCode <- relevel(data$TrialArmCode, ref = "1")

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


# Fit the linear mixed model
# Assuming 'BoxCox_HRV_RMSSD_z' is the dependent variable
model_with_pvalues <- lmer(
   model_formula <- BoxCox_HRV_RMSSD_z ~ 
  caps_total_diff_z * Timepoint +
  # psd_nominal_diff_z * Timepoint +
  ph2_nominal_diff_z * Timepoint +
  Trial_Type * Timepoint + 
  # psd_nominal_baseline_z +
  caps_total_baseline_z + 
  ph2_nominal_baseline_z +
  (1 | Participant) + 
  (1 | Run_Number),
    data = data
)

# Display model summary
summary(model_with_pvalues)

# Save model output as a text file
sink(
    paste0(
        savefolder, "lmm_model_summary.txt"
    )
)
summary(model_with_pvalues)
sink()

# Save the model as an RDS file
saveRDS(
    model_with_pvalues,
    file = paste0(
        savefolder, "lmm_model.rds"
    )
)

# Calculate VIF for fixed effects using car::vif directly on the lmer model
# This works for lmerTest/lme4 models with car >= 3.0-12
vif_values <- car::vif(model_with_pvalues)

# Optionally, visualize the results
# source("R/visualization/plot_results.R")
# lmm_analysis.R
sink(paste0(savefolder, "lmm_model_diagnostics.txt"))
cat("AIC:", AIC(model_with_pvalues), "\n")
cat("BIC:", BIC(model_with_pvalues), "\n")
cat("Log-Likelihood:", logLik(model_with_pvalues), "\n")

# Calculate VIF for fixed effects in the mixed model


cat("VIF for fixed effects in the mixed model:\n")
print(vif_values)



sink(paste0(savefolder, "sample_info.txt"))
cat("BoxCox lambda:", lambda, "\n")
cat("Mean (before scaling):", mean(data$BoxCox_HRV_RMSSD, na.rm = TRUE), "\n")
cat("Std (before scaling):", sd(data$BoxCox_HRV_RMSSD, na.rm = TRUE), "\n")

cat("Means and SDs of questionnaires (from unique participants):\n")
cat(sprintf("CAPS total diff: mean = %.3f, sd = %.3f\n", caps_total_diff_mean, caps_total_diff_sd))
cat(sprintf("CAPS total baseline: mean = %.3f, sd = %.3f\n", caps_total_baseline_mean, caps_total_baseline_sd))
cat(sprintf("PSD nominal diff: mean = %.3f, sd = %.3f\n", psd_nominal_diff_mean, psd_nominal_diff_sd))
cat(sprintf("PSD nominal baseline: mean = %.3f, sd = %.3f\n", psd_nominal_baseline_mean, psd_nominal_baseline_sd))
cat(sprintf("PH2 nominal diff: mean = %.3f, sd = %.3f\n", ph2_nominal_diff_mean, ph2_nominal_diff_sd))
cat(sprintf("PH2 nominal baseline: mean = %.3f, sd = %.3f\n", ph2_nominal_baseline_mean, ph2_nominal_baseline_sd))

cat("Participants in sample:\n")
cat(unique(as.character(data$Participant)), sep = "\n")

sink()

#