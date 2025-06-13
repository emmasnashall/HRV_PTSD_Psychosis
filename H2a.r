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

savefolder <- "Analysis/H1c/All/"

# Create the folder if it doesn't exist
if (!dir.exists(savefolder)) {
    dir.create(savefolder, recursive = TRUE)
}

#filter out the Include
data <- data %>% filter(Include %in% c("TRUE", "REVIEW - TRUE"))

data <- data %>% filter(Timepoint == "T1")
data <- data %>% filter(Trial_No != 0)


#filter out rows with missing values in the variables of interest
data <- data %>% filter(!is.na(HRV_RMSSD))
data <- data %>% filter(!is.na(caps_total))
data <- data %>% filter(!is.na(psd_nominal))
data <- data %>% filter(!is.na(ph2_nominal))
data <- data %>% filter(!is.na(caps_dissociation_cont))
data <- data %>% filter(!is.na(Trial_Type))

# Filter out the lowest quartile of average_quality
# q1 <- quantile(data$average_quality, 0.25, na.rm = TRUE)
# data <- data %>% filter(average_quality > q1)



# 1. Get one value per participant (e.g., first occurrence)
questionnaire_unique <- data %>%
  group_by(Participant) %>%
  slice(1) %>%
  ungroup()

# 2. Compute mean and sd from these unique values
caps_mean <- mean(questionnaire_unique$caps_total, na.rm = TRUE)
caps_sd <- sd(questionnaire_unique$caps_total, na.rm = TRUE)

psd_mean <- mean(questionnaire_unique$psd_nominal, na.rm = TRUE)
psd_sd <- sd(questionnaire_unique$psd_nominal, na.rm = TRUE)

ph2_mean <- mean(questionnaire_unique$ph2_nominal, na.rm = TRUE)
ph2_sd <- sd(questionnaire_unique$ph2_nominal, na.rm = TRUE)

dissociation_mean <- mean(questionnaire_unique$caps_dissociation_cont, na.rm = TRUE)
dissociation_sd <- sd(questionnaire_unique$caps_dissociation_cont, na.rm = TRUE)


# 3. Standardize the variables
data$caps_total_z <- (data$caps_total - caps_mean) / caps_sd
data$psd_nominal_z <- (data$psd_nominal - psd_mean) / psd_sd 
data$ph2_nominal_z <- (data$ph2_nominal - ph2_mean) / ph2_sd
data$dissociative_ptsd_z <- (data$caps_dissociation_cont - dissociation_mean) / dissociation_sd

# 4. Create a new variable for Trial_Type
data$Trial_Type <- droplevels(data$Trial_Type)  # Remove unused levels
data <- data %>% filter(!Trial_Type %in% c("", NA)) %>%  # Remove empty strings and NA
               filter(Trial_Type %in% c("Own", "Control"))
data$Trial_Type <- factor(data$Trial_Type, levels = c("Own", "Control"))  # Set the order of levels


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
    BoxCox_HRV_RMSSD_z ~  (caps_total_z + ph2_nominal_z + psd_nominal_z + dissociative_ptsd_z) * Trial_Type + 
    (1 | Participant) + (1|Run_Number),
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
cat(sprintf("CAPS total: mean = %.3f, sd = %.3f\n", caps_mean, caps_sd))
cat(sprintf("PSD nominal: mean = %.3f, sd = %.3f\n", psd_mean, psd_sd))
cat(sprintf("PH2 nominal: mean = %.3f, sd = %.3f\n", ph2_mean, ph2_sd))
cat(sprintf("Dissociation (CAPS): mean = %.3f, sd = %.3f\n", dissociation_mean, dissociation_sd))

cat("Participants in sample:\n")
cat(unique(as.character(data$Participant)), sep = "\n")

sink()

#