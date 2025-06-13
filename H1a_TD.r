
# lmm_analysis.R

# Load necessary libraries
library(lme4)
library(ggplot2)
library(dplyr)
library(lmerTest)
library(influence.ME)
library(MASS)

# Load data
data <- read.csv(paste0(
  "Processed/Neurokit2/Collated/Reprocessed_23_05/TDA_WR_Collated.csv"
))

savefolder <- paste0("Analysis/H1a/TD/Sensitivity/")
# Create the folder if it doesn't exist
if (!dir.exists(savefolder)) {
  dir.create(savefolder, recursive = TRUE)
}


#filter out the Include
data <- data %>% filter(Include == "True")
data <- data %>% filter(!is.na(Include))
data <- data %>% filter(Timepoint == "T1")


data$Run_Type <- ifelse(data$Series %in% c("Run1", "Run2", "Run3"), "Trauma", "Rest")
data$Run_Type <- factor(data$Run_Type, levels = c("Rest", "Trauma"))
data$Series <- factor(data$Series)


#Rest_Pre == 1, Run1 == 2, Run2 == 3, Run3 == 4, Rest_Post == 5
data$Run_Number <- ifelse(data$Series == "Rest_Pre", 1,
                            ifelse(data$Series == "Run1", 2,
                                   ifelse(data$Series == "Run2", 3,
                                          ifelse(data$Series == "Run3", 4, 5))))


# # # Filter out the lowest quartile of average_quality
q1 <- quantile(data$average_quality, 0.25, na.rm = TRUE)
data <- data %>% filter(average_quality > q1)


#boxcox transform HRV_RMSSD


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

data$Participant <- as.factor(data$Participant)

#z-score the BoxCox_HRV_RMSSD
data$BoxCox_HRV_RMSSD_z <- as.numeric(scale(data$BoxCox_HRV_RMSSD))

# # Simple linear regression
# lm_hrv_adj <- lm(HRV_RMSSD ~ Heart_Period, data = data)

# # Extract residuals
# data$HRV_RMSSD_adj <- residuals(lm_hrv_adj)

#model_with_pvalues <- lmer(BoxCox_HRV_RMSSD ~ Run_Type + average_quality + (1 | Participant) + (1 | Participant:Run_number), data = data)
model_with_pvalues <- lmer(BoxCox_HRV_RMSSD_z ~ Run_Type + (1 | Participant), 
                          # weights = weights,
                          data = data)


summary(model_with_pvalues)
# Save model output as txt file
sink(paste0(
  savefolder, "Lmm_model_summary.txt"
))
summary(model_with_pvalues)
sink()

saveRDS(model_with_pvalues, file = paste0(savefolder, "lmm_model.rds"))

# Optionally, visualize the results
# source("R/visualization/plot_results.R")
# lmm_analysis.R
sink(paste0(savefolder, "lmm_model_diagnostics.txt"))
cat("AIC:", AIC(model_with_pvalues), "\n")
cat("BIC:", BIC(model_with_pvalues), "\n")
cat("Log-Likelihood:", logLik(model_with_pvalues), "\n")

sink()

sink(paste0(savefolder, "sample_info.txt"))
cat("BoxCox lambda:", lambda, "\n")
cat("Mean (before scaling):", mean(data$BoxCox_HRV_RMSSD, na.rm = TRUE), "\n")
cat("Std (before scaling):", sd(data$BoxCox_HRV_RMSSD, na.rm = TRUE), "\n")

cat("Participants in sample:\n")
cat(unique(as.character(data$Participant)), sep = "\n")

sink()

#
