# Load necessary libraries
library(lme4)
library(ggplot2)
library(dplyr)
library(lmerTest)
library(MASS)

# Load data

data <- read.csv(
    paste0(
        "Processed/Neurokit2/Collated/Reprocessed_23_05/TDA_FQ_Collated_Symptoms.csv"
    )
)

savefolder <- "Analysis/H2a/FQ/Sensitivity/"
#create the folder if it doesn't exist
if (!dir.exists(savefolder)) {
  dir.create(savefolder, recursive = TRUE)
}


#filter out the Include#filter out the Include
#filter out the Include
data <- data %>% filter(Include == "True")

# # Filter out the lowest quartile of average_quality
q1 <- quantile(data$average_quality, 0.25, na.rm = TRUE)
data <- data %>% filter(average_quality > q1)

colnames(data)
data$TrialArmCode <- as.factor(data$TrialArmCode)
data$TrialArmCode <- relevel(data$TrialArmCode, ref = "0")

data$Timepoint <- as.factor(data$Timepoint)
data$Timepoint <- relevel(data$Timepoint, ref = "T1")


data$Run_Type <- ifelse(data$Series %in% c("Run1", "Run2", "Run3"), "Trauma", "Rest")
data$Run_Type <- factor(data$Run_Type, levels = c("Rest", "Trauma"))
data$Series <- factor(data$Series)

data$Run_Number <- ifelse(data$Series == "Rest_Pre", 1,
                            ifelse(data$Series == "Run1", 2,
                                   ifelse(data$Series == "Run2", 3,
                                          ifelse(data$Series == "Run3", 4, 5))))


# Ensure RMSSD is positive and non-zero for BoxCox
data$HRV_HF <- as.numeric(data$HRV_HF)
min_HF <- min(data$HRV_HF, na.rm = TRUE)
if (min_HF <= 0) {
  data$HRV_HF_bc <- data$HRV_HF + abs(min_HF) + 1e-6
} else {
  data$HRV_HF_bc <- data$HRV_HF
}

# Find optimal lambda for BoxCox
bc <- boxcox(HRV_HF_bc ~ 1, data = data, plotit = FALSE)
lambda <- bc$x[which.max(bc$y)]

# Apply BoxCox transformation
if (abs(lambda) < 1e-6) {
  data$BoxCox_HRV_HF <- log(data$HRV_HF_bc)
} else {
  data$BoxCox_HRV_HF <- (data$HRV_HF_bc^lambda - 1) / lambda
}

data$BoxCox_HRV_HF_z <- as.numeric(scale(data$BoxCox_HRV_HF))
#remove outliers
data <- data %>% filter(BoxCox_HRV_HF_z < 3 & BoxCox_HRV_HF_z > -3)


# Fit the linear mixed model
model_with_pvalues <- lmer(BoxCox_HRV_HF_z ~ Run_Type * TrialArmCode * Timepoint + (1 | Participant) + (1 | Run_Number), data = data)

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
cat("Mean (before scaling):", mean(data$BoxCox_HRV_HF, na.rm = TRUE), "\n")
cat("Std (before scaling):", sd(data$BoxCox_HRV_HF, na.rm = TRUE), "\n")

cat("Participants in sample:\n")
cat(unique(as.character(data$Participant)), sep = "\n")

sink()