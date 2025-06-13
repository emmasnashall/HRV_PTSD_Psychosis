
# lmm_analysis.R

# Load necessary libraries
library(lme4)
library(ggplot2)
library(dplyr)
library(lmerTest)
library(MASS)

# Load data
data <- read.csv("Processed/Neurokit2/Collated/Reprocessed_23_05/TDA_EV_Collated.csv")

savefolder <- paste0("Analysis/H1b/Sensitivity/")
# Create the folder if it doesn't exist
if (!dir.exists(savefolder)) {
  dir.create(savefolder, recursive = TRUE)
}

#filter out the Include
data <- data %>% filter(Include == "TRUE")
data <- data %>% filter(Timepoint == "T1")
data <- data %>% filter(Trial_No != 0)


# # Filter out the lowest quartile of average_quality
q1 <- quantile(data$average_quality, 0.25, na.rm = TRUE)
data <- data %>% filter(average_quality > q1)

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

#Participant <- as.factor(data$Participant)
data$Participant <- as.factor(data$Participant)


#change Run_Type to a categorical variable
if (!is.factor(data$Trial_Type)) {
  data$Trial_Type <- as.factor(data$Trial_Type)
}


data$Run_Number <- ifelse(data$Series == "Rest_Pre", 1,
                            ifelse(data$Series == "Run1", 2,
                                   ifelse(data$Series == "Run2", 3,
                                          ifelse(data$Series == "Run3", 4, 5))))

#check levels of Trial_Type

#filter for Trial_Type == "Own" and "Control"
# Remove empty levels and filter
data$Trial_Type <- droplevels(data$Trial_Type)  # Remove unused levels
data <- data %>% filter(!Trial_Type %in% c("", NA)) %>%  # Remove empty strings and NA
               filter(Trial_Type %in% c("Own", "Control"))
levels(data$Trial_Type)
#model_with_pvalues <- lmer(BoxCox_HRV_RMSSD ~ Run_Type + average_quality + (1 | Participant) + (1 | Participant:Run_number), data = data)
model_with_pvalues <- lmer(BoxCox_HRV_RMSSD_z ~ Trial_Type + (1 | Participant) + (1| Run_Number), data = data)


summary(model_with_pvalues)
# Save model output as txt file
sink(paste0(
  savefolder, "Lmm_model_summary.txt"
))
summary(model_with_pvalues)
sink()

saveRDS(model_with_pvalues, file = paste0(savefolder,"lmm_model.rds"))

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