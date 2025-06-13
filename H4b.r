# Load necessary libraries
library(lme4)
library(ggplot2)
library(dplyr)
library(lmerTest)

# Load data
data <- read.csv(
    paste0(
        "Processed/Neurokit2/Collated/BoxCox/T1_T2/H2b/",
        "H2b_TDA_WR_Collated_Pruned_BoxCox_96_filtered_T1_T2.csv"
    )
)


# variable as a factor
data$Timepoint <- as.factor(data$Timepoint)

#filter data to Treatmentdummy == 1
data <- data %>% filter(Treatmentdummy == 1)
data <- data %>% filter(Timepoint == "T1")


# head(data)
# colnames(data)

dependent_variables <- c(
    "caps_total_z_diff",
    "psd_nominal_z_diff",
    "ph2_nominal_z_diff",
    "ph3_nominal_z_diff",
    "psyrats_nom_total_z_diff",
    "dissociative_ptsd_diff"
)
# Fit the linear mixed model
# Assuming 'BoxCox_HRV_RMSSD_z' is the dependent variable


model_formula <- caps_total_z_diff ~ BC_BoxCox_HRV_RMSSD_z_new + (1 | Participant)
model_with_pvalues <- lmer(
    as.formula(model_formula),
    data = data
)

summary(model_with_pvalues)

sink(
    paste0(
        savefolder, "lmm_model_summary.txt"
    )
)
summary(model_with_pvalues)
sink()

saveRDS(
    model_with_pvalues,
    file = paste0(
        savefolder, "lmm_model.rds"

for (dependent_variable in dependent_variables) {
    savefolder <- paste("Processed/Neurokit2/Analysis/H2c/ALL/", dependent_variable, "/", sep = "")
    if (!dir.exists(savefolder)) {
        dir.create(savefolder, recursive = TRUE)
    }

    model_formula <- paste(dependent_variable, "~ BC_BoxCox_HRV_RMSSD_z_new + (1 | Participant) + (1|Run_number)")
    model_with_pvalues <- lmer(
        as.formula(model_formula),
        data = data
    )

    summary(model_with_pvalues)

    sink(
        paste0(
            savefolder, "lmm_model_summary.txt"
        )
    )
    summary(model_with_pvalues)
    sink()

    saveRDS(
        model_with_pvalues,
        file = paste0(
            savefolder, "lmm_model.rds"
        )
    )
}