# These R codes are used to assess the performance
# of the trained models on example, CASF, OOD, 0LB datasets. 


# library Packages
library(Kendall)
library(ggplot2)
library(ggExtra)
library(ggside)
library(MLmetrics)
library(pROC)
library(yardstick)
library(dplyr)
library(boot)
library(cowplot)
library(tibble)
library(scales)
library(patchwork)
library(grid)


# Example dataset:
# Read in the example_predictions.csv
# There are 241 predictions in total
example_predictions <- read.csv("example_predictions.csv")


# PCC for example dataset predictions 
example_pcc <- cor(example_predictions$pK, 
                   example_predictions$final_pred_pKd, method = "pearson")


# 95% BCa Confidence Interval for PCC on example dataset predictions
# We first set.seed()
set.seed(2025)


# Define the function used for calculating PCC 
stat_pcc <- function(data, indices) {
  cor(data$y[indices], data$yhat[indices], method = "pearson")
}


# Create the dataframe from example dataset which only 
# contain the true pK column and the predicted pK column
df <- data.frame(y = example_predictions$pK,
                 yhat = example_predictions$final_pred_pKd)


# Using bootstrap to calculate the 95% confidence interval
# We repeat the process for 10000 times.
example_PCC_CI <- boot(data = df, statistic = stat_pcc, R = 10000)


# 95% BCa Confidence interval for example_PCC_CI
boot.ci(example_PCC_CI, type = "bca")


# PCC plot for example dataset predictions
# Calculate the range of the data 
example_lims <- range(c(example_predictions$pK, 
                example_predictions$final_pred_pKd))


# Plot the dot figure using ggplot
# We want the figure to be squared
example_figure <- ggplot(example_predictions, aes(x = pK, y = final_pred_pKd)) +
  geom_point(color = "darkgreen", size = 1.0, alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  coord_equal() +
  xlim(example_lims) + ylim(example_lims) +
  labs(
    title = "Predicted vs Experimental pK",
    x = "Experimental PK",
    y = "Predicted PK"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(size = 10, hjust = 0),
    axis.title = element_text(size = 10),
    axis.text  = element_text(size = 10)
  )


# Add the marginal density line 
example_pcc_figure <- ggMarginal(
  example_figure, type = "density", margins = "both",
  fill = "darkgreen", alpha = 0.25
)


example_pcc_figure


# RMSE Calculation for example dataset predictions 
example_RMSE <- sqrt(mean((example_predictions$pK - 
                             example_predictions$final_pred_pKd)^2))


# RMSE 95% Confidence interval for example dataset predictions
set.seed(2025)


# Define the RMSE function we used in bootstrap
stat_rmse <- function(d, i) sqrt(mean((d$y[i] - d$yhat[i])^2))


# Analogy to the PCC, we extract the same dataframe from 
# example dataset for RMSE
d <- data.frame(
  y    = example_predictions$pK,
  yhat = example_predictions$final_pred_pKd
)


# Calculate the bootstrap 10000 times.
example_RMSE_CI  <- boot(d, stat_rmse, R = 10000)          
        
                               
boot.ci(example_RMSE_CI, type = "bca")       


# Define tau-a: this is written in the dissertation
# We define this tau_a manually in a function.
kendall_tau_a <- function(x, y) {        
  n <- length(x)                         
  nc <- nd <- 0L                         
  for (i in 1:(n-1)) {                   
    s <- (x[i] - x[(i+1):n]) *           
      (y[i] - y[(i+1):n])             
    nc <- nc + sum(s > 0)                
    nd <- nd + sum(s < 0)                
    
  }
  (nc - nd) / (n * (n - 1) / 2)          
}


#  tau-a calculation for example dataset predictions
example_tau_a <- kendall_tau_a(example_predictions$pK, 
                               example_predictions$final_pred_pKd)


# 95% Confidence interval for example dataset predictions for tau_a
# The whole calculation process is similar to above.
set.seed(2025)


stat_tau_a <- function(d, i) kendall_tau_a(d$y[i], d$yhat[i])


d <- data.frame(
  y    = example_predictions$pK,
  yhat = example_predictions$final_pred_pKd
)


# bootstrap + BCa
example_tau_a_CI  <- boot(d, stat_tau_a, R = 10000)
          
                
boot.ci(example_tau_a_CI , type = "bca") 


# Kendall’s tau correlation coefficient 
# Apart from tau_a, we also given out tau_b for 3 benchmarks, 
# this is slightly different to the equation written in dissertation.
# we calculate the tau_b using Kendall function in Kendall package,
# and we could also calculate it using cor function;
example_tau_b <- Kendall(example_predictions$pK, 
                         example_predictions$final_pred_pKd)


# 95% Confidence interval for tau_b for example dataset predictions
# Same logic is applied to above 95% BCa Confidence interval above.
set.seed(2025)


stat_tau_b <- function(d, i) Kendall(d$y[i], d$yhat[i])$tau


d <- data.frame(
  y    = example_predictions$pK,
  yhat = example_predictions$final_pred_pKd
)


example_tau_b_CI  <- boot(d, stat_tau_b, R = 10000)          # bootstrap
                  
              
boot.ci(example_tau_b_CI, type = "bca")           


# Classification task 
# example_label indicates the true pK label in example dataset 
example_label <- as.integer(example_predictions$pK >= 4)


# ROC-AUC for example dataset predictions 
# Method 1:
example_roc <- roc(example_label, example_predictions$final_bind_prob)

example_auc <- auc(example_roc)

plot(example_roc, col = "darkgreen", lwd = 2,
     main = paste("ROC Curve (AUC =", round(auc(example_roc), 3), ")"))
 

# Method 2
df <- data.frame(
  truth = factor(example_label, levels = c(1,0)),        
  .pred_1 = example_predictions$final_bind_prob             
)


# AUC
roc_auc(df, truth = truth, .pred_1)


# ROC Curve
roc_curve(df, truth = truth, .pred_1) %>%
  autoplot(colour = "red") +
  ggtitle("ROC Curve for example dataset")



# We set threshold to be 0.5 temporarily, 
# if the probability is >= 0.5, we consider the sample as positive
# and each time it can be changed manually
example_predict <- as.integer(example_predictions$final_bind_prob >= 0.5)


# Create labels for example dataset predictions
TP <- sum(example_predict == 1 & example_label == 1)
FP <- sum(example_predict == 1 & example_label == 0)
TN <- sum(example_predict == 0 & example_label == 0)
FN <- sum(example_predict == 0 & example_label == 1)


# Precision for example dataset predictions
example_precision <- ifelse(TP+FP == 0, NA, TP/(TP+FP))


# Recall for example dataset predictions
example_recall <- ifelse(TP+FN == 0, NA, TP/(TP+FN))


# Specificity for example dataset predictions
example_Specificity <- ifelse((TN + FP) == 0, NA, TN / (TN + FP))


# Balanced Accuracy for example dataset predictions
example_BA <- (example_recall+example_Specificity)/2


# F1 Score for example dataset predictions
example_F1 <- F1_Score(example_label, example_predict, positive = "1")


# MCC for example dataset predictions
example_mcc_den <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
example_mcc     <- ifelse(example_mcc_den == 0,
                          NA, (TP * TN - FP * FN) / example_mcc_den)


# Balanced MCC for example predictions
example_BA_mcc <- (example_recall + example_Specificity -1) / 
  sqrt(1-(example_recall - example_Specificity)^2)
# Each iterations and for each different datasets predictions, 
# we will update the value of TP, TN, FP and FN.






# CASF-2016 dataset Result Analysis
# Read the CASF_predictions.csv 
# 199 predictions in total
CASF_predictions <- read.csv("CASF_predictions.csv")



# PCC calculation for CASF predictions 
CASF_pcc <- cor(CASF_predictions$pK, 
                   CASF_predictions$final_pred_pKd, method = "pearson")


# 95% Confidence interval for PCC on CASF predictions 
# Same logic is applied for CASF predictions 
set.seed(2025)


stat_pcc <- function(data, idx) {
  d <- data[idx, ]
  cor(d$y, d$yhat, method = "pearson")
}


df <- data.frame(y = CASF_predictions$pK,
                 yhat = CASF_predictions$final_pred_pKd)

# bootstrap
CASF_PCC_CI <- boot(data = df, statistic = stat_pcc, R = 10000)


# 95% BCa Confidence interval 
boot.ci(CASF_PCC_CI , type = "bca")


# plots for PCC for CASF predictions 
# Same logic is also used for CASF predictions 

# Calculate the range first 
lims_CASF <- range(c(CASF_predictions$pK, 
                CASF_predictions$final_pred_pKd))


CASF_figure <- ggplot(CASF_predictions, 
                              aes(x = pK, y = final_pred_pKd)) +
  geom_point(color = "darkgreen", size = 1.5, alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  coord_equal() +
  xlim(lims_CASF) + ylim(lims_CASF) +
  labs(
    title = "Predicted vs Experimental pK",
    x = "Experimental PK",
    y = "Predicted PK"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, hjust = 0),
    axis.title = element_text(size = 12),
    axis.text  = element_text(size = 12)
  )


CASF_pcc_figure <- ggMarginal(
  CASF_figure, type = "density", margins = "both",
  fill = "darkgreen", alpha = 0.25
)


CASF_pcc_figure



# RMSE for CASF predictions 
CASF_RMSE <- sqrt(mean((CASF_predictions$pK - 
                                  CASF_predictions$final_pred_pKd)^2))


# Calculating the 95% BCa Confidence interval for RMSE for CASF predictions 
set.seed(2025)


stat_rmse <- function(d, i) sqrt(mean((d$y[i] - d$yhat[i])^2))

d <- data.frame(
  y    = CASF_predictions$pK,
  yhat = CASF_predictions$final_pred_pKd
)


CASF_RMSE_CI  <- boot(d, stat_rmse, R = 10000)          
         
                                 
boot.ci(CASF_RMSE_CI, type = "bca")     


# tau_a for CASF predictions 
CASF_tau_a <- kendall_tau_a(CASF_predictions$pK, 
                                   CASF_predictions$final_pred_pKd)


# 95% BCa Confidence interval for tau_a on CASF predictions 
set.seed(2025)


stat_tau_a <- function(d, i) kendall_tau_a(d$y[i], d$yhat[i])


d <- data.frame(
  y    = CASF_predictions$pK,
  yhat = CASF_predictions$final_pred_pKd
)


CASF_tau_a_CI  <- boot(d, stat_tau_a, R = 10000)
         
                   
boot.ci(CASF_tau_a_CI , type = "bca") 


# Kendall’s tau_b correlation coefficient for CASF predictions 
CASF_tau_b <- Kendall(CASF_predictions$pK,
                            CASF_predictions$final_pred_pKd)


# 95% BCa Confidence interval for tau_b for CASF predictions 
set.seed(2025)


stat_tau_b <- function(d, i) Kendall(d$y[i], d$yhat[i])$tau


d <- data.frame(
  y    = CASF_predictions$pK,
  yhat = CASF_predictions$final_pred_pKd
)


CASF_tau_b_CI  <- boot(d, stat_tau_b, R = 10000)          # bootstrap


boot.ci(CASF_tau_b_CI , type = "bca")           


# We assess the classification performance on CASF predictions 
# Classification task 
# CASF_label indicates the true binary label for pK values 
CASF_label <- as.integer(CASF_predictions$pK >= 4)


# ROC-AUC for CASF predictions
# Method 1:
CASF_roc <- roc(CASF_label, 
                CASF_predictions$final_bind_prob)


CASF_auc <- auc(CASF_roc)


plot(CASF_roc, col = "darkgreen", lwd = 2,
     main = paste("ROC Curve (AUC =", round(auc(CASF_roc), 3), ")"))


# Method 2 
df <- data.frame(
  truth = factor(CASF_label, levels = c(1,0)),         
  .pred_1 = CASF_predictions$final_bind_prob             
)


# AUC
roc_auc(df, truth = truth, .pred_1)


# ROC Curve
roc_curve(df, truth = truth, .pred_1) %>%
  autoplot(colour = "red") +
  ggtitle("ROC Curve for CASF predictions")


# We perform the same process for CASF predictions 
# We set threshold to be 0.5 temporarily, 
# and each time it can be changed manually
CASF_predict <- as.integer(CASF_predictions$final_bind_prob >= 0.5)


# Create labels 
TP <- sum(CASF_predict == 1 & CASF_label == 1)
FP <- sum(CASF_predict == 1 & CASF_label == 0)
TN <- sum(CASF_predict == 0 & CASF_label == 0)
FN <- sum(CASF_predict == 0 & CASF_label == 1)


# Precision for CASF predictions 
CASF_precision <- ifelse(TP+FP == 0, NA, TP/(TP+FP))


# Recall for CASF predictions 
CASF_recall <- ifelse(TP+FN == 0, NA, TP/(TP+FN))


# Specificity for CASF predictions 
CASF_Specificity <- ifelse((TN + FP) == 0, NA, TN / (TN + FP))


# Balanced Accuracy for CASF predictions 
CASF_BA <- (CASF_recall + CASF_Specificity ) / 2
 

# F1 Score for CASF predictions 
CASF_F1 <- F1_Score(CASF_label, CASF_predict, positive = "1")


# MCC  for CASF predictions 
CASF_mcc_den <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
CASF_mcc     <- ifelse(CASF_mcc_den == 0, 
                       NA, (TP * TN - FP * FN) / CASF_mcc_den)


# Balanced MCC for CASF predictions
CASF_BA_mcc <- (CASF_recall + CASF_Specificity - 1) / 
  sqrt(1-(CASF_recall - CASF_Specificity)^2)




# OOD dataset Result Analysis
# Read the OOD_predictions.csv 
# 257 predictions in total
OOD_predictions <- read.csv("OOD_predictions.csv")


# PCC for OOD predictions 
OOD_pcc <- cor(OOD_predictions$pK, 
                        OOD_predictions$final_pred_pKd, 
                       method = "pearson")


# 95% BCa Confidence interval for PCC on OOD predictions 
set.seed(2025)


stat_pcc <- function(data, idx) {
  d <- data[idx, ]
  cor(d$y, d$yhat, method = "pearson")
}


df <- data.frame(y = OOD_predictions$pK,
                 yhat = OOD_predictions$final_pred_pKd)


OOD_PCC_CI <- boot(data = df, statistic = stat_pcc, R = 10000)


boot.ci(OOD_PCC_CI , type = "bca")


# plots for PCC on OOD predictions 
# Calculating the range for OOD predictions 
lims_OOD <- range(c(OOD_predictions$pK, 
                     OOD_predictions$final_pred_pKd))


OOD_figure <- ggplot(OOD_predictions, 
                              aes(x = pK, y = final_pred_pKd)) +
  geom_point(color = "darkgreen", size = 1.5, alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  coord_equal() +
  xlim(lims_OOD) + ylim(lims_OOD) +
  labs(
    title = "Predicted vs Experimental pK",
    x = "Experimental PK",
    y = "Predicted PK"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, hjust = 0),
    axis.title = element_text(size = 12),
    axis.text  = element_text(size = 12)
  )


OOD_pcc_figure <- ggMarginal(
  OOD_figure, type = "density", margins = "both",
  fill = "darkgreen", alpha = 0.25
)


OOD_pcc_figure


# RMSE for OOD predictions 
OOD_RMSE <- sqrt(mean((OOD_predictions$pK - 
                                OOD_predictions$final_pred_pKd)^2))


# 95% BCa Confidence interval for RMSE for OOD predictions 
set.seed(2025)


stat_rmse <- function(d, i) sqrt(mean((d$y[i] - d$yhat[i])^2))


d <- data.frame(
  y    = OOD_predictions$pK,
  yhat = OOD_predictions$final_pred_pKd
)


OOD_RMSE_CI  <- boot(d, stat_rmse, R = 10000)          
    
                              
boot.ci(OOD_RMSE_CI , type = "bca")  


# tau_a for OOD predictions 
OOD_tau_a <- kendall_tau_a(OOD_predictions$pK, 
                                   OOD_predictions$final_pred_pKd)


# 95% Confidence interval for tau_a for OOD predictions 
stat_tau_a <- function(d, i) kendall_tau_a(d$y[i], d$yhat[i])


set.seed(2025)
d <- data.frame(
  y    = OOD_predictions$pK,
  yhat = OOD_predictions$final_pred_pKd
)


OOD_tau_a_CI  <- boot(d, stat_tau_a, R = 10000)
              
       
boot.ci(OOD_tau_a_CI, type = "bca") 


# Kendall’s tau_b correlation coefficient
OOD_tau_b <- Kendall(OOD_predictions$pK, 
                             OOD_predictions$final_pred_pKd)

# 95% BCa Confidence interval for tau_b for OOD predictions 
set.seed(2025)


stat_tau_b <- function(d, i) Kendall(d$y[i], d$yhat[i])$tau


d <- data.frame(
  y    = OOD_predictions$pK,
  yhat = OOD_predictions$final_pred_pKd
)


OOD_tau_b_CI  <- boot(d, stat_tau_b, R = 10000)          # bootstrap


boot.ci(OOD_tau_b_CI, type = "bca")           


# We also assess the classification performance
# on the OOD predictions 
# Classification task
# OOD_label indicates the true pK label of the OOD dataset 
OOD_label <- as.integer(OOD_predictions$pK >= 4)


# ROC-AUC for OOD predictions
# Method 1:
OOD_roc <- roc(OOD_label, 
               OOD_predictions$final_bind_prob)


OOD_auc <- auc(OOD_roc)


plot(OOD_roc, col = "darkgreen", lwd = 2,
     main = paste("ROC Curve (AUC =", round(auc(OOD_roc), 3), ")"))


# Method 2:
df <- data.frame(
  truth = factor(OOD_label, levels = c(1,0)),         
  .pred_1 = OOD_predictions$final_bind_prob            
)


# AUC
roc_auc(df, truth = truth, .pred_1)


# ROC Curve 
roc_curve(df, truth = truth, .pred_1) %>%
  autoplot(colour = "red") +
  ggtitle("ROC Curve for OOD predictions")


# We set threshold to be 0.5 temporarily, 
# and each time it can be changed manually afterwards


# OOD_predict indicates the predicted binary label 
# if we set the threshold to be 0.5
OOD_predict <- as.integer(OOD_predictions$final_bind_prob >= 0.5)


# Create labels for OOD predictions 
TP <- sum(OOD_predict == 1 & OOD_label == 1)
FP <- sum(OOD_predict == 1 & OOD_label == 0)
TN <- sum(OOD_predict == 0 & OOD_label == 0)
FN <- sum(OOD_predict == 0 & OOD_label == 1)



# Precision for OOD predictions 
OOD_precision <- ifelse(TP+FP == 0, NA, TP/(TP+FP))


# Recall  for OOD predictions 
OOD_recall <- ifelse(TP+FN == 0, NA, TP/(TP+FN))


# Specificity for OOD predictions 
OOD_Specificity <- ifelse((TN + FP) == 0, NA, TN / (TN + FP))


# Balanced Accuracy for OOD predictions 
OOD_BA <- (OOD_Specificity + OOD_recall)/2


# F1 Score for OOD predictions 
OOD_F1 <- F1_Score(OOD_label, OOD_predict, positive = "1")


# MCC for OOD predictions 
OOD_mcc_den <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
OOD_mcc     <- ifelse(OOD_mcc_den == 0, NA, (TP * TN - FP * FN) / OOD_mcc_den)


# Balanced MCC for OOD predictions
OOD_BA_mcc <- (OOD_recall + OOD_Specificity - 1) / 
  sqrt(1-(OOD_recall - OOD_Specificity)^2)


# 0 Ligand Bias dataset Result Analysis
# Read the LB_predictions.csv 
# 329 predictions in total
LB_predictions <- read.csv("LB_predictions.csv")


# PCC for 0LB predictions 
LB_pcc <- cor(LB_predictions$pk, 
              LB_predictions$final_pred_pKd, method = "pearson")


# 95% BCa Confidence inerval for PCC on 0LB predictions 
# Same logic was used in the example predictions 
set.seed(2025)


stat_pcc <- function(data, idx) {
  d <- data[idx, ]
  cor(d$y, d$yhat, method = "pearson")
}


df <- data.frame(y = LB_predictions$pk,
                 yhat = LB_predictions$final_pred_pKd)


LB_PCC_CI <- boot(data = df, statistic = stat_pcc, R = 10000)


boot.ci(LB_PCC_CI, type = "bca")


# plots for PCC on 0LB predictions 
# Calculate the range for pK in 0LB predictions 
lims_LB <- range(c(LB_predictions$pk, 
                    LB_predictions$final_pred_pKd))


LB_figure <- ggplot(LB_predictions, 
                              aes(x = pk, y = final_pred_pKd)) +
  geom_point(color = "darkgreen", size = 1.5, alpha = 0.5) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
  coord_equal() +
  xlim(lims_LB) + ylim(lims_LB) +
  labs(
    title = "Predicted vs Experimental pK",
    x = "Experimental PK",
    y = "Predicted PK"
  ) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(size = 14, hjust = 0),
    axis.title = element_text(size = 12),
    axis.text  = element_text(size = 12)
  )


LB_pcc_figure <- ggMarginal(
 LB_figure, type = "density", margins = "both",
  fill = "darkgreen", alpha = 0.25
)


LB_pcc_figure


# RMSE for 0LB predictions 
LB_RMSE <- sqrt(mean((LB_predictions$pk -
                                LB_predictions$final_pred_pKd)^2))


# 95% BCa Confidence interval for RMSE for 0LB predictions 
set.seed(2025)

stat_rmse <- function(d, i) sqrt(mean((d$y[i] - d$yhat[i])^2))


d <- data.frame(
  y    = LB_predictions$pk,
  yhat = LB_predictions$final_pred_pKd
)


LB_RMSE_CI  <- boot(d, stat_rmse, R = 10000)          # bootstrap
                                        

boot.ci(LB_RMSE_CI , type = "bca")           


# tau_a for 0LB predictions 
LB_tau_a <- kendall_tau_a(LB_predictions$pk, 
                           LB_predictions$final_pred_pKd)


# 95% BCa Confidence interval for tau_a for 0LB predictions 
stat_tau_a <- function(d, i) kendall_tau_a(d$y[i], d$yhat[i])


d <- data.frame(
  y    = LB_predictions$pk,
  yhat = LB_predictions$final_pred_pKd
)


LB_tau_a_CI  <- boot(d, stat_tau_a, R = 10000)
                              

boot.ci(LB_tau_a_CI, type = "bca") 


# Kendall’s tau_b correlation coefficient for 0LB predictions 
LB_tau_b <- Kendall(LB_predictions$pk,
                    LB_predictions$final_pred_pKd)


# 95% BCa Confidence interval for tau_b for 0LB predictions 
set.seed(2025)


stat_tau_b <- function(d, i) Kendall(d$y[i], d$yhat[i])$tau


d <- data.frame(
  y    = LB_predictions$pk,
  yhat = LB_predictions$final_pred_pKd
)


LB_tau_b_CI  <- boot(d, stat_tau_b, R = 10000)          # bootstrap


boot.ci(LB_tau_b_CI, type = "bca")  


# Classification task 
# We start to assess the classification performance on 0LB predictions
# LB_label indicates the true binary label for 0LB dataset 
LB_label <- as.integer(LB_predictions$pk >= 4)


# ROC-AUC for 0LB predictions 
# Method 1:
LB_roc <- roc(LB_label, LB_predictions$final_bind_prob)


LB_auc <- auc(LB_roc)


plot(LB_roc, col = "darkgreen", lwd = 2,
     main = paste("ROC Curve (AUC =", round(auc(LB_roc), 3), ")"))


# Method 2:
df <- data.frame(
  truth = factor(LB_label, levels = c(1,0)),        
  .pred_1 = LB_predictions$final_bind_prob             
)


# AUC
roc_auc(df, truth = truth, .pred_1)


# ROC Curve on 0LB predictions 
roc_curve(df, truth = truth, .pred_1) %>%
  autoplot(colour = "red") +
  ggtitle("ROC Curve for 0LB predictions ")


# We set threshold to be 0.5 temporarily, 
# and each time it can be changed manually
# If the predicted probability >= 0.5, we consider that observations
# as positive samples.
LB_predict <- as.integer(LB_predictions$final_bind_prob >= 0.5)


# Create labels for 0LB predictions 
TP <- sum(LB_predict == 1 & LB_label == 1)
FP <- sum(LB_predict == 1 & LB_label == 0)
TN <- sum(LB_predict == 0 & LB_label == 0)
FN <- sum(LB_predict == 0 & LB_label == 1)


# Precision for 0LB predictions 
LB_precision <- ifelse(TP+FP == 0, NA, TP/(TP+FP))


# Recall for 0LB predictions
LB_recall <- ifelse(TP+FN == 0, NA, TP/(TP+FN))


# Specificity for 0LB predictions
LB_Specificity <- ifelse((TN + FP) == 0, NA, TN / (TN + FP))


# Balanced Accuracy for 0LB predictions
LB_BA <- (LB_recall + LB_Specificity)/2


# F1 Score for 0LB predictions
LB_F1 <- F1_Score(LB_label, LB_predict, positive = "1")


# MCC for 0LB predictions
LB_mcc_den <- sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
LB_mcc  <- ifelse(LB_mcc_den == 0, NA, (TP * TN - FP * FN) / LB_mcc_den)


# Balanced MCC for 0LB predictions
LB_BA_mcc <- (LB_recall + LB_Specificity - 1) / 
  sqrt(1-(LB_recall - LB_Specificity)^2)


# PCC plots for CASF-2016, OOD Test and 0 Ligand Bias

# -------- CASF-2016 --------
CASF_labeled <- CASF_figure +
  labs(title = NULL) +   
  annotate("text", x = -Inf, y = Inf,
           hjust = -0.05, vjust = 1.1,
           label = sprintf("PCC = %.2f\nRMSE = %.2f\nKendall τ_a = %.2f",
                           CASF_pcc, CASF_RMSE, as.numeric(CASF_tau_a)),
           size = 3.8)  
CASF_plot <- ggMarginal(CASF_labeled, type = "density", margins = "both",
                        fill = "darkgreen", alpha = 0.25)

# -------- OOD Test --------
OOD_labeled <- OOD_figure +
  labs(title = NULL) +
  annotate("text", x = -Inf, y = Inf,
           hjust = -0.05, vjust = 1.1,
           label = sprintf("PCC = %.2f\nRMSE = %.2f\nKendall τ_a = %.2f",
                           OOD_pcc, OOD_RMSE, as.numeric(OOD_tau_a)),
           size = 3.8)  
OOD_plot <- ggMarginal(OOD_labeled, type = "density", margins = "both",
                       fill = "darkgreen", alpha = 0.25)

# -------- 0 Ligand Bias --------
LB_labeled <- LB_figure +
  labs(title = NULL) +
  annotate("text", x = -Inf, y = Inf,
           hjust = -0.05, vjust = 1.1,
           label = sprintf("PCC = %.2f\nRMSE = %.2f\nKendall τ_a = %.2f",
                           LB_pcc, LB_RMSE, as.numeric(LB_tau_a)),
           size = 3.8)  
LB_plot <- ggMarginal(LB_labeled, type = "density", margins = "both",
                      fill = "darkgreen", alpha = 0.25)


# --------  3×1 plots --------
final_plot <- plot_grid(
  CASF_plot, OOD_plot, LB_plot,
  ncol = 1,
  labels = c("A  CASF-2016", "B  OOD Test", "C  0 Ligand Bias"),
  label_size = 16, label_fontface = "bold"
)


# Save the figure 
ggsave("PCC_plots.png", final_plot,
       width = 7, height = 12, dpi = 300, bg = "white")







# ROC-AUC plots for CASF-2016, OOD Test and 0 Ligand Bias
# CASF
df_CASF <- data.frame(
  truth = factor(CASF_label, levels = c(1,0)),
  .pred_1 = CASF_predictions$final_bind_prob
)


CASF_auc <- roc_auc(df_CASF, truth = truth, .pred_1)$.estimate


CASF_roc_plot <- roc_curve(df_CASF, truth = truth, .pred_1) %>%
  autoplot(colour = "darkgreen", size = 1) +
  geom_abline(linetype = "dashed", colour = "grey50") +
  ggtitle(sprintf("CASF-2016 ROC Curve\n(AUC = %.3f)", CASF_auc)) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(size = 12, hjust = 0.5),
    axis.title = element_text(size = 10),
    axis.text  = element_text(size = 9)
  )


# -------- OOD ROC --------
df_OOD <- data.frame(
  truth = factor(OOD_label, levels = c(1,0)),
  .pred_1 = OOD_predictions$final_bind_prob
)


OOD_auc <- roc_auc(df_OOD, truth = truth, .pred_1)$.estimate


OOD_roc_plot <- roc_curve(df_OOD, truth = truth, .pred_1) %>%
  autoplot(colour = "darkgreen", size = 1) +
  geom_abline(linetype = "dashed", colour = "grey50") +
  ggtitle(sprintf("OOD Test ROC Curve\n(AUC = %.3f)", OOD_auc)) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(size = 12, hjust = 0.5),
    axis.title = element_text(size = 10),
    axis.text  = element_text(size = 9)
  )


# -------- 0LB ROC --------
df_LB <- data.frame(
  truth = factor(LB_label, levels = c(1,0)),
  .pred_1 = LB_predictions$final_bind_prob
)


LB_auc <- roc_auc(df_LB, truth = truth, .pred_1)$.estimate


LB_roc_plot <- roc_curve(df_LB, truth = truth, .pred_1) %>%
  autoplot(colour = "darkgreen", size = 1) +
  geom_abline(linetype = "dashed", colour = "grey50") +
  ggtitle(sprintf("0 Ligand Bias ROC Curve\n(AUC = %.3f)", LB_auc)) +
  theme_bw(base_size = 12) +
  theme(
    plot.title = element_text(size = 12, hjust = 0.5),
    axis.title = element_text(size = 10),
    axis.text  = element_text(size = 9)
  )


# --------  3×1 --------
final_roc_plot <- plot_grid(
  CASF_roc_plot, OOD_roc_plot, LB_roc_plot,
  ncol = 1,
  labels = c("A  CASF-2016", "B  OOD Test", "C  0 Ligand Bias"),
  label_size = 14, label_fontface = "bold"
)


# -------- Save --------
ggsave("ROC_Curve for benchmarks.png", final_roc_plot,
       width = 7, height = 12, dpi = 300, bg = "white")


# Plots for Confidence Interval for each quantities 
# under three benchmarks for comparison


pcc_df <- tibble(
  dataset  = c("CASF-2016", "OOD Test", "0 Ligand Bias"),
  estimate = c(0.852, 0.780, 0.374),
  low      = c(0.786, 0.714, 0.243),
  high     = c(0.890, 0.827, 0.484)
)


rmse_df <- tibble(
  dataset  = c("CASF-2016", "OOD Test", "0 Ligand Bias"),
  estimate = c(1.242, 1.328, 1.597),
  low      = c(1.115, 1.216, 1.470),
  high     = c(1.415, 1.499, 1.779)
)


kendall_df <- tibble(
  dataset  = rep(c("CASF-2016", "OOD Test", "0 Ligand Bias"), each = 2),
  type     = rep(c("tau_a", "tau_b"), 3),
  estimate = c(0.672, 0.672, 0.566, 0.567, 0.244, 0.245),
  low      = c(0.613, 0.609, 0.506, 0.505, 0.166, 0.164),
  high     = c(0.722, 0.721, 0.621, 0.620, 0.318, 0.316)
)


ds_levels <- c("CASF-2016", "OOD Test", "0 Ligand Bias")
ds_cols   <- c("CASF-2016"="#1b9e77", "OOD Test"="#2c7fb8", "0 Ligand Bias"="#fdb863")
pcc_df$dataset     <- factor(pcc_df$dataset, levels = ds_levels)
rmse_df$dataset    <- factor(rmse_df$dataset, levels = ds_levels)
kendall_df$dataset <- factor(kendall_df$dataset, levels = ds_levels)


er    <- function(x) grDevices::extendrange(x, f = 0.04)
brks2 <- function(lims) pretty(lims, n = 4)
fmt2  <- label_number(accuracy = 0.01, trim = TRUE)


ylim_pcc     <- er(c(pcc_df$low,     pcc_df$high))
ylim_rmse    <- er(c(rmse_df$low,    rmse_df$high))
ylim_kendall <- er(c(kendall_df$low, kendall_df$high))


tight_margin <- margin(1.5, 1.5, 1.5, 1.5)
base_theme <- theme_bw(base_size = 12) +
  theme(
    plot.margin  = tight_margin,
    aspect.ratio = 1,
    axis.text.x  = element_text(angle = 0, face = "bold.italic", size = 9,
                                vjust = 1, hjust = 0.5)
  )


x_tight <- scale_x_discrete(expand = expansion(mult = c(0.01, 0.01)))


legend_left  <- theme(
  legend.position   = c(0.18, 0.24),
  legend.background = element_rect(fill = alpha("white", 0), colour = NA),
  legend.key.height = unit(9, "pt"),
  legend.key.width  = unit(14, "pt"),
  legend.text       = element_text(size = 9),
  legend.title      = element_text(size = 9)
)


legend_right <- theme(
  legend.position   = c(0.82, 0.24),
  legend.background = element_rect(fill = alpha("white", 0), colour = NA),
  legend.key.height = unit(9, "pt"),
  legend.key.width  = unit(14, "pt"),
  legend.text       = element_text(size = 9),
  legend.title      = element_text(size = 9)
)


pt_size         <- 2.6
legend_pt_size  <- 1.3
legend_override <- list(size = legend_pt_size)


linetype_guide_big <- guide_legend(
  title     = "Kendall’s τ",
  order     = 1,
  keywidth  = unit(56, "pt"),   
  keyheight = unit(12, "pt"),
  override.aes = list(
    colour    = "black",
    size      = 0.9,            
    linewidth = 0.9
  )
)


# Plot the figure 
p1 <- ggplot(pcc_df, aes(dataset, estimate, colour = dataset)) +
  geom_errorbar(aes(ymin = low, ymax = high), width = 0.16, size = 0.9) +
  geom_point(size = pt_size) +
  x_tight +
  scale_color_manual(values = ds_cols, name = NULL) +
  guides(colour = guide_legend(override.aes = legend_override)) +
  scale_y_continuous(limits = ylim_pcc, breaks = brks2(ylim_pcc),
                     labels = fmt2, expand = expansion(mult = c(0.005, 0.005))) +
  labs(x = NULL, y = "PCC", title = "PCC (with 95% BCa Bootstrap CI)") +
  base_theme + legend_left +
  theme(plot.title = element_text(size = 11, hjust = 0.5))


p2 <- ggplot(rmse_df, aes(dataset, estimate, colour = dataset)) +
  geom_errorbar(aes(ymin = low, ymax = high), width = 0.16, size = 0.9) +
  geom_point(size = pt_size) +
  x_tight +
  scale_color_manual(values = ds_cols, name = NULL) +
  guides(colour = guide_legend(override.aes = legend_override)) +
  scale_y_continuous(limits = ylim_rmse, breaks = brks2(ylim_rmse),
                     labels = fmt2, expand = expansion(mult = c(0.005, 0.005))) +
  labs(x = NULL, y = "RMSE", title = "RMSE (with 95% BCa Bootstrap CI)") +
  base_theme + legend_right +
  theme(plot.title = element_text(size = 11, hjust = 0.5))


p3 <- ggplot(kendall_df, aes(dataset, estimate, colour = dataset, linetype = type)) +
  geom_errorbar(aes(ymin = low, ymax = high),
                width = 0.16, size = 0.9, position = position_dodge(0.5)) +
  geom_point(size = pt_size, position = position_dodge(0.5)) +
  x_tight +
  scale_color_manual(values = ds_cols, name = NULL) +
  scale_linetype_manual(
    values = c("solid", "longdash"),  
    name   = "Kendall’s τ",
    labels = c(expression("\u03C4"[a]), expression("\u03C4"[b]))
  ) +
  guides(
    colour   = guide_legend(override.aes = legend_override),
    linetype = linetype_guide_big
  ) +
  scale_y_continuous(limits = ylim_kendall, breaks = brks2(ylim_kendall),
                     labels = fmt2, expand = expansion(mult = c(0.005, 0.005))) +
  labs(x = NULL, y = "Kendall τ", title = "Kendall τ (with 95% BCa Bootstrap CI)") +
  base_theme + legend_left +
  theme(plot.title = element_text(size = 11, hjust = 0.5))


# =====================SAVE =====================
final_plot <- p1 / p2 / p3
ggsave("Results_for_three_benchmarks.png", final_plot,
       width = 5, height = 12, dpi = 300, bg = "white")
