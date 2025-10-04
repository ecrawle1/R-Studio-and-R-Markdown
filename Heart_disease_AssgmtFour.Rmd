---
title: "Heart Disease_AssmtFour"
author: "Elizabeth Crawley"
date: "`r Sys.Date()`"
output: pdf_document
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## R Markdown

# Load necessary libraries
```{r}
library(e1071)
library(caret)
library(dplyr)
```
# Read the heart disease dataset
```{r}
Heart_disease <- read.csv("C:/Users/ecrawle1/OneDrive - Kent State University/Desktop/Machine Learning/Heart_disease.csv")
```
#Create a dummy variable called "Target" that will be "Yes" if the "MAX_HeartRate" exceeds170 and "No" if it is 170 or below. Additionally, you will create another dummy variable named "BP_New," which will be "Yes" if the blood pressure is above 120 and "No" if it is 120 or below.
```{r}
Heart_disease_data <- Heart_disease %>%
  mutate(
    Target = ifelse(MAX_HeartRate > 170, "Yes", "No"),
    BP_New = ifelse(Blood_Pressure > 120, "Yes", "No")
  )
```
#Q1) Prediction Based on Initial Information: [30 Points] If a person with chest pain presents without any further information, what prediction should we make regarding heart disease?

# The prediction is "Yes", heart disease. If the only available information is the presence of chest pain, with no further details about the person’s heart rate, blood pressure, or other, we need to make a prediction about the likelihood of heart disease based solely on chest pain. Chest pain is often a significant indicator of potential heart issues. However, not all types of chest pain are associated with heart disease.
```{r}
# Display counts of the Target variable
Target_table <- table(Heart_disease_data$Target)
print(Target_table)
```
# Q2) Analysis of the First 30 Records: [3*20=60 Points] Select the first 30 records in the dataset and focus on the "Target" response variable and the two predictors: blood pressure and chest pain type. Create a pivot table that examines heart disease as a function of these two predictors for these 30 records, incorporating all three variables as rows and columns.

# Select the first 30 records and focus on the relevant variables
```{r}
Heart_disease30 <- Heart_disease_data[1:30, c("Target", "BP_New", "chest_pain_type")]

Object1 <- ftable(Heart_disease30)
print(Object1)
 
Object2 <- ftable(Heart_disease30[, -1])
print(Object2)
```
# a. Compute Bayes Conditional Probabilities: Calculate the exact Bayes conditional probabilities of the "Target (Target=yes)" variable given the four possible combinations of the predictors.
```{r}
# Probability calculations
P1 = Object1[3, 1] / Object2[1, 1]  # Target=yes, BP_New=No & chest_pain_type=0
P2 = Object1[4, 1] / Object2[2, 1]  # Target=No, BP_New=Yes, chest_pain_type=0
P3 = Object1[3, 2] / Object2[1, 2]  # Target=No, BP_New=No, chest_pain_type=1
P4 = Object1[4, 2] / Object2[2, 2]  # Target=Yes, BP_New=Yes, chest_pain_type=1

Probability_Target <- rep(0, 30)
for (i in 1:30) {
  if (Heart_disease30$BP_New[i] == "1") {
    if (Heart_disease30$chest_pain_type[i] == "0") {
      Probability_Target[i] = P1
    } else if (Heart_disease30$chest_pain_type[i] == "1") {
      Probability_Target[i] = P3
    }
  } else {
    if (Heart_disease30$chest_pain_type[i] == "0") {
      Probability_Target[i] = P2
    } else if (Heart_disease30$chest_pain_type[i] == "1") {
      Probability_Target[i] = P4
    }
  }
}
```
# b. Classification of Accidents: Classify the 30 records using these probabilities with a cutoff of 0.5.
```{r}
Heart_disease30$Probability_Target <- Probability_Target
Heart_disease30$Pred_Probability <- ifelse(Heart_disease30$Probability_Target > 0.5, "Yes", "No")

print (Heart_disease30)
```
#c. Manual Calculation of Naive Bayes Probability: Manually compute the naive Bayes conditional probability of an injury given that "BP_New" is "Yes" and "chest_pain_type" is 1.
```{r}
# Calculate P(Target = Yes | BP_New = Yes, chest_pain_type = 1)
P_BPYes_CP1 <- sum(
  Heart_disease30$BP_New == "Yes" &
  Heart_disease30$chest_pain_type == 1 &
  Heart_disease30$Target == "Yes"
) / sum(Heart_disease30$BP_New == "Yes" & Heart_disease30$chest_pain_type == 1)

cat("P(Target = Yes | BP_New = Yes, chest_pain_type = 1):", P_BPYes_CP1, "\n")
```
# Q3) Full Dataset Analysis: [50 Points] Now, use the complete dataset. Partition the data into training (60%) and validation (40%) sets. Run the Naive Bayes classifier on the entire training set using the relevant predictors, with "Target" as the response variable. Note that all predictors are categorical. Present the confusion matrix
```{r}
# Set seed for reproducibility
set.seed(1)

# Check structure of the dataset and column names
str(Heart_disease)
colnames(Heart_disease)

# Handle missing values, if any (optional)
Heart_disease <- na.omit(Heart_disease)

# Ensure 'MAX_HeartRate' and 'Blood_Pressure' exist and are correctly referenced
if (!("MAX_HeartRate" %in% colnames(Heart_disease)) || 
    !("Blood_Pressure" %in% colnames(Heart_disease))) {
  stop("Columns 'MAX_HeartRate' or 'Blood_Pressure' are missing.")
}

# Create the dummy variables Target and BP_New
Heart_disease_data <- Heart_disease %>%
  mutate(
    Target = ifelse(MAX_HeartRate > 170, "Yes", "No"),
    BP_New = ifelse(Blood_Pressure > 120, "Yes", "No")
  )

# Check if the new columns were successfully created
head(Heart_disease_data)

# Convert 'Target' and other predictors to factors
Heart_disease_data$Target <- as.factor(Heart_disease_data$Target)
Heart_disease_data$BP_New <- as.factor(Heart_disease_data$BP_New)
Heart_disease_data$chest_pain_type <- as.factor(Heart_disease_data$chest_pain_type)

# Partition data into training (60%) and validation (40%) sets
set.seed(1)
train.index <- sample(row.names(Heart_disease_data), 0.6 * nrow(Heart_disease_data))
valid.index <- setdiff(row.names(Heart_disease_data), train.index)

train.df <- Heart_disease_data[train.index, ]
valid.df <- Heart_disease_data[valid.index, ]

# Train the Naive Bayes model
nb_model <- naiveBayes(Target ~ chest_pain_type + BP_New, data = train.df)

# Predict on the validation set
valid_pred <- predict(nb_model, newdata = valid.df)

# Generate and display the confusion matrix
conf_matrix <- confusionMatrix(valid_pred, valid.df$Target)
print(conf_matrix)
```
