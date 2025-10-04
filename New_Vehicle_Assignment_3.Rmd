---
title: "Assignment 3"
author: "Elizabeth Crawley"
date: "`r Sys.Date()`"
output: html_document
---

```{r}
# Load necessary libraries
library(class)
library(caret)
library(readr)
library(ggplot2)
library(lattice)

# Read the data
Vehicles_data <- read.csv("C:/Users/ecrawle1/OneDrive - Kent State University/Desktop/Machine Learning/Vehicles_Sales.csv")

```
```{r}
# Drop YEAR_ID and PRODUCTLINE variables
Vehicles_data <- Vehicles_data[,-c(9, 10)]
```

```{r}
# DEALSIZE needs to be converted to factor
Vehicles_data$DEALSIZE <- as.factor(Vehicles_data$DEALSIZE)
# Convert DEALSIZE to dummy variables
groups <- dummyVars(~., data = Vehicles_data)
# Create Dummy variable.names
Vehicles_data_dummy <- as.data.frame(predict(groups, Vehicles_data))
```
```{r}
# Partition the data into Training (50%), Validation (30%), and Testing (20%) sets. 

# Set seed for reproducibility
set.seed(123)

# Get indices for the 50% training data
train_index <- createDataPartition(Vehicles_data_dummy$STATUS, p = 0.5, list = FALSE)

# Split the data into training and the remaining 50%
train_data <- Vehicles_data_dummy[train_index, ]
remaining_data <- Vehicles_data_dummy[-train_index, ]

# Split the remaining data into validation (30%) and test (20%)
# 30% out of the remaining 50% corresponds to 0.6, and 20% corresponds to 0.4
val_index <- createDataPartition(remaining_data$STATUS, p = 0.6, list = FALSE)

validation_data <- remaining_data[val_index, ]
test_data <- remaining_data[-val_index, ]

# Check the sizes of the partitions
cat("Training set size:", nrow(train_data), "\n")
cat("Validation set size:", nrow(validation_data), "\n")
cat("Testing set size:", nrow(test_data), "\n")


```
```{r}
# Load necessary package
library(caret)

# Status is the target variable and it's the 6th column
target_col <- 6

# Exclude the target column for normalization
train.norm.df <- train_data[,-target_col]
valid.norm.df <- validation_data[,-target_col]
test.norm.df <- test_data[,-target_col]

# Normalize the datasets using preProcess
norm.values <- preProcess(train_data[, -target_col], method = c("center", "scale"))

# Apply normalization to training, validation, and test datasets
train.norm.df <- predict(norm.values, train_data[, -target_col])
valid.norm.df <- predict(norm.values, validation_data[, -target_col])
test.norm.df <- predict(norm.values, test_data[, -target_col])


```
```{r}
# Check the dimensions of the normalized datasets
cat("Normalized training set size:", nrow(train.norm.df), "\n")
cat("Normalized validation set size:", nrow(valid.norm.df), "\n")
cat("Normalized testing set size:", nrow(test.norm.df), "\n")

```
```{r}
library(class)
new_vehicle <- data.frame(
  ORDERNUMBER = 10322,
  QUANTITYORDERED = 50,
  PRICEEACH = 100,
  ORDERLINENUMBER = 6,
  SALES = 12536.5,
  QTR_ID = 4,
  MONTH_ID = 11,
  MSRP = 127,
  DEALSIZE.Large = 1,
  DEALSIZE.Small = 0,
  DEALSIZE.Medium = 0
)
new_vehicle.norm <- predict(norm.values, new_vehicle)

k_value <- 1
knn_result <- knn(
  train = train.norm.df, 
  test = new_vehicle.norm, 
  cl = train.norm.df$QUANTITYORDERED,
  k = k_value
)

knn_result

```
```{r}
# The result (knn_result) is "1.184..." representing 'Shipped', indicating how the k-NN algorithm classifies the new vehicle based on the training data.
```


