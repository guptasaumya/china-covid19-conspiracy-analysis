## AMI22T Home Exercise 1, Problem 2
## COVID 19 - A Chinese Conspiracy
## Saumya Gupta, DS


# set path of all-data export import directory
setwd('C:/Users/gupta/OneDrive/Documents/MS-DS/AMI22T/HomeExercises/HomeExercise1/')


# load required packages
library(data.table)
library(dplyr)
library(fastDummies)
library(heplots)
library(energy)
library(klaR)
library(e1071)
library(MASS)
library(caret)
library(class)
library(ModelMetrics)
library(pROC)


# define functions for accuracy and weighted accuracy calculations
# the following code has been taken and modified form this website:
# https://www.datascienceblog.net/post/machine-learning/performance-measures-multi-class-problems/
calculate.accuracy <- function(predictions, ref.labels) {
  return(length(which(predictions == ref.labels)) / length(ref.labels))
}

calculate.w.accuracy <- function(predictions, ref.labels) {
  lvls <- levels(ref.labels)
  
  accs <- lapply(lvls, function(x) {
    idx <- which(ref.labels == x)
    return(calculate.accuracy(predictions[idx], ref.labels[idx]))
  })
  
  acc <- mean(unlist(accs))
  return(acc)
}


# load data and store in a data table
conspiracy <- fread("Conspiracy.txt")
conspiracy <- data.frame(conspiracy)


# set seed
set.seed(9899)


str(conspiracy)

# remove variables not required for this analysis
conspiracy <-
  conspiracy %>% dplyr::select(
    -c(
      "cov_beh_sum",
      "cons_covax",
      "cons_covax_dummy",
      "cons_biowpn_dummy",
      "weight",
      "pid3"
    )
  )

# check missing data points
sum(is.na(conspiracy))

# check number of rows with missing data
sum(apply(conspiracy, 1, anyNA))

# calculate % of rows with missing data
mean(is.na(conspiracy))

# remove missing data
conspiracy <- na.omit(conspiracy)


# factor the nominal qualitative variables
conspiracy$gender[conspiracy$gender == 1] <- "Male"
conspiracy$gender[conspiracy$gender == 2] <- "Female"

conspiracy$pid2[conspiracy$pid2 == 1] <- "DemL"
conspiracy$pid2[conspiracy$pid2 == 2] <- "RepL"

conspiracy$idlg[conspiracy$idlg == 5 |
                  conspiracy$idlg == 7 |
                  conspiracy$idlg == 6] <- "Conservative"

conspiracy$idlg[conspiracy$idlg == 1 |
                  conspiracy$idlg == 2 |
                  conspiracy$idlg == 3 |
                  conspiracy$idlg == 4] <- "Liberal"

discrete_cols <-
  names(conspiracy[, c("gender",
                       "pid2",
                       "idlg",
                       "cons_biowpn")])

conspiracy[, discrete_cols] <-
  lapply(conspiracy[, discrete_cols], factor)


# dummy code factors (necessary for knn())
cons_data <-
  dummy_cols(
    conspiracy,
    select_columns = c("gender", "pid2", "idlg"),
    remove_selected_columns = T
  )

cons_data <-
  cons_data %>% dplyr::select(-c(pid2_DemL, idlg_Liberal, gender_Male))

str(cons_data)

summary(cons_data)


# 2. Prediction model (analysis of Naive Bayes, LDA, QDA, and KNN) ----

# 80/20 split
bound <- floor(nrow(conspiracy) * 0.8)

train <- 1:bound
test <- (bound + 1):nrow(conspiracy)

cons_data <- cons_data[sample(nrow(cons_data)),]

attach(cons_data)

table(cons_biowpn)

str(cons_data)

# create a data frame of all predictors
predictors <- cons_data %>% dplyr::select(
  # ms_news,
  md_agg,
  md_localpap,
  md_localtv,
  md_broadcast,
  md_national,
  md_radio,
  # rw_news,
  md_con,
  md_fox,
  pid2_RepL,
  populism_1,
  populism_2,
  populism_3,
  populism_4,
  populism_5,
  white,
  highered,
  hispanic,
  gender_Female,
  age,
  hhi,
  trust_1,
  idlg_Conservative
)

# checking the assumptions for lda and qda
# check for homogeneous variance-covariance matrix
boxm <- boxM(predictors, cons_data$cons_biowpn)

# check for multivariate normality
for (i in 1:4) {
  result <- mvnorm.etest(
    cons_data %>%
      dplyr::filter(cons_biowpn == i) %>%
      dplyr::select(-c(ms_news,
                       rw_news,
                       cons_biowpn)) %>%
      dplyr::select(where(is.numeric)),
    R = 1000
  )
  print(result)
}


## Variable Selection with lda
sc_obj <- stepclass(
  cons_biowpn ~
    # ms_news +
    md_agg + md_localpap + md_localtv + md_broadcast + md_national + md_radio +
    # rw_news +
    md_con + md_fox +
    pid2_RepL +
    populism_1 +
    populism_2 +
    populism_3 +
    populism_4 +
    populism_5 +
    white +
    highered +
    hispanic +
    gender_Female +
    age +
    hhi +
    trust_1 + idlg_Conservative,
  data = cons_data[train, ],
  method = "lda",
  direction = "backward",
  criterion = "CR"
)

sc_obj


# variables selection with qda
sc_obj <- stepclass(
  cons_biowpn ~
    # ms_news +
    md_agg + md_localpap + md_localtv + md_broadcast + md_national + md_radio +
    # rw_news +
    md_con + md_fox +
    pid2_RepL +
    populism_1 +
    populism_2 +
    populism_3 +
    populism_4 +
    populism_5 +
    white +
    highered +
    hispanic +
    gender_Female +
    age +
    hhi +
    trust_1 + idlg_Conservative,
  data = cons_data[train, ],
  method = "qda",
  direction = "backward",
  criterion = "CR"
)

sc_obj

# for backward selection with lda and qda, we get different results
# hence, we use all one-factor terms (without aliased covariates)


## Naive Bayes ----
# train the Naive Bayes model
nb.model <- naiveBayes(
  cons_biowpn ~
    ms_news +
    # md_agg + md_localpap + md_localtv + md_broadcast + md_national + md_radio +
    rw_news +
    # md_con + md_fox +
    pid2_RepL +
    populism_1 +
    populism_2 +
    populism_3 +
    populism_4 +
    populism_5 +
    white +
    highered +
    hispanic +
    gender_Female +
    age +
    hhi +
    trust_1 + idlg_Conservative,
  data = cons_data[train, ]
)

# get prediction with probabilities output for log loss calculation
nb.out <- predict(nb.model, cons_data[test,], "raw")

# get prediction in classes output for confusion matrix
nb.pred <- predict(nb.model, cons_data[test,])

# confusion matrix
prop.table(table(nb.pred, cons_data$cons_biowpn[test]), margin = 2)


## LDA ----
# train the LdA model
lda.model <- lda(
  cons_biowpn ~
    ms_news +
    # md_agg + md_localpap + md_localtv + md_broadcast + md_national + md_radio +
    rw_news +
    # md_con + md_fox +
    pid2_RepL +
    populism_1 +
    populism_2 +
    populism_3 +
    populism_4 +
    populism_5 +
    white +
    highered +
    hispanic +
    gender_Female +
    age +
    hhi +
    trust_1 + idlg_Conservative
  ,
  data = cons_data[train, ]
)

# get prediction with probabilities output for log loss calculation
lda.out <- predict(lda.model, cons_data[test, ], type = 'prob')

# get prediction in classes output for confusion matrix
lda.pred <-
  predict(lda.model, cons_data[test, ], type = 'prob')$class

# confusion matrix
prop.table(table(lda.pred, cons_data$cons_biowpn[test]), margin = 2)


## QDA ----
# train the QDA model
qda.model <- qda(
  cons_biowpn ~
    ms_news +
    # md_agg + md_localpap + md_localtv + md_broadcast + md_national + md_radio +
    rw_news +
    # md_con + md_fox +
    pid2_RepL +
    populism_1 +
    populism_2 +
    populism_3 +
    populism_4 +
    populism_5 +
    white +
    highered +
    hispanic +
    gender_Female +
    age +
    hhi +
    trust_1 + idlg_Conservative
  ,
  data = cons_data[train, ]
)

# get prediction with probabilities output for log loss calculation
qda.out <- predict(qda.model, cons_data[test, ], type = 'prob')

# get prediction in classes output for confusion matrix
qda.pred <-
  predict(qda.model, cons_data[test, ], type = 'prob')$class

# confusion matrix
prop.table(table(qda.pred, cons_data$cons_biowpn[test]), margin = 2)


## KNN ----
# create controls for train function
trControl <- trainControl(method = "cv",
                          number = 10)

# perform cross-validation for best k value for KNN
fit <- train(
  cons_biowpn ~
    ms_news +
    # md_agg + md_localpap + md_localtv + md_broadcast + md_national + md_radio +
    rw_news +
    # md_con + md_fox +
    pid2_RepL +
    populism_1 +
    populism_2 +
    populism_3 +
    populism_4 +
    populism_5 +
    white +
    highered +
    hispanic +
    gender_Female +
    age +
    hhi +
    trust_1 + idlg_Conservative
  ,
  method     = "knn",
  tuneGrid   = expand.grid(k = 1:20),
  trControl  = trControl,
  metric     = "Accuracy",
  # Accuracy as criteria
  data       = cons_data[train,]
)

# check the best k calculated
fit

# training the KNN model with the best k
knn.model <- knn(
  train = as.matrix(predictors[train,]),
  test = as.matrix(predictors[test,]),
  cl = cons_data$cons_biowpn[train],
  k = 8,
  prob = TRUE
)

# confusion matrix
prop.table(table(knn.model, cons_data$cons_biowpn[test]), margin = 2)


# calculate metrics for evaluation for model predictions
## Accuracy ----
# Naive Bayes
calculate.accuracy(nb.pred, cons_data$cons_biowpn[test])

# LDA
calculate.accuracy(lda.pred, cons_data$cons_biowpn[test])

# QDA
calculate.accuracy(qda.pred, cons_data$cons_biowpn[test])

# KNN
calculate.accuracy(knn.model, cons_data$cons_biowpn[test])


## Weighted Accuracy ----
# Naive Bayes
calculate.w.accuracy(nb.pred, cons_data$cons_biowpn[test])

# LDA
calculate.w.accuracy(lda.pred, cons_data$cons_biowpn[test])

# QDA
calculate.w.accuracy(qda.pred, cons_data$cons_biowpn[test])

# KNN
calculate.w.accuracy(knn.model, cons_data$cons_biowpn[test])


## Log Loss ----
# Naive Bayes
mlogLoss(cons_data$cons_biowpn[test], as.matrix(nb.out))

# LDA
mlogLoss(cons_data$cons_biowpn[test], data.matrix(lda.out$posterior))

# QDA
mlogLoss(cons_data$cons_biowpn[test], data.matrix(qda.out$posterior))

# KNN
mlogLoss(cons_data$cons_biowpn[test], as.matrix(attr(knn.model, "prob")))


## AUC ----
all_models = data.frame(
  true_class = cons_data$cons_biowpn[test],
  nb_model = nb.pred,
  knn_model = knn.model,
  lda_model = lda.pred,
  qda_model = qda.pred
)

all_models[, ] <- lapply(all_models[, ], as.numeric)

# Naive Bayes
multiclass.roc(all_models$true_class,
               all_models$nb_model,
               levels = c(1, 2, 3, 4))

# LDA
multiclass.roc(all_models$true_class,
               all_models$lda_model,
               levels = c(1, 2, 3, 4))

# QDA
multiclass.roc(all_models$true_class,
               all_models$qda_model,
               levels = c(1, 2, 3, 4))

# KNN
multiclass.roc(all_models$true_class,
               all_models$knn_model,
               levels = c(1, 2, 3, 4))


# save the models
# saveRDS(nb.model, "model_3.rds")
# saveRDS(lda.model, "model_4.rds")
# saveRDS(qda.model, "model_5.rds")
# saveRDS(knn.model, "model_6.rds")

# my_model_3 <- readRDS("model_3.rds")
# my_model_4 <- readRDS("model_4.rds")
# my_model_5 <- readRDS("model_5.rds")
# my_model_6 <- readRDS("model_6.rds")
