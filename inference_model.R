## AMI22T Home Exercise 1, Problem 1
## COVID 19 - A Chinese Conspiracy
## Saumya Gupta, DS


# set path of all-data export import directory
setwd('C:/Users/gupta/OneDrive/Documents/MS-DS/AMI22T/HomeExercises/HomeExercise1/')


# load required packages
library(data.table)
library(dplyr)
library(GGally)
library(car)
library(glmulti)
library(oddsratio)
library(ggplot2)


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
      "cons_biowpn",
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

# create variables use combined effects of populist views questions in analysis
conspiracy$populism123 <- rowMeans(conspiracy[, 2:4])
conspiracy$populism45 <- rowMeans(conspiracy[, 5:6])

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
                       "cons_biowpn_dummy")])

conspiracy[, discrete_cols] <-
  lapply(conspiracy[, discrete_cols], factor)

str(conspiracy)

summary(conspiracy)


# 1. Inference model using logistic regression ----

attach(conspiracy)

table(cons_biowpn_dummy)

# check correlation
ggcorr(
  conspiracy %>% dplyr::select(where(is.numeric)),
  nbreaks = 5,
  vjust = "inward",
  hjust = "inward",
  size = 3
)

# DO NOT use all variables because some represent aliased variables
# as shown by NA's in the below code
summary(glm(cons_biowpn_dummy ~ .,
            data = conspiracy,
            family = binomial))

# model with all one-factor terms
glm.fit <-
  glm(
    cons_biowpn_dummy ~
      # ms_news +
      md_agg + md_localpap + md_localtv + md_broadcast + md_national + md_radio +
      # rw_news +
      md_con + md_fox +
      pid2 +
      populism_1 +
      populism_2 +
      populism_3 +
      populism_4 +
      populism_5 +
      # populism123 +
      # pipulism45 +
      white +
      highered +
      hispanic +
      gender +
      age +
      hhi +
      trust_1 +
      idlg
    ,
    data = conspiracy,
    family = binomial
  )

summary(glm.fit)

# check for multicollinearity
vif(glm.fit)


# use glmulti for variable selection with AIC as information criterion
# glmulti should take around 12 minute to run in a 24 GB system.
# Either don't run or be prepared.
glmulti.logistic.out <-
  glmulti(
    cons_biowpn_dummy ~
      ms_news +
      # md_agg + md_localpap + md_localtv + md_broadcast + md_national + md_radio +
      rw_news +
      # md_con + md_fox +
      pid2 +
      populism_1 +
      populism_2 +
      populism_3 +
      populism_4 +
      populism_5 +
      # populism123 +
      # populism45 +
      white +
      highered +
      hispanic +
      gender +
      age +
      hhi +
      trust_1 +
      idlg
    ,
    data = conspiracy,
    level = 1,
    # No interaction considered
    method = "h",
    # Exhaustive approach
    crit = "aic",
    # AIC as criteria
    confsetsize = 5,
    # Keep 5 best models
    plotty = F,
    report = F,
    # No plot or interim reports
    fitfunction = "glm",
    # glm function
    family = binomial
  )

glmulti.logistic.out@formulas

summary(glmulti.logistic.out@objects[[1]])

# save the important models for later use
# saveRDS(glmulti.logistic.out@objects[[1]], "model_1.rds")
# saveRDS(glmulti.logistic.out@objects[[1]], "model_2.rds")
# my_model1 <- readRDS("model_1.rds")
# my_model2 <- readRDS("model_2.rds")

# create the finalized model object
final_model <- glm(
  cons_biowpn_dummy ~
    rw_news +
    pid2 +
    populism_1 +
    populism_3 +
    populism_5 +
    white +
    age
  ,
  data = conspiracy,
  family = binomial
)

# calculate the odd ratios with 95 % CI
or_glm(
  data = conspiracy,
  model = final_model,
  incr = list(
    rw_news = 1,
    populism_1 = 1,
    populism_3 = 1,
    populism_5 = 1,
    white = 1,
    age = 1
  )
)

# plot the odd ratio on log scale
# the following code has been taken and modified form this website:
# https://stackoverflow.com/questions/47085514/simple-way-to-visualise-odds-ratios-in-r
boxLabels = c("rw_news",
              "pid2",
              "populism_1",
              "populism_3",
              "populism_5",
              "white",
              "age")

# create a data frame with odd ratios and CI's
df <-
  data.frame(
    yAxis = 7:1,
    boxOdds = c(1.733,
                1.946,
                1.627,
                1.261,
                1.707,
                0.679,
                0.989),
    boxCILow = c(1.474,
                 1.370,
                 1.318,
                 1.053,
                 1.448,
                 0.462,
                 0.979),
    boxCIHigh = c(2.045,
                  2.772,
                  2.017,
                  1.510,
                  2.018,
                  0.995,
                  0.999)
  )

(
  p <- ggplot(df, aes(x = boxOdds, y = boxLabels)) +
    geom_vline(aes(xintercept = 1), size = .25, linetype = 'dashed') +
    geom_errorbarh(
      aes(xmax = boxCIHigh, xmin = boxCILow),
      size = .5,
      height = .2,
      color = 'gray50'
    ) +
    geom_point(size = 3.5, color = 'orange') +
    theme_minimal() +
    theme(panel.grid.minor = element_blank()) +
    coord_trans(x = 'log10') +
    ylab('') +
    xlab('Odds ratio (log scale)')
)
