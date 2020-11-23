# Title:    Quant III (Lab 10)
# Name:     Junlong Aaron Zhou
# Date:     November 20, 2020
# Summary:  Penalized regression estimation (lasso, ridge, elastic net)
#           Bayesian model selection and averaging
#           Based on: 
#           - Stefan Zeugner (2011) "Bayesian Model Averaging with BMS"
#           - Adrian E. Raftery, Ian S. Painter and Christopher T. Volinsky (2005) "BMA: An R package for Bayesian Model Averaging"
#################################################################################

rm(list = ls())

# install.packages('BMS')
# install.packages('BMA')
# install.packages('glmnet')
# install.packages('caret)
library(BMS)
library(BMA)
library(glmnet)
library(caret)

d <- get(data(attitude))
# rating -- (response) -- satisfaction rating of an organization's employees
# complaints -- (explanatory)	-- Handling of employee complaints
# privileges -- (explanatory)	--	Does not allow special privileges
# learning	-- (explanatory)	--	Opportunity to learn
# raises	-- (explanatory)	--	Raises based on performance
# critical	-- (explanatory)	--	Too critical
# advance	-- (explanatory)	--	Advancement
# help("attitude")
Sys.sleep(1)
rm(attitude)
str(d)

# 6 explanatory vars --> 2^6 = 64 models


set.seed(123456)
# Split data into: training and test sets
row_test <- sample(x = 1:nrow(d), size = 10, replace = F)
d_test <- d[row_test,]
d_train <- d[-row_test,]
Sys.sleep(1) 



##########################################################################################
#################################### Ridge and lasso #####################################
##########################################################################################
x_trans <- caret::preProcess(x = d_train[, 2:ncol(d_train)], method = c("center", "scale"))
x_scaled_train <- as.matrix( predict(x_trans, d_train[,2:ncol(d_train)]) )
x_scaled_test <- as.matrix( predict(x_trans, d_test[,2:ncol(d_test)]) )
# Scale your x's! 
# NB! You should apply the training set scaling to the test set!!! 
# caret::preProcess() helpful here


mod.l1.cv <- cv.glmnet(x = x_scaled_train, y = d_train[,1], 
                       nfolds = 5, alpha = 1, family = "gaussian")
# NB! x should be a matrix, not a dataframe. Also accepts sparse matrices!
# alpha is weight on l1-regularization; alpha = 0 produces ridge;
# (alpha > 0 & alpha < 1) produces elastic nets 

# See cross-validation curve
plot(mod.l1.cv)


# Extract the best lambda (or, best lambda + 1 se)
(lambda.l1 <- mod.l1.cv$lambda.1se) # lambda.min
# lambda.min: lambda that gives minimum mean cross-validated error
# lambda.1se: the most regularized model s.t. CV error is within 1 sd


# Reestimate the model
mod.l1 <- glmnet(x = x_scaled_train, y =  d_train[,1], alpha = 1, family = "gaussian")

plot(mod.l1, label = T, xvar = 'lambda')
plot(mod.l1, label = T, xvar = 'norm')
# By default, xvar = 'norm', which is not as intuitive, imo


coef(mod.l1, s = lambda.l1)
coef(mod.l1)
# NB! No standard errors. No conventional way. Hastie and Tibshirani recommend bootstrap
# cf: coefs directly extracted from the CV-version
coef(mod.l1.cv) 
 

# Here, I'm predicting y for the 1st 5 rows of the training dataset
pred.l1 <- predict(mod.l1, s = lambda.l1, newx = x_scaled_train[1:5,])
cat('Lasso RSS:', sum( (d_train$rating[1:5] - pred.l1)^2 ) )

# Compare to OLS
mod.ols <- lm(rating ~ ., data = d_train)
pred.ols <- predict(object = mod.ols, d_train)
cat('OLS RSS:', sum( (d_train$rating[1:5] - pred.ols[1:5])^2 ) )
# Why is OLS RSS smaller? Would we expect this?


# Now, consider the test set
pred.ols.test <- predict(object = mod.ols, d_test)
pred.l1.test <- predict(object = mod.l1, s = lambda.l1, newx = x_scaled_test )

cat('Lasso test RSS:', sum( (d_test$rating - pred.l1.test)^2 ) )
cat('OLS test RSS:', sum( (d_test$rating - pred.ols.test)^2 ) )



########################
# LAB ASSIGNMENT: estimate a ridge model 

########################
# LAB ASSIGNMENT: estimate an elastic net model
 
##########################################################################################
################################ Bayesian Model Selection ################################
##########################################################################################
mod.bms.exact <- bms(rating ~ ., data = d_train, mprior = 'uniform', g = 'UIP', user.int = F)
# UIP = unit information prior: g = N in the prior variance for coefs
# mprior = 'uniform' --> uniform model prior
# user.int = F --> to suppress user-interactive output


###### Inspect 5 best models
topmodels.bma(mod.bms.exact)[, 1:5]
# PMP = posterior model probability Pr(M | y, X)

# Visualize models with cum. PMP
image(mod.bms.exact)
# blue - positive sign
# red - negative sign
# white - zero (variable not included in the model)


###### Inspect variable effects model by model (!)
beta.draws.bma(mod.bms.exact)[, 1:5] # posterior expectations of parameters for 5 best models
beta.draws.bma(mod.bms.exact, stdev = T)[, 1:5] # posterior sd of parameters for 5 best models




# Predictions with the best model
bma_best_expect <- beta.draws.bma(mod.bms.exact)[, 1]
pred.bms.test <- as.matrix(d_test[, names(bma_best_expect)]) %*% bma_best_expect
cat('BMS test RSS:', sum( (d_test$rating - pred.bms.test)^2 ) )
cat('Lasso test RSS:', sum( (d_test$rating - pred.l1.test)^2 ) )
cat('OLS test RSS:', sum( (d_test$rating - pred.ols.test)^2 ) )



##########################################################################################
################################ Bayesian Model Averaging ################################
##########################################################################################

###### Inspect aggregated variable importance and effects
coef(mod.bms.exact)
# - PIP = posterior inclusion probability (sum of PMPs wherein the covariate was included)
# - Post Mean = Unconditional posterior expectation for coefficient (unconditional on inclusion)
# i.e. coef averaged over all models (including those where the variable was not included)
# 
# To condition on inclusion: coef(mod.bms.exact, condi.coef = T)
# - Cond.Pos.Sign = proportion of models with a positive sign for the variable 
#                   (conditional on inclusion)
# - Idx = original order of variables

# Check that PIP = sum of PMPs wherein the covariate was included
(a <- topmodels.bma(mod.bms.exact))
sapply(1:(nrow(a)-2), function(k) sum(a['PMP (Exact)', 
                                        which(a[k,] > 0)]) )
rownames(a)


sapply(1:(nrow(a)-2), function(k)  
    sum(beta.draws.bma(mod.bms.exact)[k,]*a['PMP (Exact)',]) )
 

# Posterior marginal density for a coefficient
density(mod.bms.exact, reg = 'complaints')
# model-weighted mixture of posterior densities for each model

pred.bma.test <- as.matrix(d_test[,names(coef(mod.bms.exact)[,2])]) %*% coef(mod.bms.exact)[,2]
cat('BMA test RSS:', sum( (d_test$rating - pred.bma.test)^2 ) )
cat('BMS test RSS:', sum( (d_test$rating - pred.bms.test)^2 ) )
cat('Lasso test RSS:', sum( (d_test$rating - pred.l1.test)^2 ) )
cat('OLS test RSS:', sum( (d_test$rating - pred.ols.test)^2 ) )


# Predictions for each ob
pred.bma.test.dens <- pred.density(mod.bms.exact, newdata = d_test)
plot(pred.bma.test.dens)




#----------------
##### Additional info
summary(mod.bms.exact) 
# Posterior mean no. regressors
coef(mod.bms.exact)
round( as.numeric(summary(mod.bms.exact)["Mean no. regressors"]), 3) ==
round( sum( coef(mod.bms.exact)[,1] ) , 3)
# coef(mod.bms.exact)[,1] extracts PIP (posterior inclusion prob) for each variable


# Visualize model size
plotModelsize(mod.bms.exact)


# Number of models considered
as.numeric(summary(mod.bms.exact)["No. models visited"]) == 2^(ncol(d)-1)

# Model prior: "uniform / 3": uniform prior on models, and E_{prior}(model size) = K/2



#-----------------
##### Robustness of bms results to changes in priors
mod.bms.prior <- bms(rating ~ ., data = d_train, mprior = 'random', mprior.size = 2) 
summary(mod.bms.prior)
# "random" model prior: 'random theta' prior by Ley and Steel (2009): 
#                       binomial-beta hyperprior on the a priori inclusion probability



#-----------------
##### Optional: HUGE models 
# 2^K produces stackoverflow quite quickly. Alternative - MCMC, i.e.
# instead of computing exact PMPs, APPROXIMATE them with MCMC

mod.bms.mcmc <- bms(rating ~ ., data= d_train, 
                    burn = 1000, iter = 1000, 
                    g = "BRIC", mprior = "uniform", 
                    nmodel = 500, mcmc = "bd", user.int = F)

# Check convergence (Corr PMP)
summary(mod.bms.mcmc)

plotConv(mod.)
plotConv(mod.bms.mcmc[1:10])
# MCMC and exact PMP are almost identical --> good convergence


# Inspect PMPs more closely
pmp.bma(mod.bms.mcmc)[1:3,]


# Inspect variables
print(mod.bms.mcmc)



##########################################################################################
################################ Bayesian Model Averaging Using BMA ################################
##########################################################################################

mod.bma <- bicreg(y = d[,1], x = d[,2:ncol(d)])
summary(mod.bma)
# p!=0 : posterior prob that variable is in the model
# EV : posterior mean
# SD : posterior sd
# model 1 - 5: 5 best models with parameter estimates and model quality stats

# cf: beta.draws.bma(mod.bms.exact)[, 1:5]

plot(mod.bma)
# Plots of posterior densities, given a variable is in the model
# approximated by a finite misture of normals and scaled so that 
# the height = posterior prob of variable in the model


# GLM case:

d$d_rating <- as.numeric(d$rating>median(d$rating))

mod.bma.glm <- bic.glm(y = d[,ncol(d)], x= d[,2:(ncol(d)-1)],
                       glm.family="binomial")

str(mod.bma.glm)
sum(mod.bma.glm$postprob)

summary(mod.bma.glm)
imageplot.bma(mod.bma.glm)
