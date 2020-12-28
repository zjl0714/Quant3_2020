# Title:    Quant III (Lab 12)
# Name:     Junlong Aaron Zhou
# Date:     December 12, 2020
# Summary:  Trees and forests
#################################################################################

rm(list = ls())
 
library(ISLR)
library(tree)
library(rpart)
library(rpart.plot)
library(randomForest)
library(gbm)
library(dplyr)
library(ROCR) 


##### We are using `Carseats` from ISL book

data("Carseats")
d_all <- Carseats
rm(Carseats)
d_all$High <- ifelse(d_all$Sales <= 8, "No", "Yes")
d_all$High <- as.factor(d_all$High)

str(d_all)
 
##### We don't need sales when training the model

d <- d_all %>% select(-"Sales")

set.seed(123456)
##### Split train -- test
test.obs <- sample(x = 1:nrow(d), size = 0.2*nrow(d), replace = F)
test.d <- d[test.obs,]
train.d <- d[-test.obs,]

#################################################################################
##################################### TREES #####################################
#################################################################################
##### Two main packages: tree and rpart. Very similar

##### CART: Categorical and Regression Tree

#------------------------ tree package
##### Fit the model
mod.tree.dev <- tree(High ~ ., data = train.d, split = 'deviance')
mod.tree.gini <- tree(High ~ ., data = train.d, split = 'gini')

summary(mod.tree.dev)
summary(mod.tree.gini)
# Deviance is -2 \sum_m \sum_k n_{mk}*log \hat p_{mk},
#             where n_{mk} is num of obs in mth node, kth class
# Residual mean deviance is deviance divieded by n-|T|, 
#                         where |T| is num of terminal nodes


##### Plot tree
plot(mod.tree.dev)
text(mod.tree.dev, pretty = 1) # pretty=0 to have names for qualitative predictors
# Look at the bottom split: Why No - No?


##### Get tree info
print(mod.tree.dev)
# 2) ShelveLoc: Bad,Medium 254 310.000 No ( 0.70079 0.29921 ) 
# ShelveLoc: Bad,Medium - split criterion
# 254 - num of obs in this branch
# 310.0 - deviance
# No - overall prediction
# (  0.70079 0.29921 ) - prop. of obs that take values No and Yes


##### Prediction on test set
pred.tree.dev <- predict(object = mod.tree.dev, 
                         newdata = test.d, 
                         type = 'class')
xtabs(~ pred.tree.dev + test.d$High)

pred.tree.gini <- predict(object = mod.tree.gini, newdata = test.d, type = 'class')
xtabs(~ pred.tree.gini + test.d$High)


#### Can we do better with pruning?
cv.res <- cv.tree(object = mod.tree.dev, 
                  FUN = prune.tree, method = 'misclass') # or method='deviance'
# size is num of terminal nodes
# k is \alpha
# dev is CV error rate

par(mfrow = c(1,2))
plot(cv.res$dev ~ cv.res$size, type = 'l', xlab = 'Number of nodes', ylab = 'CV error rate')
plot(cv.res$dev ~ cv.res$k, type = 'l', xlab = 'Complexity penalty', ylab = 'CV error rate')
par(mfrow = c(1,1))

# Get the best model size
(best.cv.size <- cv.res$size[which.min(cv.res$dev)])

# Prune the model
mod.prune.dev <- prune.tree(tree = mod.tree.dev, 
                            best = best.cv.size, 
                            method = 'misclass')
plot(mod.prune.dev)
text(mod.prune.dev, pretty = 0)

# Inspect model quality
pred.prune.dev <- predict(object = mod.prune.dev, newdata = test.d, type = 'class')
xtabs(~ pred.prune.dev + test.d$High)

# cf: original
xtabs(~ pred.tree.dev + test.d$High)


# ROC
pred.roc.tree.dev.prune <- prediction(predictions = predict(object = mod.prune.dev, 
                                                   newdata = test.d, type = 'vector')[,2], 
                             labels = test.d$High)
plot( performance(pred.roc.tree.dev.prune, "tpr", "fpr"), col="Blue" )
abline(0, 1, lty = 2, col = 'red')

pred.roc.tree.dev <- prediction(predictions = predict(object = mod.tree.dev, 
                                                   newdata = test.d, type = 'vector')[,2], 
                             labels = test.d$High)
plot( performance(pred.roc.tree.dev, "tpr", "fpr"), add=T, col="black" ) 






#------------------------ rpart package


# The rpart package is an alternative method for 
# fitting trees in R. It is much more feature rich, 
# including fitting multiple cost complexities and 
# performing cross-validation by default. 
# It also has the ability to produce much nicer trees.
# Based on its default settings, it will often result 
# in smaller trees than using the tree package.  
# rpart can also be tuned via caret.
# https://cran.r-project.org/web/packages/rpart/vignettes/longintro.pdf


##### Fit the model
mod.rpart.gini <- rpart(High ~ ., data = train.d, 
                        method = 'class', 
                        parms = list(split = 'gini'))


summary(mod.rpart.gini)
print(mod.rpart.gini)
##### Plot tree
plot(mod.rpart.gini)
text(mod.rpart.gini)
rpart.plot(mod.rpart.gini)

##### Cross-validation

plotcp(mod.rpart.gini)

##### Prediction on test set

pred.rpart.gini <- predict(object = mod.rpart.gini, 
                           newdata = test.d, 
                           type = 'class')
xtabs(~ pred.rpart.gini + test.d$High)

pred.rpart.gini.prob <- predict(object = mod.rpart.gini, 
                                newdata = test.d, 
                                type = 'prob')
str(pred.rpart.gini.prob)

# Also, ROC curve
pred.roc.rpart <- prediction(predictions = pred.rpart.gini.prob[,2], 
                             labels = test.d$High)
plot( performance(pred.roc.rpart, "tpr", "fpr"), col="Blue")
abline(0, 1, lty = 2, col = 'red')

pred.roc.tree.gini <- prediction(predictions = predict(object = mod.tree.gini, 
                                                      newdata = test.d, type = 'vector')[,2], 
                                labels = test.d$High)
plot( performance(pred.roc.tree.gini, "tpr", "fpr"), add=T, col="black" ) 





##### Regression Tree 

##### Fit the model
d.r <- d_all %>% select(-"High")
test.d.r <- d.r[test.obs,]
train.d.r <- d.r[-test.obs,]
mod.rpart.reg <- rpart(Sales ~ ., data = train.d.r, 
                        method = 'anova', 
                        parms = list(split = 'information'))


print(mod.rpart.reg)

##### Plot tree
rpart.plot(mod.rpart.reg)

##### Cross-validation

plotcp(mod.rpart.reg)


##### Prediction on test set
pred.rpart.reg <- predict(object = mod.rpart.reg, 
                           newdata = test.d.r, 
                           type = 'vector') 


mean((pred.rpart.reg-test.d.r$Sales)^2)

plot(test.d.r$Sales~pred.rpart.reg)
abline(0, 1, lty = 2, col = 'red')
 


#################################################################################
#################################### BAGGING ####################################
#################################################################################
# What's the relation between bagging and random forests?
mod.bag <- randomForest(High ~ ., data = train.d, 
                        ntree = 500, 
                        mtry = 10,
                        importance = T)

print(mod.bag)
# OOB stands for "out-of-bag"

mod.bag$confusion
mod.bag$votes # one row for each input data point and one column for each class, giving the fraction or number of (OOB) ‘votes??? from the random forest.
mod.bag$importance
# class-specific mean decreases in accuracy: decrease of accuracy in predictions on the out of bag samples when a given variable is excluded from the model


##### Importance of predictor

importance(mod.bag)
varImpPlot(mod.bag)

##### Prediction
pred.bag <- predict(object = mod.bag, test.d, 
                    type = 'class')
xtabs(~ pred.bag + test.d$High)

# ROC
pred.bag.prob <- predict(object = mod.bag, test.d, type = 'prob')
pred.roc.bag <- prediction(predictions = pred.bag.prob[,2], labels = test.d$High)
plot( performance(pred.roc.bag, 'tpr', 'fpr') )
abline(0, 1, lty = 2, col = 'red')

# Both ROC curves together
plot( performance(pred.roc.rpart, 'tpr', 'fpr'), lwd = 2 )
plot( performance(pred.roc.bag, 'tpr', 'fpr'), add = T, col = 'blue', lwd = 2 )
abline(0, 1, lty = 2, col = 'red')

# AUC
performance(pred.roc.rpart, 'auc')@y.values
performance(pred.roc.bag, 'auc')@y.values


#################################################################################
################################# RANDOM FOREST #################################
#################################################################################
mod.rf <- randomForest(High ~ ., data = train.d, 
                       ntree = 500, 
                       mtry = sqrt(10), # We usually choose sqrt(p)
                       importance = T)

##### Prediction
pred.rf <- predict(object = mod.rf, newdata = test.d, 
                   type = 'response')
xtabs(~ pred.rf + test.d$High)


##### Importance of predictor

importance(mod.rf)
varImpPlot(mod.rf)


##### Model performance
pred.rf.prob <- predict(object = mod.rf, newdata = test.d, 
                        type = 'prob')
pred.roc.rf <- prediction(predictions = pred.rf.prob[,2], 
                          labels = test.d$High)

plot( performance(pred.roc.rpart, 'tpr', 'fpr'), lwd = 2 )
plot( performance(pred.roc.bag, 'tpr', 'fpr'), add = T, col = 'blue', lwd = 2 )
plot( performance(pred.roc.rf, 'tpr', 'fpr'), add = T, col = 'red', lwd = 2 )
abline(0, 1, lty = 2, col = 'red')

# AUC
performance(pred.roc.rf, 'auc')@y.values



#################################################################################
################################### BOOSTING ####################################
#################################################################################
train.d$High_num <- ifelse(train.d$High == "No", 0, 1)
test.d$High_num <- ifelse(test.d$High == "No", 0, 1)


mod.boost <- gbm(High_num ~ ., data = subset(train.d, select = -High), 
                 distribution = 'bernoulli', 
                 n.trees = 5000, interaction.depth = 4)

print(mod.boost)
summary(mod.boost)


##### Prediction
pred.boost.prob <- predict(object = mod.boost, subset(test.d, select = -High), 
                           n.trees = 5000, 
                           type="response")

pred.boost <- as.numeric(pred.boost.prob > 0.5)
xtabs(~ pred.boost + test.d$High_num)


##### Model performance
pred.roc.boost <- prediction(predictions = pred.boost.prob, 
                             labels = test.d$High_num)

plot( performance(pred.roc.rpart, 'tpr', 'fpr'), lwd = 2 )
plot( performance(pred.roc.bag, 'tpr', 'fpr'), add = T, col = 'blue', lwd = 2 )
plot( performance(pred.roc.rf, 'tpr', 'fpr'), add = T, col = 'red', lwd = 2 )
plot( performance(pred.roc.boost, 'tpr', 'fpr'), add = T, col = 'green', lwd = 2 )
abline(0, 1, lty = 2, col = 'red')


# AUC
cat('AUC:\nTree:', unlist( performance(pred.roc.rpart, 'auc')@y.values ),
    '\nBagging:', unlist( performance(pred.roc.bag, 'auc')@y.values ),
    '\nRandom Forest:', unlist( performance(pred.roc.rf, 'auc')@y.values ),
    '\nBoosting:', unlist( performance(pred.roc.boost, 'auc')@y.values )) 


###############################################################################
################################### THE END ###################################
###############################################################################
####################### Hooraaay!! You've made it!! :-D #######################
###############################################################################
