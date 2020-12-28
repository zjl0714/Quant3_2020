# Title:    Quant III (Lab 11)
# Name:     Junlong Aaron Zhou
# Date:     December 4, 2020
# Summary:  Nonparametrics: splines
#                           locfit 
#################################################################################

library(haven)
library(mgcv) 
library(locfit)

rm(list = ls())

setwd('~/Dropbox/Teaching/2020_Fall_Quant_3/lab11_nonparametric_regression/')
 
##### Load data
#dat <- read_dta("qog_std_ts_jan17.dta")
#d <- subset(dat, wef_infl < 200)
#save(x = d, file = 'qog_subdat.Rdata')

d <- get(load('qog_subdat.Rdata'))
d <- subset(d, !is.na(wef_infl) & !is.na(p_polity2) & !is.na(log(gle_rgdpc)))
dim(d)


#######################################################################################
############################ Inflation and democracy + gdp ############################
#######################################################################################

################# First, bivariate analysis: inflation and democracy


##### Plot variables
plot(d$wef_infl ~ d$p_polity2)


#### GAM: Generalized Linear Model:

mod.lm <- gam(wef_infl~p_polity2, data=d)
print(mod.lm)
summary(mod.lm)
lines(mod.lm$fitted.values~mod.lm$model[,2])

#---------------------
##### A) Basic TPRS
# thin plate regression spline (TPRS)
mod.tprs <- gam(wef_infl ~ s(p_polity2), data = d)

# What are we doing here?
?gam
?s

# By default, method is thin plate
# bs="tp". These are low rank isotropic smoothers of any number 
# of covariates. By isotropic is meant that rotation of the covariate
# co-ordinate system will not change the result of smoothing. 
# By low rank is meant that they have far fewer coefficients 
# than there are data to smooth. They are reduced 
# rank versions of the thin plate splines and use the thin
# plate spline penalty. They are the default smooth for s
# terms because there is a defined sense in which they are
# the optimal smoother of any given basis dimension/rank (Wood, 2003).
# Thin plate regression splines do not have ‘knots’ 
# (at least not in any conventional sense): a truncated
# eigen-decomposition is used to achieve the rank reduction. 



print(mod.tprs)
summary(mod.tprs)
plot(mod.tprs, residuals = T) # 95% Bayesian credible intervals
### Things to notice here:
# 1) Parametric and smooth terms
# 2) edf for smooth terms = 8.107
# 3) GCV = 22.0396 

# Lambda value? Well, only a rescaled one 
# (https://stackoverflow.com/questions/38644943/mgcv-how-to-return-estimated-smoothing-parameter):
mod.tprs$sp    # sp stands for smoothing parameter
# 4) Lambda-tilde = 0.00997 

##### B) TPRS: GCV gamma
mod.tprs.gamma <- gam(wef_infl ~ s(p_polity2), data = d, gamma = 1.4)
summary(mod.tprs.gamma)
# 2) edf for smooth terms = 7.697 (vs. 8.107)
# 3) GCV = 22.257 (vs. 22.0396)
mod.tprs.gamma$sp
# 4) Lambda-tilde = 0.01733
# Lambda-tilde is larger, since we penalized overfitting --> more smoothing, worse fit

# But no obvious differences 
par(mfrow = c(1,2))
plot(mod.tprs, residuals = T)
plot(mod.tprs.gamma, residuals = T)
par(mfrow = c(1,1))

# Compare predictions
preds.tprs <- predict(mod.tprs, se = T)
preds.tprs.gamma <- predict(mod.tprs.gamma, se = T)

plot(preds.tprs.gamma$fit ~ preds.tprs$fit, 
     xlab = 'TPRS, gamma=1', ylab = 'TPRS, gamma=1.4',
     main = 'Predicted values (in-sample)')
abline(a = 0, b = 1, col = 'red')
# gamma = 1.4 did not produce any major changes in the model



#### B) TPRS: dimensionality (k)
# NB! k-1 is the upper bound on edf
# K is not number of knots in TPRS, it's the number of basis function.
# In cubic spline, the basic function is given in a certain form.

mod.tprs.k10 <- gam(wef_infl ~ s(p_polity2, k=10), data = d) 
mod.tprs.k20 <- gam(wef_infl ~ s(p_polity2, k=20), data = d) 


plot(mod.tprs.k10)

summary(mod.tprs.k10)
mod.tprs.k10$coefficients
# GCV = 22.04; edf = 8.107; lambda-tilde = 0.00997 (exactly as before!)
summary(mod.tprs.k20)
# GCV = 21.362; edf = 16.96; lambda-tilde = 0.003427

preds.tprs.k10 <- predict(mod.tprs.k10, se = T)
preds.tprs.k20 <- predict(mod.tprs.k20, se = T)

plot(preds.tprs.k20$fit ~ preds.tprs.k10$fit, 
     xlab = 'TPRS, k=10', ylab = 'TPRS, k=20',
     main = 'Predicted values (in-sample)')
abline(a = 0, b = 1, col = 'red')
# Yes, k does make a difference

par(mfrow = c(1,2))
plot(mod.tprs.k10, residuals = T)
plot(mod.tprs.k20, residuals = T)
par(mfrow = c(1,1))
# Overfitting with k=20?

# There is a "test" for too small k. 
# R-help: "The test of whether the basis dimension for a smooth is adequate 
#         (Wood, 2017, section 5.9) is based on computing an estimate of the 
#         residual variance based on differencing residuals that are near 
#         neighbours according to the (numeric) covariates of the smooth. 
#         This estimate divided by the residual variance is the k-index reported. 
#         The further below 1 this is, the more likely it is that there is missed 
#         pattern left in the residuals. The p-value is computed by simulation: 
#         the residuals are randomly re-shuffled k.rep times to obtain the null 
#         distribution of the differencing variance estimator, if there is no pattern 
#         in the residuals."
# If p-val < \alpha, increase k
# But...
gam.check(mod.tprs.k10)
gam.check(mod.tprs.k20)


#### C) Different smoothers
?smooth.terms

# K is number of knows in cubic regression splines 

mod.cr <- gam(wef_infl ~ s(p_polity2, bs = 'cr', k =15), data = d)
mod.cr$coefficients
# bs - type of basis smoother
### "tp" (default) Optimal low rank approximation to thin plate spline
### "cr" a penalized cubic regression spline 
### "ps" Eilers and Marx style P-splines 
mod.ps <- gam(wef_infl ~ s(p_polity2, bs = 'ps'), data = d)

par(mfrow = c(2,2))
plot(mod.tprs.k10, residuals = T, main = 'TPRS')
plot(mod.cr, residuals = T, main = 'Cubic regression')
plot(mod.ps, residuals = T, main = 'P-spline')
par(mfrow = c(1,1))

# Compare predictions
preds.cr <- predict(mod.cr, se = T)
preds.ps <- predict(mod.ps, se = T)

par(mfrow = c(2,2))
plot(preds.tprs.k10$fit ~ preds.cr$fit, xlab = 'CR', ylab = 'TPRS')
abline(a = 0, b = 1, col = 'red')
plot(preds.tprs.k10$fit ~ preds.ps$fit, xlab = 'PS', ylab = 'TPRS')
abline(a = 0, b = 1, col = 'red')
plot(preds.cr$fit ~ preds.ps$fit, xlab = 'PS', ylab = 'CR')
abline(a = 0, b = 1, col = 'red')
par(mfrow = c(1,1))

# Why using cubic splines instead of TPRS? 
# TPRS are VERY slow with large N 



#-----------------
##### Predictions and plots
# In-sample
preds.tprs <- predict(mod.tprs, se = T)
plot(preds.tprs$fit ~ mod.tprs$fitted.values)
abline(a = 0, b = 1, col = 'red')
# How many UNIQUE predicted values are out there? Why?


preds.tprs.out <- predict(mod.tprs, se = T, newdata = data.frame('p_polity2' = seq(-10, 10)))
plot(preds.tprs.out$fit ~ seq(-10, 10), type = 'l')



################# Second, multivariate analysis: inflation ~ democracy and gdp


# Two separate function g(democracy) g(log gdp)

mod.bv1 <- gam(wef_infl ~ s(p_polity2)+ s(log(gle_rgdpc)), data = d) # thin plate regression spline

plot(mod.bv1, page=1) 

preds.bv1 <- predict(mod.bv1, se = T)
vis.gam(mod.bv1, theta=20,phi=20)
vis.gam(mod.bv1, theta=90,phi=10)

par(mfrow = c(1,2))
plot(d$wef_infl ~ preds.bv1$fit)
plot(d$wef_infl ~ preds.tprs$fit)
par(mfrow = c(1,1))



# Now: a bivariate function g(democracy, log gdp)

mod.bv2 <- gam(wef_infl ~ s(p_polity2,log(gle_rgdpc)), data = d) # thin plate regression spline
plot(mod.bv2) 

vis.gam(mod.bv2, theta=20,phi=20)
vis.gam(mod.bv2, plot.type = "contour")


# Not sure if you want to go this way...


preds.bv2 <- predict(mod.bv2, se = T)

par(mfrow = c(1,2))
plot(d$wef_infl ~ preds.bv2$fit)
plot(d$wef_infl ~ preds.tprs$fit)
par(mfrow = c(1,1))
 

#######################################################################################
#################################### Locfit ###########################################
#######################################################################################


mod.locfit <- locfit(wef_infl~p_polity2, data=d)
plot(y = d$wef_infl, x = d$p_polity2)
plot(mod.locfit, add = T)

mod.locfit.1 <- locfit(wef_infl~lp(p_polity2), data=d)
plot(mod.locfit.1, add = T, col="red")


# h changes bandwidth in kernel

mod.locfit.h <- locfit(wef_infl~lp(p_polity2,h=0.5), data=d)
plot(mod.locfit.h, add = T, col="blue")

mod.locfit.hn <- locfit(wef_infl~lp(p_polity2,h=2), data=d)
plot(mod.locfit.hn, add = T, col="black")

# change kernel
 
mod.locfit.kern <- locfit(wef_infl~lp(p_polity2,h=2), kern="tria", data=d)
plot(mod.locfit.kern, add = T, col="green")

out <- lapply(1:5, function(k){
    return(locfit(wef_infl~lp(p_polity2, deg = k), data=d))
})

plot(y = d$wef_infl, x = d$p_polity2)
plot(out[[1]], add = T, col=1)
plot(out[[2]], add = T, col=2)
plot(out[[3]], add = T, col=3)
plot(out[[4]], add = T, col=4)
plot(out[[5]], add = T, col=5)
