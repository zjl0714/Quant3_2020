# Title:    Quant III (Lab 3)
# Name:     Junlong Aaron Zhou
# Date:     September 25, 2020
# Summary:
#         Likelihood Ratio Test
#         Poisson regression: parametric bootstrap, marginal effect and CI for them  
#         Known Distribution: Monte Carlo Simulation
#         Unknown Distribution: Bootstrap: nonparametric and parametric
#           
#################################################################################

rm(list = ls())

# install.packages("mfx")
# install.packages('mvtnorm')
library(lmtest)
library(mfx)
library(mvtnorm)


########################################################################
#                                    Model Test 
########################################################################



##### Poisson regression example
# Let's generate some data for a Poisson regression model
set.seed(12345)
n = 1000
beta0 = 1
beta1 = 0.5
beta2 = -0.5
dat <- data.frame(x = rnorm(n, mean = 3, sd = 0.5), z = rnorm(n, mean = 3, sd = 0.5))
lambda <- exp(beta0 + beta1 * dat$x + beta2 * dat$z) # the mean function
dat$y <- rpois(n = n, lambda = lambda) # the dep.var

mod.unres <- glm(y~x+z, family = poisson(link="log"), data= dat)
mod.res <- glm(y~x, family = poisson(link="log"), data=dat)


llik_unres <- function(par, data) {
  - sum(data$y * (par[1] + par[2] * data$x + par[3] * dat$z) ) + 
    sum( exp(par[1] + par[2] * dat$x + par[3] * dat$z) ) 
}

llik_res <- function(par, data) {
  - sum(data$y * (par[1] + par[2] * data$x + 0* dat$z) ) + 
    sum( exp(par[1] + par[2] * dat$x + 0 * dat$z) ) 
}


optim.mod.unres <- optim(par = rnorm(3), fn = llik_unres, data = dat, hessian = T)
optim.mod.res <- optim(par = rnorm(2), fn = llik_res, data = dat, hessian = T)

(llk_unres <- logLik(mod.unres))
(llk_res <- logLik(mod.res))
(optim.mod.unres$value)
(optim.mod.res$value)

# Quiz: why there is discrepency?


-optim.mod.unres$value-sum(log(factorial(dat$y)))
-optim.mod.res$value-sum(log(factorial(dat$y)))

lmtest::lrtest(mod.res,mod.unres)



#######################################################################################
#                                     Marginal Effects   
#######################################################################################

##### Poisson regression example
#  Use the same data above
#--------------------------
##### Marginal effects: 

# What's a marginal effect here? How do we compute it? Write it down

# Function for SAME (Sample Average Marginal Effects) of variables
mef <- function(data, betas, dep.var = 'y') {
  X <- cbind( 1, as.matrix( data[,-which(names(data) == dep.var)] ) ) # add column of 1s
  storage <- numeric(length = length(betas) - 1) # -1 since no marginal effect the intercept
  for (j in 2:length(betas)) {
    # For each beta: take a row and compute ME with that data; then average across rows 
    storage[j - 1] <- mean(apply(X, 1, function(k) betas[j] * exp( k %*% betas)))
  }
  return(storage)
}

betas <- coef(mod.unres)

# Estimate sample average ME of x with our function
mef(data = dat, betas = betas)

# Compare to results from a package
mfx::poissonmfx(y ~ x + z, data = dat, atmean = F) 


# Marginal effect at mean
mfx::poissonmfx(y ~ x + z, data = dat, atmean = T) 

betas%*%exp(betas%*%c(1,colMeans(dat[,c("x","z")])))
 
#---------------
# What if you have a model with an INTERACTION? 
# i.e. lambda = beta0 + beta1 * x + beta2 * z + beta12 * x * z
rm(dat)

set.seed(12345)
n = 1000
beta0 = 1
beta1 = 0.5
beta2 = -0.5
beta12 = 0.25

dat <- data.frame(x = rnorm(n, mean = 3, sd = 0.5), z = rnorm(n, mean = 3, sd = 0.5))
dat$xz <- dat$x * dat$z
lambda <- exp(beta0 + beta1 * dat$x + beta2 * dat$z + beta12 * dat$xz + rnorm(n, sd = 0.25))
dat$y <- rpois(n = n, lambda = lambda)
head(dat)
table(dat$y)

(betas <- coef(glm(y ~ x * z, data = dat, family = poisson(link = 'log'))))


####################################################
##### LAB WORK: Fill in the gaps in the code below to implement SAME mef for
#               a model with an interaction
#               lambda = beta0 + beta1 * x + beta2 * z + beta12 * x * z
##### YOUR CODE STARTS HERE:






##### END OF CODE
####################################################

mef_inter(data = dat, betas = betas)

mfx::poissonmfx(y ~ x * z, data = dat, atmean = F)

# Check what happens if we assume no interaction (re-run mef() from above)
mef(data = dat, betas = betas)

# Wait!!! What does this mean?


# Check this:

(me_inter <- mfx::poissonmfx(y ~ x * z, data = dat, atmean = T))

(betas[2]+betas[4]*mean(dat$z))*exp(betas%*%c(1,colMeans(dat[,c("x","z","xz")])))
 
me_inter$mfxest[1]+me_inter$mfxest[3]*mean(dat$z)

#######################################################################################
#                                     Monte Carlo   
#######################################################################################


# Integrate log(x) from 0 to 1

# 1) Generate uniform values (0-1), 2) compute the integrand, 3) take the average
N = 1e5
u <- runif(N)
n <- 1:N
h_sim <- log(u)
s <- cumsum(h_sim)/n
plot(s, type = 'l', main = 'ln(x) 0-1')

sum(h_sim) / N # answer

# Simulate the N(0,1) cdf at t: F(t) = Pr(X <= t)
rm(list = ls())

t = 1.6
set.seed(20190905)

# Option 1: simulate an N(0,1) sample and compute proportions
n.sim = 1e6
x <- rnorm(n.sim)
sum(x <= t) / n.sim


# Monte Carlo Simulation of Quantile 

rm(list=ls())
set.seed(12345)
n = 1000
beta0 = 1
beta1 = 0.5
beta2 = -0.5
dat <- data.frame(x = rnorm(n, mean = 3, sd = 0.5), z = rnorm(n, mean = 3, sd = 0.5))
head(dat)

lambda <- exp(beta0 + beta1 * dat$x + beta2 * dat$z) # the mean function
dat$y <- rpois(n = n, lambda = lambda) # the dep.var


mod.glm <- glm(y ~ x + z, data = dat, family = poisson(link = 'log'))

# Point estimates 
betas <- coef(mod.glm)
# Variance-covariance matrix for coef estimators
vc <- vcov(mod.glm)


B = 1000 # num of simulation
# sample parameter estimates from the asymptotic sampling distribution 
betas_mat <- mvtnorm::rmvnorm(n = B, mean = betas, sigma = vc) 
str(betas_mat)

# CI
boot_ci <- function(x, alpha = 0.05) {
  x <- x[which(!is.na(x) & !is.nan(x))] # remove NAs and NaNs
  x <- x[order(x)]
  c(x[ceiling(alpha/2 * length(x))], x[ceiling((1 - alpha/2) * length(x))])
}

apply(betas_mat, 2, boot_ci)

t(confint(mod.glm)) # Compare to LR-based CI

# se
apply(betas_mat, 2, sd)

sqrt( diag(vcov(mod.glm)) ) # Compare to glm


# How do we generate the CI of a point prediction? i.e. E(Y|X_i=x)?

#######################################################################################
#                                     Bootstrap
#######################################################################################
 

#################### Nonparametric bootstrap

dat_boot <- function(x, nboots) {
  # returns a list of bootstrapped obs
  lapply(1:nboots, function(k) sample(x = x, size = length(x), replace = T))
}


data_frame_boot <- function(data, nboots) {
  lapply(1:nboots, function(k) data[sample(x = 1:nrow(data), size = nrow(data), replace = T),] )
}


boot_ci <- function(x, alpha = 0.05) {
  x <- x[which(!is.na(x) & !is.nan(x))] # remove NAs and NaNs
  x <- x[order(x)]
  c(x[ceiling(alpha/2 * length(x))], x[ceiling((1 - alpha/2) * length(x))])
}

## Example 1: uncertainty of the sample mean
set.seed(12345678)

n = 50      # 50 is a small sample size!!
sigma = 1
x <- rnorm(n, mean = 30, sd = sigma)

# theoretical value
sigma^2 / n

est_boot <- sapply(dat_boot(x = x, nboots = 1000), FUN = mean)
str(est_boot)

##### ASSIGNMENT: a) Compute bootstrap variance of the mean
#                 b) Plot bootstrapped means
##### YOUR CODE STARTS HERE:




##### END OF CODE
####################################################


## Example 2: uncertainty of the sample variance

# theoretical value (given normality)
2 * sigma^4 / (n - 1)

est_boot <- sapply(dat_boot(x = x, nboots = 1000), FUN = var)
var(est_boot) # uncertainty of variance estimator
hist(est_boot)

# What if the sample size gets larger?


##### Poisson regression example
# Let's generate some data for a Poisson regression model
set.seed(12345)
n = 1000
beta0 = 1
beta1 = 0.5
beta2 = -0.5
dat <- data.frame(x = rnorm(n, mean = 3, sd = 0.5), z = rnorm(n, mean = 3, sd = 0.5))
head(dat)

lambda <- exp(beta0 + beta1 * dat$x + beta2 * dat$z) # the mean function
dat$y <- rpois(n = n, lambda = lambda) # the dep.var
head(dat)


# Let's get the "official" glm output for further reference
mod.glm <- glm(y ~ x + z, data = dat, family = poisson(link = 'log'))
coef(mod.glm)


# Let's bootstrap the data
dbooted <- data_frame_boot(data = dat, nboots = 1000)

coef_x_boot <- sapply(dbooted, function(k) 
  coef(glm(y ~ x + z, data = k, family = poisson(link = 'log')))[2])

boot_ci(x = coef_x_boot, alpha = 0.05)
 

# Compare to LR-based CI
confint(mod.glm)['x',]


#################### Parametric bootstrap
# Point estimates 
betas <- coef(mod.glm)
# Variance-covariance matrix for coef estimators
vc <- vcov(mod.glm)


B = 1000 # num of bootstrap samples
# sample parameter estimates from the asymptotic sampling distribution 

dat_boot_para <- function(data,beta,nboots){
  dat <- data
  lapply(1:nboots, function(k){
  lambda <- exp(beta[1]+dat$x*beta[2]+dat$z*beta[3]) 
  dat$y <- rpois(n = nrow(dat), lambda = lambda)
  return(dat)})
}
 

dbooted <- dat_boot_para(data = dat, nboots = 1000, beta=betas)

coef_x_boot <- sapply(dbooted, function(k) 
  coef(glm(y ~ x + z, data = k, family = poisson(link = 'log')))[2])

boot_ci(x = coef_x_boot, alpha = 0.05)


# Compare to LR-based CI
confint(mod.glm)['x',]


# CI
apply(betas_mat, 2, boot_ci)

t(confint(mod.glm)) # Compare to LR-based CI

# se
apply(betas_mat, 2, sd)

sqrt( diag(vcov(mod.glm)) ) # Compare to glm



#-------------------------------------------------
# Standard errors and CI for margial effects

# How would be get CI for marginal effects?


####################################################
##### LAB WORK: Get s.e. for marginal effects in a Poisson model with an interaction
#               lambda = beta0 + beta1 * x + beta2 * z + beta3 * xz
# NONPARAMETRIC BOOTSTRAP

set.seed(12345)
n = 1000
beta0 = 1
beta1 = 0.5
beta2 = -0.5
beta12 = 0.25

dat <- data.frame(x = rnorm(n, mean = 3, sd = 0.5), z = rnorm(n, mean = 3, sd = 0.5))
dat$xz <- dat$x * dat$z
lambda <- exp(beta0 + beta1 * dat$x + beta2 * dat$z + beta12 * dat$xz + rnorm(n, sd = 0.25))
dat$y <- rpois(n = n, lambda = lambda)

mef_inter <- function(data, betas, dep.var = 'y') {
  X <- cbind( 1, as.matrix( data[,-which(names(data) == dep.var)] ) )
  # X has 3 columns now: 1 = 1s; 2 = xs, 3 = zs
  # This function outputs a 2x1 vector with mef for x (1st element) and z (2nd element)
  storage <- numeric(length = 2)
  storage[1] <- mean(apply(X, 1, function(k) (betas[2] + betas[4] * k[3]) * exp( k %*% betas)))
  storage[2] <- mean(apply(X, 1, function(k) (betas[3] + betas[4] * k[2]) * exp( k %*% betas)))
  return(storage)
}

llik <- function(par, data) {
  - sum(data$y * (par[1] + par[2] * data$x + par[3] * dat$z + par[4]*dat$xz) ) + 
    sum( exp(par[1] + par[2] * dat$x + par[3] * dat$z + par[4]*dat$xz) ) 
}

boot_ci <- function(x, alpha = 0.05) {
  x <- x[which(!is.na(x) & !is.nan(x))] # remove NAs and NaNs
  x <- x[order(x)]
  c(x[ceiling(alpha/2 * length(x))], x[ceiling((1 - alpha/2) * length(x))])
}

##### YOUR CODE STARTS HERE:





##### END OF CODE
####################################################
