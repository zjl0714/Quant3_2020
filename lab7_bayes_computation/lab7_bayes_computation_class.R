# Title:    Quant III (Lab 7)
# Name:     Junlong Aaron Zhou
# Date:     October 30, 2020
# Summary:  Stan (RStan)
#           Bayesian stats: linear, logistic, hierarchical models
#           Bayesian stats: convergence diagnostics
#################################################################################

rm(list = ls())

# install.packages(c("rstan","bayesplot","mvtnorm","coda","ggmcmc"))

library(rstan)
library(ggmcmc)
library(coda)
library(mvtnorm)
library(e1071) # for skewness
library(lme4)

setwd("~/Dropbox/Teaching/2020_Fall_Quant_3/lab7_bayes_computation")

#######################################################################################
#                                     Bernoulli Probability
#######################################################################################
 

##### Generate data
set.seed(12345)

p = 0.4
N = 15
(y <- rbinom(n = N, size = 1, prob = p))  # our data
(z <- sum(y))      # number of successes
cat('MLE estimate is z / N = ', round(z / N, 3), '\n')



##### Theoretical result
# Prior: Beta(1,1), i.e. uniform
# Theoretical result: p(theta | y) = Beta(z+1, N-z+1)
cat('Theoretical posterior expectation of theta is ', round((z+1) / (z+1 + N-z+1), 4) )


##### Define and run a Stan model

modstring <- '
data {
  int<lower=0> N ;
  int y[N] ;
}

parameters {
  real<lower=0, upper=1> theta;
  // define parameter theta here
}

model {
  theta ~ beta(1, 1) ;
  y ~ bernoulli(theta) ;
}
'

modfit <- stan(model_code = modstring, data = list(N = N, y = y),
               chains = 3, iter = 5000, warmup = 1000, thin = 1, seed = 12345)
# Don't worry about chains, warmup, and thinning for the moment
# The output contains samples from the posterior. That's what we need!
# How many samples? chains*(iter - warmup) = 3*(5000-1000) = 12000

# NB! no factors! convert to integers
# Stan cannot handle missing values in data automatically, so no element of the data can contain NA values.

class(modfit) # stanfit

# Sample from the posterior
postsample <- rstan::extract(modfit)
str(postsample)


# plot the posterior
par(mfrow = c(1,2))
plot(density(postsample$theta), main = 'Posterior distribution', xlab = 'Parameter values')
hist(postsample$theta, breaks = 200, main = 'Posterior distribution', xlab = 'Parameter values')
par(mfrow = c(1,1))

# do you desperately want a point estimate?
mean(postsample$theta)


####################################################
##### LAB WORK: Re-write modstring from above using target += syntax
# 
##### YOUR CODE STARTS HERE:


modstring_target <- '
data {
  // Define your data
}

parameters {
   // Define theta
}

model {
  // remember y is a vector! And Bernoulli is a discrete distribution!
}
'
 
modfit <- stan(model_code = modstring_target, data = list(N = N, y = y),
               chains = 3, iter = 5000, warmup = 1000, thin = 1, seed = 12345)
postsample <- rstan::extract(modfit)
mean(postsample$theta)

par(mfrow = c(1,2))
plot(density(postsample$theta), main = 'Posterior distribution', xlab = 'Parameter values')
hist(postsample$theta, breaks = 200, main = 'Posterior distribution', xlab = 'Parameter values')
par(mfrow = c(1,1))

##### END OF CODE
####################################################




#######################################################################################
#                                     Logistic Regression
#######################################################################################

rm(list = ls())

##### Generate data
set.seed(1234567)

N = 1000
p = 4

# make a covariance matrix
s <- matrix(rnorm(p*p), nrow = p, ncol = p)
s <- t(s) %*% s
s <- s/(max(s))

X <- mvtnorm::rmvnorm(N, mean = rep(5, p), sigma = s)
det(cor(X))
X.design <- cbind(1, X)

betas <- c(-2, -2, 1, -4, 5)

y <- rbinom(n = N, size = 1, prob = 1 / (1 + exp(- X.design %*% betas)))
table(y)



##### MLE (logistic regression)
summary(glm(y ~ X, family = binomial(link = 'logit')))



##### Define and run a Stan model
modstr_logistic <- '
data {
  int<lower=0> N ;             // Number of observations
  int<lower=1> p ;             // Number of parameter
  matrix[N, p] X ;             // Define covariates X
  int<lower=0, upper=1> y[N] ; // Define data Y 
}

parameters {
  vector[p] b ;                // Define beta 
}

model {
  b ~ cauchy(0, 5) ;           
  y ~ bernoulli_logit(X * b) ;
}
'

modfit <- stan(model_code = modstr_logistic, 
               data = list('N' = nrow(X.design), 
                           'p' = ncol(X.design), 
                           'X' = X.design, 
                           'y' = y),
               chains = 3, iter = 2000, warmup = 1000, thin = 1, cores = 3)


##### Sample from the posterior
postsample <- rstan::extract(modfit)
str(postsample$b)
apply(postsample$b, 2, mean)
apply(postsample$b, 2, sd)




#######################################################################################
#                                     Linear Regression
#######################################################################################

 
rm(list = ls())

##### Generate data
set.seed(1234567)

N = 1000
p = 4

# make a covariance matrix
s <- matrix(rnorm(p*p), nrow = p, ncol = p)
s <- t(s) %*% s
s <- s/(max(s) + 0.1)

X <- mvtnorm::rmvnorm(N, mean = rep(5, p), sigma = s)
X.design <- cbind(1, X)

betas <- c(-2, 3, 1, 4, 5)

y <- X.design %*% betas + rnorm(N)
y <- y[,1]
str(y)


##### OLS
summary(lm(y ~ X))


##### Define and run a Stan model
modstr_reg <- '
data {
  int p ;
  int N ;
  matrix[N, p] X ;
  vector[N] y;
}

parameters {
  vector[p] b ;  // define b
  real<lower=0> sigma ; // define sigma
}

model {
  // Priors
  b ~ cauchy(0, 2.5) ;
  sigma ~ cauchy(0, 2.5) ;
  
  // Likelihood
  y ~ normal(X*b, sigma);
  // define likelihood
}
'

modfit <- stan(model_code = modstr_reg, 
               data = list('p' = ncol(X.design), 
                           'N' = nrow(X.design), 
                           'X' = X.design, 
                           'y' = y), 
               chains = 3, iter = 4000, warmup = 1000, thin = 3, cores = 3)


##### Sample from the posterior
postsample <- rstan::extract(modfit)
str(postsample$b)
apply(postsample$b, 2, mean)
apply(postsample$b, 2, sd)

# recall our ols:
coef(lm(y ~ X))

##### Convergence diagnostics (1): traceplot
rstan::traceplot(modfit, pars = 'b')


##### If there are not too many parameters: 
postsample.coda <- rstan::extract(modfit, permuted = F, inc_warmup = F)
str(postsample.coda) # it's a tensor: nrow = num.samples, ncol = num.chains, 3rd.dim = num.par
# List elements correspond to different chains
coda.obj <- coda::mcmc.list( lapply( 1:ncol(postsample.coda), function(k) coda::mcmc(postsample.coda[,k,]) ) )
geweke.diag(coda.obj) # t-test for first 10% and last 50% of sampled values


##### If there are too many parameters, ggmcmc graphs are particularly useful for diagnostics
# More about this here: http://xavier-fim.net/packages/ggmcmc/
ggmcmc.b.obj <- ggmcmc::ggs(rstan::As.mcmc.list(modfit, pars = c("b")),
                            inc_warmup = F)
# ggmcmc::ggs creates a ggmcmc object out of a coda object returned by rstan::As.mcmc.list

# Rhat (Gelman and Rubin): potential variance reduction factor
ggmcmc::ggs_Rhat(ggmcmc.b.obj, family = 'b')

# Geweke: t-tests; defaults are first 0.1 and last 0.5
ggmcmc::ggs_geweke(ggmcmc.b.obj, frac1 = 0.1, frac2 = 0.5)

# Now, inspect convergence parameter after parameter. Have fun! :-) 
ggmcmc::ggs_density(ggmcmc.b.obj, family = 'b')
# These are densities from 3 chains. Should look similar.

# Compare whole density to density of the last k% of iterations
ggmcmc::ggs_compare_partial(ggmcmc.b.obj, family = 'b', partial = 0.1)

# Another way to see a traceplot graph
ggmcmc::ggs_traceplot(ggmcmc.b.obj, family = 'b', original_burnin = T)

ggmcmc::ggs_running(ggmcmc.b.obj, family = 'b')
# These are dynamics of sample means as new iterations arrive
# Convergence:  1) three horizontal lines at the same level
#               2) means balancing around horizontal lines
# Here: a clear failure:  1) horizontal lines at different levels
#                         2) running means do not balance around lines






####################################################################
############################### TSCS ###############################
####################################################################
# Using data from BDA3, section 15.2, pp.383-388
# Predicting presidential election outcomes in the U.S. in 1948-1988 



#-------------------
##### Data preparation.
# Raw data available at: http://www.stat.columbia.edu/~gelman/book/data/presidential.asc
d <- read.table(file = 'forecast.txt', header = T, sep = ' ', stringsAsFactors = F)

#--- Nationwide variables:
# n1 = Support for Dem. candidate in Sept. poll
# n2 = (Presidential approval in July poll) × Inc
# n3 = (Presidential approval in July poll) × Presinc
# n4 = (2nd quarter GNP growth) × Inc
#
#--- Statewide variables:
# s1 = Dem. share of state vote in last election
# s2 = Dem. share of state vote two elections ago
# s3 = Home states of presidential candidates
# s4 = Home states of vice-presidential candidates
# s5 = Democratic majority in the state legislature
# s6 = (State economic growth in past year) × Inc
# s7 = Measure of state ideology
# s8 = Ideological compatibility with candidates
# s9 = Proportion Catholic in 1960 (compared to U.S. avg.)
#
#--- Regional/subregional variables:
# r1 = South
# r2 = (South in 1964) × (−1)
# r3 = (Deep South in 1964) × (−1)
# r4 = New England in 1964
# r5 = North Central in 1972
# r6 = (West in 1976) × (−1)
#
#--- Regions:
# South:  1  4  9 10 17 18 24 33 36 40 42 43 46
# Northeast:  7  8 19 20 21 29 30 32 38 39 45 48
# Midwest:  13 14 15 16 22 23 25 27 34 35 41 49
# West:  2  3  5  6 11 12 26 28 31 37 44 47 50
#

stat <- data.frame("num" = 1:50, 'stat.name' = state.name, 'stat.abb' = state.abb, stringsAsFactors = F)
stat$region <- 'South'
stat$region[c(7, 8, 19, 20, 21, 29, 30, 32, 38, 39, 45, 48)] <- 'Northeast'
stat$region[c(13, 14, 15, 16, 22, 23, 25, 27, 34, 35, 41, 49)] <- 'Midwest'
stat$region[c(2, 3, 5, 6, 11, 12, 26, 28, 31, 37, 44, 47, 50)] <- 'West'

d <- merge(d, stat, by.x = 'state', by.y = 'num')
d <- na.omit(d)

years <- unique(d$year)
years <- years[years != 1992]
 

save(x = d, file = 'gelman_15_2.RData')


#--------------------
##### Load data
rm(list = ls())

d <- get(load("gelman_15_2.RData"))
str(d)

dat.obs <- subset(d, year != 1992)



#---------------------
##### Pre-analysis visualization
#--- Fig 15.1-a
par(mfrow = c(1,2))
d.88.84 <- subset(dat.obs, year == 1988 | year == 1984)
d.76.72 <- subset(dat.obs, year == 1976 | year == 1972)
d.76.72 <- subset(d.76.72, stat.abb != 'AL')

plot(subset(d.88.84, year == 1988)$Dvote ~  subset(d.88.84, year == 1984)$Dvote, 
     xlab = 'Dem vote by state, 1984', ylab = 'Dem vote by state, 1988', 
     xaxt = 'n', yaxt = 'n', ylim = c(0.28, 0.62), type = 'n')
axis(1, at = c(0.25, 0.35, 0.45))
axis(2, at = seq(0.3, 0.6, 0.1))
text(subset(d.88.84, year == 1988)$Dvote ~  subset(d.88.84, year == 1984)$Dvote, 
     labels = subset(d.88.84, year == 1988)$stat.abb)

#--- Fig 15.1-b
plot(subset(d.76.72, year == 1976)$Dvote ~  subset(d.76.72, year == 1972)$Dvote, 
     xlab = 'Dem vote by state, 1972', ylab = 'Dem vote by state, 1976', 
     xaxt = 'n', yaxt = 'n', ylim = c(0.3, 0.72), xlim = c(0.15, 0.62), type = 'n')
axis(1, at = seq(0.2, 0.6, 0.1))
axis(2, at = seq(0.3, 0.7, 0.1))
text(subset(d.76.72, year == 1976)$Dvote ~  subset(d.76.72, year == 1972)$Dvote, 
     labels = subset(d.76.72, year == 1976)$stat.abb)
par(mfrow = c(1,1))

rm(d.76.72, d.88.84)

# Takeaways:
# 1) Overall a strong pattern
# 2) But with outliers: 
#   "Georgia (‘GA’), the home state of Jimmy Carter, the Democratic candidate in 1976." (p.384)

 

######### Hierarchical analysis

### Fit Bayesian model
mod.string.h <- '
data {
  int N ;
  int p_exp ;
  int p_years ;
  int p_s ;
  // int p_nots_y ;
  matrix[N,p_exp] X_exp ;
  int X_years[N];
  int  X_s [N] ;
  // int X_nots_y[N] ;
  real y[N] ;
}

parameters {
  real a ;
  vector[p_exp] b_exp ;
  vector[p_years] b_years ;
  vector[p_s] b_s ;
  real<lower=0> sigma ;
  real<lower=0> tau_b_exp;
  real<lower=0> tau_b_years ;
  real<lower=0> tau_b_s ;
}

model {
  // hyperpriors
  sigma ~ uniform(0, 10) ;
  tau_b_exp ~ uniform(0, 1) ;
  tau_b_years ~ uniform(0, 10) ;
  tau_b_s ~ uniform(0, 10) ;
  
  // priors
  a ~ normal(0, 5) ;
  b_exp ~ normal(0, tau_b_exp) ;
  b_years ~ normal(0, tau_b_years) ;
  b_s ~ normal(0, tau_b_s) ;
  
  // likelihood
  for (n in 1:N){
    y[n] ~ normal( a + X_exp[n, ] * b_exp +
    b_years[X_years[n]] + b_s[X_s[n]] , sigma);
  }
}

generated quantities {
  vector[N] yrep ;
  for (n in 1:N) {
    yrep[n] = normal_rng( a + X_exp[n,] * b_exp +
    b_years[X_years[n]] + b_s[X_s[n]] , sigma ) ;
  }
}
'

#--------------
#### Select data

d.exp.obs <- dat.obs[,  c("s1","s2","s3","s4")]
d.years.obs <- as.integer(as.factor(dat.obs$year))
d.s  <-as.integer(as.factor(dat.obs$state)) 

# Make data list for rstan
dat.h <- list(N = nrow(d.exp.obs), 
              p_exp = ncol(d.exp.obs), 
              p_years = length(unique(d.years.obs)), 
              p_s = length(unique(d.s)), 
              X_exp = d.exp.obs, 
              X_years = d.years.obs, 
              X_s = d.s,  
              y = dat.obs$Dvote)
 
# # Estimate

parallel::detectCores()

# t1 <- Sys.time()
# mod.h.bayes <- stan(model_code = mod.string.h, 
#                     data = dat.h,
#                     chains = 3, iter = 10000,  thin = 2, 
#                     cores = parallel::detectCores(),
#                     control = list(adapt_delta = 0.999,
#                                    max_treedepth = 15))
# Sys.time()-t1

# Takes 20 min on my machine (4 cores)
# save(x = mod.h.bayes, file = 'multi_mod_h_20th.RData')

mod.h.bayes <- get(load("multi_mod_h_20th.RData"))


m1 <- lmer(Dvote ~  1+s1+s2+s3+s4 +(1|state)+(1|year),
           data=dat.obs) 

### Sample from the posterior

params <- rstan::extract(mod.h.bayes)
rstan::traceplot(mod.h.bayes, pars="b_exp")
str(params)

### Compare Fixed effect
str(params$b_exp)
apply(params$b_exp,2,mean)

fixef(m1)

### What is year random effect? 

str(params$b_years)
hist(params$b_years[,10])
coda::HPDinterval(as.mcmc(params$b_years[,10]))

# Which year? 

levels(as.factor(dat.obs$year))

ranef(m1)$year
apply(params$b_years,2,mean)

### Make predictions
ypred <- rep(NA, nrow(dat.obs)) 
for (j in (1:nrow(dat.obs))){
  ypred[j] <- mean(params$a) + colMeans(params$b_exp) %*% t(d.exp.obs[j,]) +
      mean(params$b_s[,d.s[j]]) + mean(params$b_years[,d.years.obs[j]])
}
 



# How good is the model fit
plot(dat.obs$Dvote ~ ypred, main = 'Model fit1', 
     xlab = 'Predicted results', ylab = 'Observed results', 
     xlim = c(0.2,0.9), ylim = c(0.2,0.9))

# But we already have generated quantities!
ypredd <- params$yrep

plot(dat.obs$Dvote ~ apply(ypredd, 2, mean), main = 'Model fit2', 
     xlab = 'Predicted results', ylab = 'Observed results', 
     xlim = c(0.2,0.9), ylim = c(0.2,0.9)) 

# Note there is a small difference but they are basically the same

plot(ypred ~ apply(ypredd, 2, mean), main = 'Check Predication', 
     xlab = 'Predicted results from Stan', ylab = 'Predicted results using posterior mean', 
     xlim = c(0.2,0.9), ylim = c(0.2,0.9)) 
abline(a=0,b=1,col="red")

##### Takeaways:
# 1) Hierarchical models are straightforward to set up and estimate with Stan
# 2) Doing Bayesian stats with Stan is fun! :-)  

