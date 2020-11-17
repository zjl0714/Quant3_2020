# Title:    Quant III (Lab 9)
# Name:     Junlong Aaron Zhou
# Date:     November 13, 2020
# Summary:  Mclust
#           Flexmix
#           Bayesian stats: Single-membership finite mixtures
#################################################################################


rm(list = ls())

library(rstan)
library(ggmcmc)
library(ggplot2)
library(MASS)
library(MCMCpack)
library(flexmix)
library(mclust)


setwd("~/Dropbox/Teaching/2020_Fall_Quant_3/lab9_mixture")


################################################################################
#################################     mclust     ###############################
################################################################################

set.seed(12345)

sigma1 <- matrix(c(1,0,0,1),2,2)
mu1 <- c(1,1)

sigma2 <- matrix(c(1,0.5,0.5,1),2,2)
mu2 <- c(-1,2)

y1 <- mvrnorm(1000, mu1,sigma1)
y2 <- mvrnorm(1000, mu2,sigma2)

y <- rbind(y1,y2)
group <- factor(c(rep(1,1000),rep(2,1000)))


# Look at the data

plot(y, col=group)
clPairs(y, group) # identical, but useful when higher dimension

hist(y[,1], freq = F, ylim=c(0,0.4)) 
lines(density(y[group==1,1]), col="red")
lines(density(y[group==2,1]), col="blue")


# Analysis
m <- Mclust(y, G=1:10)
summary(m, parameters = TRUE)

plot(m)

plot(m, what="BIC")
m$BIC
plot(m, what="density")


# Retrive membership

m$z
m$classification
table(m$classification)

member <- ifelse(m$z[,1]>m$z[,2],1,2)
which(member!=m$classification)

################################################################################
#################################    Flexmix     ###############################
################################################################################
set.seed(12345)

data("NPreg")
str(NPreg)

plot(NPreg$yn~NPreg$x, col = factor(NPreg$class)) 
# Truth:
# Class 1: y = 5x+e
# Class 2: y = 15+10x-x^2+e

m1 <- lm(yn~x, data=NPreg)
summary(m1)
abline(a = coef(m1)[1], b = coef(m1)[2])
table(NPreg$class)

m2 <- flexmix(yn~x, data=NPreg, k =2)
summary(m2)
m2
clusters(m2)
parameters(m2) 

# To get significance level for inference
rm2 <- refit(m2) 
summary(rm2)

# Adjust the model specification

m3 <- flexmix(yn~x + I(x^2), data=NPreg, k =2)
summary(m3)
m3

parameters(m3) 

rm3 <- refit(m3) 
summary(rm3)


# What if we think there are more cluster?

m4 <- flexmix(yn~x + I(x^2), data=NPreg, k =5)
summary(m4)
m4

parameters(m4) 

rm4 <- refit(m4) 
summary(rm4)

BIC(m4)
BIC(m3)


# What if we have repeated measure (e.g. TSCS?)
table(NPreg$class, clusters(m3))
str(NPreg$id2)

m5 <- flexmix(yn~x+I(x^2)|id2, data=NPreg, k=2)
summary(m5)
parameters(m5)
table(NPreg$class, clusters(m5))

# What's going on? 



# Many things you can adjust in Flexmix
# For for examples: https://cran.r-project.org/web/packages/flexmix/vignettes/flexmix-intro.pdf
# e.g. GLM is also compatible with Flexmix 

################################################################################
################################# NORMAL MIXTURE ###############################
################################################################################

#------------
rm(list = ls())

##### Generate data: mix N(1, 1), N(6, 1.5), and N(11, 2)
set.seed(12345)
mus <- c(1, 6, 11)
sigmas <- c(1, 1.5, 2)

N = 1000
z <- sample(x = 1:3, size = N, replace = T)
y <- rnorm(n = N, mean = mus[z], sd = sigmas[z])
hist(y, breaks = 200)
plot(density(y))



#------------
##### Define Stan program

mod.string <- '
data {
  int<lower=0> N ;
  int<lower=0> K ;
  real y[N] ;
}

parameters {
  simplex[K] lambda ;
  ordered[K] mu ; // to provide identification
  vector<lower=0>[K] sigma ;
}

model {
  vector[K] log_contrib ;
  // priors
  target += normal_lpdf(sigma | 0, 5) ; // sigma ~ normal(mu=0, sd=5)
  target += normal_lpdf(mu | 0, 5) ; // mu ~ normal(mu=0, sd=5)
  target += dirichlet_lpdf(lambda | rep_vector(2, K)) ; // lambda ~ dirichlet((2,2,...,2))
  
  // likelihood
  for (n in 1:N) {
    for (k in 1:K) {
      log_contrib[k] = log(lambda[k]) + normal_lpdf(y[n] | mu[k], sigma[k]) ;
    }
  target += log_sum_exp(log_contrib) ; // exponetiate, sum, take the log back
  }
}
'



#------------
##### HMC
# ETA: 2 min

# mod.fit <- rstan::stan(model_code = mod.string,
#                        data = list(N = length(y),
#                                    y = y,
#                                    K = 3), 
#                        cores = 3, chains = 3, 
#                        iter = 10000, warmup = 5000, thin = 3,
#                        seed = 12345)
 
# save(x = mod.fit, file = 'lab9_mod1.RData')
mod.fit <- get(load('lab9_mod1.RData'))


#------------
##### Diagnostics
ggob <- ggmcmc::ggs(rstan::As.mcmc.list(mod.fit), inc_warmup = F)
table(ggob$Parameter)


# Traceplots
ggmcmc::ggs_traceplot(ggob, family = 'lambda')
ggmcmc::ggs_traceplot(ggob, family = 'mu')
ggmcmc::ggs_traceplot(ggob, family = 'sigma')


# R-hats (Gelman and Rubin)
ggmcmc::ggs_Rhat(ggob)


# Running means
ggmcmc::ggs_running(ggob, family = 'lambda')
ggmcmc::ggs_running(ggob, family = 'mu')
ggmcmc::ggs_running(ggob, family = 'sigma')


# Compare densities
ggmcmc::ggs_density(ggob, family = 'lambda')
ggmcmc::ggs_density(ggob, family = 'mu')
ggmcmc::ggs_density(ggob, family = 'sigma')


# Geweke stats
ggmcmc::ggs_geweke(ggob)


# Bivariate distrib for parameters
params <- rstan::extract(mod.fit, inc_warmup = F)
names(params)
str(params$mu)

plot(params$mu[,1] ~ params$mu[,2])
plot(params$mu[,1] ~ params$mu[,3])
plot(params$mu[,2] ~ params$mu[,3])


# Posterior and prior
(sample.size = nrow(params$mu))
range(params$mu[,1], params$mu[,2], params$mu[,3], rnorm(n = sample.size, 0, 25)) # to see x-axis range 
str(params$mu[,1])

# Make a dataset for plotting prior and posterior together
# mu1, ..., mu3 define params; type defines whether it's prior or posterior
pos.prior.data <- data.frame(mu1 = c(params$mu[,1], rnorm(n = sample.size, 0, 25)),
                             mu2 = c(params$mu[,2], rnorm(n = sample.size, 0, 25)),
                             mu3 = c(params$mu[,3], rnorm(n = sample.size, 0, 25)),
                             type = rep(c('post', 'prior'), each = sample.size), 
                             stringsAsFactors = F)

gr.mu3 <- ggplot(data = pos.prior.data, aes(x = mu3, fill = type)) + 
  geom_density(alpha = 0.6) + 
  scale_x_continuous(limits = c(-20,20))
gr.mu3

gr.mu2 <- ggplot(data = pos.prior.data, aes(x = mu2, fill = type)) + 
  geom_density(alpha = 0.6) + 
  scale_x_continuous(limits = c(-20,20))
gr.mu2

gr.mu1 <- ggplot(data = pos.prior.data, aes(x = mu1, fill = type)) + 
  geom_density(alpha = 0.6) + 
  scale_x_continuous(limits = c(-20,20))
gr.mu1

# Takeaway: the posterior is super concetnrated in comparison to prior



#################################################################################
################################# LABEL SWITCHING ###############################
#################################################################################

# Use same data!



#------------
##### Define Stan program

mod.string.nonid <- '
data {
  int<lower=0> N ;
  int<lower=0> K ;
  real y[N] ;
}

parameters {
  simplex[K] lambda ;
  vector[K] mu ;          // NB! Not ordered here!
  vector<lower=0>[K] sigma ;
}

model {
  vector[K] log_contrib ;
  // priors
  target += normal_lpdf(sigma | 0, 5) ;
  target += normal_lpdf(mu | 0, 5) ;
  target += dirichlet_lpdf(lambda | rep_vector(2, K)) ;
  
  // likelihood
  for (n in 1:N) {
    for (k in 1:K) {
      log_contrib[k] = log(lambda[k]) + normal_lpdf(y[n] | mu[k], sigma[k]) ;
    }
  target += log_sum_exp(log_contrib) ;
  }
}
'


#------------
##### HMC
# ETA: 1.5 min

# mod.fit.nonid <- rstan::stan(model_code = mod.string.nonid,
#                              data = list(N = length(y),
#                                          y = y,
#                                          K = 3), 
#                              cores = 3, chains = 3, 
#                              iter = 10000, warmup = 5000, thin = 3,
#                              seed = 12345)
# save(x = mod.fit.nonid, file = 'lab9_mod2.RData')

mod.fit.nonid <- get( load('lab9_mod2.RData') )

#------------
##### Diagnostics
ggob.non <- ggmcmc::ggs(rstan::As.mcmc.list(mod.fit.nonid), inc_warmup = F)


# R-hats (Gelman and Rubin)
ggmcmc::ggs_Rhat(ggob.non)


# Traceplots
ggmcmc::ggs_traceplot(ggob.non, family = 'mu')


# Compare densities
ggmcmc::ggs_density(ggob.non, family = 'mu')


# Bivariate distrib for parameters
params <- rstan::extract(mod.fit.nonid, inc_warmup = F)

plot(params$mu[,1] ~ params$mu[,2])
plot(params$mu[,1] ~ params$mu[,3])
plot(params$mu[,2] ~ params$mu[,3])
# One cluster of points refers to one of the chains, the other - to other chains


#---------------
##### Takeaways:
# 1) Labels for mixture components are exchangeable --> non-identifiable model
# 2) Nonidentifiable model --> multimodality, chains can get stuck in a mode
# 3) One way to fix label switching: order one of the parameters (if it makes sense)


##### Question: Why ordering parameters is a better fix than imposing 
#               different informative priors?




####################################################################################
################################# DIFFERENT DISTRIBS ###############################
####################################################################################

rm(list = ls())

#------------
##### Generate data: mix LN(0.405, 0.693),
##### N(6, 1), N(13, 2), and N(18, 1)

set.seed(12345)
mus <- c(1.5, 6, 13, 18)
sigmas <- c(2, 1, 2, 1)

N = 1000
z <- sample(x = 1:4, size = N, replace = T, prob = c(0.4, 0.3, 0.2, 0.1))
y <- vector(length = N)
y[z == 1] <- rlnorm(n = length(y[z == 1]), meanlog = log(mus[1]), sdlog = log(sigmas[1]))
y[z == 2] <- rnorm(n = length(y[z == 2]), mean = mus[2], sd = sigmas[2])
y[z == 3] <- rnorm(n = length(y[z == 3]), mean = mus[3], sd = sigmas[3])
y[z == 4] <- rnorm(n = length(y[z == 4]), mean = mus[4], sd = sigmas[4])
hist(y, breaks = 200)
plot(density(y))
range(y)



#------------
##### Define Stan program

mod.string.dif <- '
data {
  int<lower=0> N ;
  int<lower=0> K ;
  real y[N] ;
}

parameters {
  simplex[K] lambda ;
  ordered[K] mu ;
  vector<lower=0>[K] sigma ;
}

model {
  vector[K] log_contrib ;
  // priors
  target += normal_lpdf(sigma | 0, 5) ;
  target += normal_lpdf(mu | 0, 5) ;
  target += dirichlet_lpdf(lambda | rep_vector(2, K)) ;
  
  // likelihood
  for (n in 1:N) {
    log_contrib[1] = log(lambda[1]) + lognormal_lpdf(y[n] | mu[1], sigma[1]) ; // lognormal component
    for (k in 2:(K)) {
      log_contrib[k] = log(lambda[k]) + normal_lpdf(y[n] | mu[k], sigma[k]) ; // normal components
    }
  target += log_sum_exp(log_contrib) ;
  }
}
'


#------------
##### HMC
# ETA: 2.5 min

# mod.fit.dif <- rstan::stan(model_code = mod.string.dif,
#                            data = list(N = length(y),
#                                        y = y,
#                                        K = 4), 
#                            cores = 3, chains = 3, 
#                            iter = 10000, warmup = 5000, thin = 3,
#                            seed = 12345)
# 
# save(x = mod.fit.dif, file = 'lab9_mod3.RData')

mod.fit.dif <- get(load('lab9_mod3.RData'))
#------------
##### Diagnostics
ggob.dif <- ggmcmc::ggs(rstan::As.mcmc.list(mod.fit.dif), inc_warmup = F)
table(ggob.dif$Parameter)


# R-hats (Gelman and Rubin)
ggmcmc::ggs_Rhat(ggob.dif)


# Geweke stats
ggmcmc::ggs_geweke(ggob.dif)


# Traceplots
ggmcmc::ggs_traceplot(ggob.dif, family = 'mu')
ggmcmc::ggs_traceplot(ggob.dif, family = 'sigma')
ggmcmc::ggs_traceplot(ggob.dif, family = 'lambda')


# Running means
ggmcmc::ggs_running(ggob.dif, family = 'mu')
ggmcmc::ggs_running(ggob.dif, family = 'sigma')
ggmcmc::ggs_running(ggob.dif, family = 'lambda')


# Compare densities
ggmcmc::ggs_density(ggob.dif, family = 'mu')
ggmcmc::ggs_density(ggob.dif, family = 'sigma')
ggmcmc::ggs_density(ggob.dif, family = 'lambda')




####################################################################################
############################### NON-SEPARABLE DISTRIBS #############################
####################################################################################

rm(list = ls())

#------------
##### Generate data: mix N(1, 2), N(2, 2), N(3, 2)

set.seed(12345)
mus <- c(1, 2, 3)
sigmas <- rep(2, 3)

N = 1000
z <- sample(x = 1:3, size = N, replace = T, prob = c(0.4, 0.3, 0.3))
y <- rnorm(N, mus[z], sigmas[z])
hist(y)
plot(density(y), ylim=c(0,0.3)) 
lines(density(y[z==1]),col="red")
lines(density(y[z==2]),col="green")
lines(density(y[z==3]),col="blue")
#------------
##### Define Stan program

mod.string.nonsep <- '
data {
  int<lower=0> N ;
  int<lower=0> K ;
  real y[N] ;
}

parameters {
  simplex[K] lambda ;
  ordered[K] mu ;
  vector<lower=0>[K] sigma ;
}

model {
  vector[K] log_contrib ;
  // priors
  target += normal_lpdf(sigma | 0, 5) ;
  target += normal_lpdf(mu | 0, 5) ;
  target += dirichlet_lpdf(lambda | rep_vector(2, K)) ;
  
  // likelihood
  for (n in 1:N) {
    for (k in 1:K) {
      log_contrib[k] = log(lambda[k]) + normal_lpdf(y[n] | mu[k], sigma[k]) ;
    }
    target += log_sum_exp(log_contrib) ;
  }
}
'



#------------
##### HMC
# ETA: 36 min
st <- proc.time()
#mod.fit.nonsep <- rstan::stan(model_code = mod.string.nonsep,
#                       data = list(N = length(y),
#                                   y = y,
#                                   K = 3),
#                       cores = 3, chains = 3,
#                       iter = 10000, warmup = 5000, thin = 3,
#                       seed = 12345,
#                       control = list(adapt_delta = 0.999, max_treedepth = 30))
#fin <- proc.time()
#(fin - st)/60

#save(x = mod.fit.nonsep, file = 'lab9_mod4.RData')

mod.fit.nonsep <- get( load('lab9_mod4.RData') )



#------------
##### Diagnostics
ggob.nonsep <- ggmcmc::ggs(rstan::As.mcmc.list(mod.fit.nonsep), inc_warmup = F)
table(ggob.nonsep$Parameter)


# R-hats (Gelman and Rubin)
ggmcmc::ggs_Rhat(ggob.nonsep)


# Geweke stats
ggmcmc::ggs_geweke(ggob.nonsep)


# Traceplots
ggmcmc::ggs_traceplot(ggob.nonsep, family = 'lambda')
ggmcmc::ggs_traceplot(ggob.nonsep, family = 'mu')
ggmcmc::ggs_traceplot(ggob.nonsep, family = 'sigma')


# Running means
ggmcmc::ggs_running(ggob.nonsep, family = 'lambda')
ggmcmc::ggs_running(ggob.nonsep, family = 'mu')
ggmcmc::ggs_running(ggob.nonsep, family = 'sigma')


# Compare densities
ggmcmc::ggs_density(ggob.nonsep, family = 'lambda')
ggmcmc::ggs_density(ggob.nonsep, family = 'mu')
ggmcmc::ggs_density(ggob.nonsep, family = 'sigma')


# Bivariate distrib for parameters
params <- rstan::extract(mod.fit.nonsep, inc_warmup = F)
names(params)
str(params$mu)

plot(params$mu[,1] ~ params$mu[,2])
plot(params$mu[,1] ~ params$mu[,3]) 
plot(params$mu[,2] ~ params$mu[,3])


# Posterior and prior
(sample.size = nrow(params$mu))
range(params$mu[,1], params$mu[,2], params$mu[,3]) # to see x-axis range 

pos.prior.data <- data.frame(mu1 = c(params$mu[,1], rnorm(n = sample.size, 0, 25)),
                             mu2 = c(params$mu[,2], rnorm(n = sample.size, 0, 25)),
                             mu3 = c(params$mu[,3], rnorm(n = sample.size, 0, 25)),
                             type = rep(c('post', 'prior'), each = sample.size), 
                             stringsAsFactors = F)

gr.mu1 <- ggplot(data = pos.prior.data, aes(x = mu1, fill = type)) + 
  geom_density(alpha = 0.6) + 
  scale_x_continuous(limits = c(-20,20))
gr.mu1

gr.mu2 <- ggplot(data = pos.prior.data, aes(x = mu2, fill = type)) + 
  geom_density(alpha = 0.6) + 
  scale_x_continuous(limits = c(-20,20))
gr.mu2

gr.mu3 <- ggplot(data = pos.prior.data, aes(x = mu3, fill = type)) + 
  geom_density(alpha = 0.6) + 
  scale_x_continuous(limits = c(-20,20))
gr.mu3


##### Takeaways:
# 1) If mixture components are too close, you might get into weird posterior geometry that is hard to explore
# 2) "I have decided that mixtures, like tequila, are inherently evil 
#     and should be avoided at all costs." (Larry Wasserman) URL: https://normaldeviate.wordpress.com/2012/08/04/mixture-models-the-twilight-zone-of-statistics/



######################################################################################
################################# MEMBERSHIP RETRIEVAL ###############################
######################################################################################

rm(list = ls())

#------------
##### Generate data: mix N(1, 1), N(6, 1.5), and N(11, 2)
set.seed(12345)
mus <- c(1, 6, 11)
sigmas <- c(1, 1.5, 2)

N = 1000
z <- sample(x = 1:3, size = N, replace = T)
y <- rnorm(N, mus[z], sigmas[z])
plot(density(y))



#------------
##### Define Stan program

mod.string.mem <- '
data {
  int<lower=0> N ;
  int<lower=0> K ;
  real y[N] ;
}

parameters {
  simplex[K] lambda ;
  ordered[K] mu ;
  vector<lower=0>[K] sigma ;
}

model {
  vector[K] log_contrib ;
  target += normal_lpdf(sigma | 0, 5) ;
  target += normal_lpdf(mu | 0, 5) ;
  for (n in 1:N) {
    for (k in 1:K) {
      log_contrib[k] = log(lambda[k]) + normal_lpdf(y[n] | mu[k], sigma[k]) ;
    }
    target += log_sum_exp(log_contrib) ;
  }
}

generated quantities {
  matrix[N, K] zet ;
  for (n in 1:N) {
    for (k in 1:K) {
      zet[n,k] = exp(normal_lpdf(y[n] | mu[k], sigma[k]) + log(lambda[k])) ;
    }
  }
}
'


#------------
##### HMC
# ETA: 1 min
# mod.fit.mem <- rstan::stan(model_code = mod.string.mem,
#                        data = list(N = length(y),
#                                    y = y,
#                                    K = 3), 
#                        cores = 3, chains = 3, 
#                        iter = 10000, warmup = 5000, thin = 3,
#                        seed = 12345)
# save(x = mod.fit.mem, file = 'lab9_mod5.RData')
mod.fit.mem <- get( load('lab9_mod5.RData') )



#------------
##### Extract and process class memberships 
params <- rstan::extract(mod.fit.mem, inc_warmup = F)
str(params$zet) 
# dim1: different samples from posterior
# dim2: obs 
# dim3: different mixture components

# For a given posterior sample, loop over obs and find component (column) 
#                                               with the largest zet prob

z.pred <- lapply(1:dim(params$zet)[1], function(k) apply(params$zet[k,,], 1, which.max))
z.pred <- do.call(rbind, z.pred)
str(z.pred)

# manual inspection
k = 600 # 800
z[k] # true class 
table(z.pred[,k]) # predicted classes

# summarize quality of clustering
# for a given ob k: table cluster assignments and compute the number of correct ones (== z[k])
errors <- sapply(1:1000, function(k) 
  1 - prop.table(table(z.pred[,k]))[ as.numeric( names( table(z.pred[,k]) ) ) == z[k] ]
)

errors <- unlist(errors)
range(errors)
hist(errors, breaks = 100)
mean(errors)

# All information about class memberships is in z.pred
str(z.pred) # columns are obs, rows are samples
# find modal class for each ob (col) across samples (rows)
z.pred.point <- sapply(1:ncol(z.pred), function(k) 
  names( table(z.pred[,k]) )[ which.max( table(z.pred[,k]) ) ] 
  )

str(z.pred.point)
table(z, z.pred.point)





######################################################################################
################################# Optional ###############################
######################################################################################

rm(list =ls())
data("NPreg")
d <- NPreg


##### Define Stan program

mod.string.reg <- '
data {
  int<lower=0> N ;
  int<lower=0> K ;
  real x[N] ;
  real y[N] ;
}

parameters {
  simplex[K] lambda ;
  ordered[K] alpha2 ; 
  real alpha1[K] ;
  real alpha0[K] ; 
  vector<lower=0>[K] sigma ;
}

model {
  vector[K] log_contrib ;
  target += normal_lpdf(sigma | 0, 5) ; 
  target += normal_lpdf(alpha0 | 0, 5) ;
  target += normal_lpdf(alpha1 | 0, 5) ;
  target += normal_lpdf(alpha2 | 0, 5) ;
  // your code starts here
  
  
  
  
  
  
  //end
}
'

d_u <- list(N = nrow(d),
            K = 2,
            y = d$yn,
            x = d$x)

# st <- proc.time()
# set.seed(12345)
# mod.fit.reg <- rstan::stan(model_code = mod.string.reg,
#                        data = d_u,
#                        cores = 3, chains = 3,
#                        iter = 10000, warmup = 5000, thin = 3, 
#                        control = list(adapt_delta = 0.999, max_treedepth = 30))
# fin <- proc.time()
# (fin - st)/60
# 
# save(x = mod.fit.reg, file = 'lab9_mod6.RData')
mod.fit.reg <- get( load('lab9_mod6.RData') ) 


# Recall the truth:
# Class 1: y = 5x+e
# Class 2: y = 15+10x-x^2+e


params <- rstan::extract(mod.fit.reg, inc_warmup=F, permu=F)
str(params)
ggob.nonsep <- ggmcmc::ggs(rstan::As.mcmc.list(mod.fit.reg), inc_warmup = F)

ggmcmc::ggs_traceplot(ggob.nonsep, family = "lambda")
ggmcmc::ggs_traceplot(ggob.nonsep, family = "alpha0")
ggmcmc::ggs_traceplot(ggob.nonsep, family = "alpha1")
ggmcmc::ggs_traceplot(ggob.nonsep, family = "alpha2")

param.chain1 <- params[,1,] 
mean(param.chain1[,"alpha0[2]"])
c(quantile(param.chain1[,"alpha0[2]"],0.025),
  quantile(param.chain1[,"alpha0[2]"],0.975))
mean(param.chain1[,"alpha1[2]"])
c(quantile(param.chain1[,"alpha1[2]"],0.025),
  quantile(param.chain1[,"alpha1[2]"],0.975))
mean(param.chain1[,"alpha2[2]"])
c(quantile(param.chain1[,"alpha2[2]"],0.025),
  quantile(param.chain1[,"alpha2[2]"],0.975))

# What happen to chain 1?


plot(NPreg$yn~NPreg$x, col = factor(NPreg$class)) 
curve(mean(param.chain1[,"alpha0[2]"])+mean(param.chain1[,"alpha1[2]"])*x+
        mean(param.chain1[,"alpha2[2]"])*x^2, add = T)
curve(mean(param.chain1[,"alpha0[1]"])+mean(param.chain1[,"alpha1[1]"])*x+
        mean(param.chain1[,"alpha2[1]"])*x^2, add = T)
