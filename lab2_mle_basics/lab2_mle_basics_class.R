# Title:    Quant III (Lab 2)
# Name:     Junlong Aaron Zhou
# Date:     September 18, 2020
# Summary:  Computational MLE (gradient ascent/descent) 
#           MLE example (poisson regression): estimation and inference (CI)
#################################################################################

rm(list = ls())

# install.packages("mfx")
# install.packages('mvtnorm')
library(mfx)
library(mvtnorm)
library(MASS) # for confint

##################################### Gradient descent / ascent #####################################
#####################################################################################################

##### Use GA to estimate the mean of a normal distribution

x <- rnorm(100, mean = 10)

GA_mean_norm <- function(x, nit = 10000, eps = 0.001) {
  # Gradient ascent to estimate theta in N(theta, sigma)
  # Parameter initialization
  theta <- rnorm(1) #random starting point
  loglik_grad <- function(theta) { sum(x - theta) }
  # Update the gradient
  for (i in 1:nit) {
    theta <- theta + eps * loglik_grad(theta)
  }
  return(theta)
}

GA_mean_norm(x)
mean(x)


##### Use GD to estimate the mean of an unknown distribution (no MLE!)
rm(list = ls())

set.seed(111222)

x <- rnorm(100, mean = 10)
# x <- rlnorm(100, meanlog = 10)

GD_mean <- function(x, loss = 'L2', eps = 0.001, nit = 1000) {
  # Gradient descent for estimating the mean for an unknown distribution
  # Loss: quadratic (L2) or L1
  # Parameter initialization
  theta <- rnorm(1)
  
  loglik_grad <- ifelse(loss == 'L2', function(theta)  - sum(x - theta), 
                        ifelse(loss == 'L1', function(theta)  sum(-sign(x - theta)) , NULL))
  if (!is.null(loglik_grad)) {
    # iterate
    for (i in 1:nit) {
      theta <- theta - eps * loglik_grad(theta)
    }
  } else {
    print('Wrong loss')
    theta <- NULL
  }
  return(theta)
}


mean(x)
GD_mean(x)

median(x)
GD_mean(x = x, loss = 'L1', eps = 0.001, nit = 500000)



##################################### Poisson regression: estimation #####################################
##########################################################################################################

rm(list = ls())

##### Poisson regression with GA
GD_pois_reg <- function(X, y, nit = 10000, eps = 0.001) {
  ### Gradient ascent optimization for Poisson regression 
  # set constants
  n = nrow(X); p = ncol(X)
  # initialize betas and ll_grad
  betas <- rnorm(p)
  ll_grad <- numeric(p)
  # iterate and perform gradient descent
  for (it in 1:nit) {
    # compute gradient with the current values of beta
    ll_grad <- sapply(1:p, function(k) ll_grad[k] <- sum(X[,k] * y) - sum( X[,k] * exp(X %*% betas)  ) )
    # update betas
    betas <- betas + eps/sqrt(sum(ll_grad^2)) * ll_grad
  }
  return(betas)
}


### Generate data
set.seed(12345)
n = 1000
beta0 = 1
beta1 = 0.5
beta2 = -0.5
dat <- data.frame(x = rnorm(n, mean = 3, sd = 0.5), z = rnorm(n, mean = 3, sd = 0.5))
head(dat)

lambda <- exp(beta0 + beta1 * dat$x + beta2 * dat$z)
dat$y <- rpois(n = n, lambda = lambda)
head(dat)


##### "Official" R output for a Poisson model
mod.glm <- glm(y ~ x + z, data = dat, family = poisson(link = 'log'))
coef(mod.glm)

# Get parameter estimates via GA
(betas <- GD_pois_reg(X = cbind(1, dat$x, dat$z), y = dat$y))


# What about st.errors?
vcov(mod.glm)

# Now, let's get se for GA estimates 
X = cbind(1, dat$x, dat$z)
h11 <- - sum( apply(X, 1, function(k) exp( betas %*% k) ) )
h22 <- - sum( apply(X, 1, function(k) k[2]^2 * exp(betas %*% k) ) )
h33 <- - sum( apply(X, 1, function(k) k[3]^2 * exp(betas %*% k) ) )
h12 = h21 <- - sum( apply(X, 1, function(k) k[2] * exp(betas %*% k) ) )
h13 = h31 <- - sum( apply(X, 1, function(k) k[3] * exp(betas %*% k) ) )
h23 = h32 <- - sum( apply(X, 1, function(k) k[3] * k[2] * exp(betas %*% k) ) )
H <- matrix(c(h11, h12, h13, h21, h22, h23, h31, h32, h33), nrow = 3, ncol = 3, byrow = T)
vc <- -solve(H)
sqrt(diag(vc)) 

# Compare to GLM
sqrt(diag(vcov(mod.glm)))





##################################### optim function #####################################
##########################################################################################################


# Rewrite the function to get mean:


mu <- 1
sigma <- 5

X_test <- rnorm(1000,mu,sigma)

get_mean <- function(par, x){
  mu <- par[1]
  sigma <- par[2]
  -sum(-log(sigma)-(x-mu)^2/(2*sigma^2))
}

optim(par=c(1,2), fn = get_mean, x=X_test)
  

#--------------------------------





##### LAB WORK
##### Poisson regression with optim
##### ASSIGNMENT: Implement the model with optim() and find variance of MLE estimates



##### YOUR CODE STARTS HERE:




##### END OF CODE
####################################################

  
# retrieve parameter estimates
optim.mod$par 
# get s.e. from the inverted Hessian
sqrt( diag(solve(optim.mod$hessian)) )

##################################### Poisson regression: inference #####################################
#########################################################################################################


# Not this time

confint(mod.glm) # LR-based CI
confint.default(mod.glm) # Wald-test based

# Let's unpack Wald-test based CI
coef(mod.glm)['x'] + c(-1.96, 1.96) * sqrt( vcov(mod.glm)[2,2] )

# Is it exactly the same?
# No if used c(-1.96, 1.96). For exact: qnorm(0.975)

coef(mod.glm)['x'] + c(qnorm(0.025), qnorm(0.975)) * sqrt( vcov(mod.glm)[2,2] )


#------------------------
##### Takeaways:
# 1) Gradient descent (for a loss function) / ascent (for log lik) is a generic numerical optimization
#     method widely used in computational stats and data science
# 2) GD/GA does NOT produce uncertainty estimates. Need to use MLE theory or bootstrap








