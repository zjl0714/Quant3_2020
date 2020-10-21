# Title:    Quant III (Lab 4)
# Name:     Junlong Aaron Zhou
# Date:     October 2, 2020
# Summary:  AIC, BIC, ROC
#           Binary regressions: Regularization for separation
#           Some overdispersion
#################################################################################

rm(list=ls())
#install.packages("lmtest")
#install.packages("logistf")
library(logistf)
library(lmtest)
library(dplyr)
library(pROC)



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
 
AIC(mod.res)
AIC(mod.unres)



bic.unres <- 2*optim.mod.unres$value+3*log(nrow(dat))
bic.res <- 2*optim.mod.res$value+2*log(nrow(dat))

bic.res-bic.unres

BIC(mod.res)
logLik(mod.res)
BIC(mod.unres) - BIC(mod.res)


##### Bernouli regression example


set.seed(12345)
n = 1000
beta0 = 1
beta1 = 1
beta2 = -1
dat <- data.frame(x = rnorm(n, mean = 3, sd = 0.5), z = rnorm(n, mean = 3, sd = 0.5))

p <- 1/(1+exp(-beta0 - beta1 * dat$x - beta2 * dat$z)) # log link
dat$y <- rbinom(n = n, size = 1, prob = p) # the dep.var

mod.unres <- glm(y~x+z, family = binomial(link="logit"), data= dat)
mod.res <- glm(y~x, family = binomial(link="logit"), data= dat)

####################################################
##### LAB WORK: Compare two model using BIC
#               1. Use BIC command  
#               2. Write BIC calculation on your own: 
#                 2a. write likelihood function and use optim function
#                 2b. calculate BIC using likelihood value from 2a.
##### YOUR CODE STARTS HERE:

BIC(mod.res) #1139.93
BIC(mod.unres) #1101.091

llik_b = function(par, x, y, z){
  pi = pnorm((par[1] + par[2]*x + par[3]*z))
  loglik = sum(y*log(pi) + (1-y)*log(1-pi))
  return(-loglik)
}

optim(par = c(0,0,0), x = dat$x, y = dat$y, z = dat$z, fn = llik_b)







##### END OF CODE
####################################################




# Now, using ROC to compare model


yhat.unres <- predict(mod.unres, type="response")
yhat.res <-  predict(mod.res, type="response")

roc.unres <- roc(dat$y, yhat.unres)
roc.unres

roc.res <- roc(dat$y, yhat.res)
roc.res 


plot(roc.unres)
lines(roc.res, col="grey") 


plot(x= (1-roc.unres$specificities), y=roc.unres$sensitivities,
     xlab = "False Alarm Rate", ylab="Hit Rate", lwd=1,
     type = "l")
lines(1-roc.res$specificities, roc.res$sensitivities, col = "grey", lwd=1)
abline(a=0,b=1)

########################################################################
#                                    SEPARATION  
########################################################################

 

rm(list = ls())

##### Consider a binary logistic model
# Check the loglik
x1 <- rnorm(1000); x2 <- rnorm(1000)
ystar <- 1 + 2 * x1 + 3 * x2 + rnorm(1000)
y <- ifelse(ystar > mean(ystar), 1, 0)

llik <- function(beta) {
  - sum( dbinom(x = y, size = 1, 
                prob = 1 / (1 + exp(-beta[1] - beta[2] * x1 - beta[3] * x2)), 
                log = T) )
}

optim(rnorm(3), llik, control = list(maxit = 10000))$par
(coef(mod1 <- glm(y~x1+x2, family=binomial(link="logit"))))


# OK, loglik is correct.



# Now, generate complete separation

set.seed(98765)
x1 <- rnorm(1000); x2 <- rnorm(1000)
y <- numeric(length = 1000)
y[x2 > quantile(x2, probs = 0.6)] <- 1
y[x2 <= quantile(x2, probs = 0.6)] <- 0

table(y)

plot(y~x1)
plot(y~x2)
abline(v=quantile(x2, probs = 0.6), col = "red")

coef(mod2<-glm(y ~ x1 + x2, family = binomial(link = 'logit')))

for (num in c(10, 20, 50,100,500,1000,10000)) {
  beta <- (mod.lik <- optim(rnorm(3), llik, control = list(maxit = num)))$par
  cat(num, 'iter; parameter = ', beta[3], '\n')
  dat.plot <- data.frame(x1=mean(x1), x2=seq(-5,5,by=0.05)) %>%
    mutate(y = 1 / (1 + exp(-beta[1] - beta[2] * x1 - beta[3] * x2)))
  lines(y=dat.plot$y,x= dat.plot$x2, col = "red", lty=2)
  readline()
}
# poor optimizer...



# Let's regularize the likelihood (L2)
llik_reg <- function(beta) {
  - sum( dbinom(x = y, size = 1, 
                prob = 1 / (1 + exp(-beta[1] - beta[2] * x1 - beta[3] * x2)), 
                log = T) ) + lambda * sum(beta[-1]^2)
}

# let's just use lambda = 1. Why? For no reason. 
lambda = 1

plot(y~x2)
abline(v=quantile(x2, probs = 0.6), col = "red")


for (num in c(10, 20, 50,100,500,1000,10000))  {
  beta <- (mod.lik.reg <- optim(rnorm(3), llik_reg, control = list(maxit = num)))$par
  cat(num, 'iter; parameter = ', beta[3], '\n')
  dat.plot <- data.frame(x1=mean(x1), x2=seq(-5,5,by=0.05)) %>%
    mutate(y = 1 / (1 + exp(-beta[1] - beta[2] * x1 - beta[3] * x2)))
  lines(y=dat.plot$y,x= dat.plot$x2, col = "red")
  readline()
}
# Here we go!



# How to choose lambda? It's a great question that we're going to talk more about 
# later in this course


# Also see logistf function

mod2 <- logistf(y ~ x1 + x2)



########################################################################
#                                   Overdispersion 
########################################################################



rm(list = ls())
set.seed(1)
n <- 500
x <- rbinom(n, 1, 1/2)
# Generates data for overdispersion z
# and fits negative binomial regression


y <- lapply(c(1,3,5), function(k) rnbinom(n, size = k, mu = exp(0 + 0.25*x)))

# Varying degrees of beta(theta)
# Recall: phi = 1+1/beta

# note the parameterization in R


out1 <- lapply(1:3, function(k) MASS::glm.nb(y[[k]] ~ x))
out2 <- lapply(1:3, function(k) glm(y[[k]] ~ x, family = poisson(link="log")))

summary(out2[[3]])
summary(out1[[3]])

out3 <- lapply(1:3, function(k) glm(y[[k]] ~ x, family = quasipoisson(link = "log")))

AER::dispersiontest(out2[[1]])
AER::dispersiontest(out2[[2]])
AER::dispersiontest(out2[[3]])


# Finally, check the real possion

set.seed(12345)
n = 1000
beta0 = 1
beta1 = 0.5
beta2 = -0.5
dat <- data.frame(x = rnorm(n, mean = 3, sd = 0.5), z = rnorm(n, mean = 3, sd = 0.5))
lambda <- exp(beta0 + beta1 * dat$x + beta2 * dat$z) # the mean function
dat$y <- rpois(n = n, lambda = lambda) # the dep.var


m1 <- MASS::glm.nb(y~x+z, data=dat)
m2 <- glm(y~x+z, data=dat, family = poisson(link="log"))
m3 <- glm(y~x+z, data=dat, family = quasipoisson(link = "log"))


AER::dispersiontest(m2)
