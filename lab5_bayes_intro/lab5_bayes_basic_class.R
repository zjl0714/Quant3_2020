# Title:    Quant III (Lab 5)
# Name:     Junlong Aaron Zhou
# Date:     October 9, 2020
# Summary:  Bayesian stats: intro
#           Beta distribution; Bernoulli example with different priors
#           Normal model: unknown mean only
#################################################################################


rm(list = ls())

# install.packages('coda')
library(coda)

############################### BETA DISTRIBUTION ######################################
########################################################################################

# Generate values for the range of x-axis
theta <- seq(0, 1, 0.001)

# Generate parameter values
a.vals <- seq(0.5, 2, 0.5)
b.vals <- seq(0.5, 2, 0.5)

# Plot Beta(a,b) density along x
par(mfrow = c(length(a.vals), length(b.vals)), mar=rep(1,4))
for (a in a.vals) {
  for (b in b.vals) {
    plot(dbeta(x = theta, shape1 = a, shape2 = b) ~ theta, type = 'l', 
         xlab = 'Parameter', ylab = 'Density', 
         main = paste0('a = ', a, ', b = ', b))
  }
}
par(mfrow = c(1,1), mar = rep(3,4))
# 8x10

# Takeaway: 
# 1. It's a very flexible distribution!


############################### Bernoulli prob (manual) ######################################
##############################################################################################

rm(list = ls())

plot_prior_posterior <- function(thetas, a, b, title="Prior") {
  plot( thetas^(a - 1) * (1-thetas)^(b - 1) ~ theta.vals, 
        type = 'l', xlab = 'Theta', ylab = 'Density', main = title)
}

# Generate some data
set.seed(123456)
(x <- rbinom(n = 10, size = 1, prob = 0.9))



#--------
##### Assume prior = Beta(1,1)

# Set prior params
a = b = 1


# Set posterior params
a.new <- a + sum(x)
b.new <- b + length(x) - sum(x)

# Recall P(p|y)~Beta(sum (y) + a, n − sum(y) + b)

# Plot: let's compare prior, likelihood, and posterior
theta.vals <- seq(0, 1, 0.001)
par(mfrow = c(3,1))

# Plot prior
plot_prior_posterior(thetas = theta.vals, a = a, b = b)

# Plot likelihood
####################################################
##### LAB WORK: Plot the likelihood function for the observed data
#               over the range of theta.vals
##### YOUR CODE STARTS HERE:
 


##### END OF CODE
####################################################

# Plot posterior
plot_prior_posterior(thetas = theta.vals, a = a.new, b = b.new, title="Posterior")
par(mfrow = c(1,1))



#----------
##### Assume prior Beta(), s.t. a+b = n = 5; mean (m) = 0.4
n = 5
m = 0.4

# Set prior params
a = n * m    # EX = a / n
b = n * (1 - m)

# Set posterior params
a.new <- a + sum(x)
b.new <- b + length(x) - sum(x)


# Plot
theta.vals <- seq(0, 1, 0.001)
par(mfrow = c(3,1))

# Plot prior
plot_prior_posterior(thetas = theta.vals, a = a, b = b)

# Plot the likelihood
plot( sapply(theta.vals, function(k) prod(dbinom(x = x, size = 1, prob = k))) ~ 
        theta.vals, type = 'l', main = 'Likelihood')

# Plot posterior
plot_prior_posterior(thetas = theta.vals, a = a.new, b = b.new, title="Posterior")
par(mfrow = c(1,1))

# What do you notice (c.f. Beta(1,1) )


### Sample from the posterior
theta.sampled <- rbeta(n = 10000, shape1 = a.new, shape2 = b.new)

mean(theta.sampled)  # c.f. a.new / ( a.new + b.new )
sd(theta.sampled)    # c.f. sqrt(a.new * b.new / ( ( a.new + b.new )^2 * ( a.new + b.new + 1) ))


### HPD Interval
coda::HPDinterval(obj = as.mcmc(theta.sampled), prob = 0.95)


### Central credible intervals
# If we know the parametric form of the posterior
qbeta(p = c(0.025, 0.975), shape1 = a.new, shape2 = b.new)

# If we don't know the parametric form of the posterior
boot_ci <- function(x, alpha = 0.05) {
  x <- x[which(!is.na(x) & !is.nan(x))] # remove NAs and NaNs
  x <- x[order(x)]
  c(x[ceiling(alpha/2 * length(x))], x[ceiling((1 - alpha/2) * length(x))])
}

boot_ci(theta.sampled)
quantile(theta.sampled, probs = c(0.025,0.975))


 

############################### Normal (mean unknown) ######################################
############################################################################################

rm(list = ls())

### Generate data
set.seed(12345678)
n = 100
sigma = 3
y <- rnorm(n, mean = 10, sd = sigma)


### Set grid of theta values
thetas <- seq(from = -20, to = 20, by = 0.1)


### Compute likelihood on the grid of thetas

lik <- sapply(thetas, function(k)  prod(dnorm(x = y, mean = k, sd = sigma)))


### Compute prior on the grid of thetas
mu.zero = 0         # Why? That's my prior!! :-) 
sigma.zero = 1000   # A vague one

prior <- dnorm(x = thetas, mean = mu.zero, sd = sigma.zero)

### Compute posterior
mu.one <- ( sum(y) / sigma^2 + mu.zero / sigma.zero^2 ) / ( n / sigma^2 + 1 / sigma.zero^2  )
sigma.one <- sqrt( 1 / ( 1 / sigma.zero^2 + n / sigma^2 ) )

posterior <- dnorm(x = thetas, mean = mu.one, sd = sigma.one)


### Plot the results
par(mfrow = c(3, 1))
plot(prior ~ thetas, type = 'l', main = 'Prior', xlab = 'Theta', ylab = 'Density')
plot(lik ~ thetas, type = 'l', main = 'Likelihood', xlab = 'Theta', ylab = 'Likelihood')
plot(posterior ~ thetas, type = 'l', main = 'Posterior', xlab = 'Theta', ylab = 'Density')
par(mfrow = c(1, 1))


### Sample from the posterior
####################################################
##### LAB WORK: 1) compute the posterior point estimate
#               2) get HPD interval
#               3) get central credible interval
##### YOUR CODE STARTS HERE:

##### END OF CODE
####################################################


















mu.sampled <- rnorm(n = 10000, mean = mu.one, sd = sigma.one)

mean(mu.sampled)  # c.f. mu.one
sd(mu.sampled)    # c.f. sigma.one


### HPD Interval
coda::HPDinterval(obj = as.mcmc(mu.sampled), prob = 0.95)


### Central credible intervals
# If we know the parametric form of the posterior
qnorm(p = c(0.025, 0.975), mean = mu.one, sd = sigma.one)

# If we don't know the parametric form of the posterior
boot_ci <- function(x, alpha = 0.05) {
  x <- x[which(!is.na(x) & !is.nan(x))] # remove NAs and NaNs
  x <- x[order(x)]
  c(x[ceiling(alpha/2 * length(x))], x[ceiling((1 - alpha/2) * length(x))])
}

boot_ci(mu.sampled)






### A severely bimodal distribution:
tst <- c(rnorm(10000), rnorm(5000, 7))
hist(tst, freq=FALSE, ylim=c(0,0.3))



hdiMC <- coda::HPDinterval(obj = as.mcmc(tst), prob = 0.95)
segments(hdiMC[1], 0, hdiMC[2], 0, lwd=3, col='red')
# This is a valid 95% CrI, but not a Highest Density Interval




library(HDInterval)
dens <- density(tst)
lines(dens, lwd=2, col='blue')
(hdiD1 <- hdi(dens))  # default allowSplit = FALSE; note the warning
(ht <- attr(hdiD1, "height"))
segments(hdiD1[1], ht, hdiD1[2], ht, lty=3, col='blue')
(hdiD2 <- hdi(dens, allowSplit=TRUE))
segments(hdiD2[, 1], ht, hdiD2[, 2], ht, lwd=3, col='blue')
# This is the correct 95% HDI.



# Mix more:

tst2 <- c(rnorm(10000), rnorm(10000,7, 3))
hist(tst2, freq=FALSE, ylim=c(0,0.2))
(hdiMC <- hdi(tst2))
segments(hdiMC[1], 0, hdiMC[2], 0, lwd=3, col='red')
# This is a valid 95% CrI, but not a Highest Density Interval


dens2 <- density(tst2)
lines(dens2, lwd=2, col='blue')
(hdiD1 <- hdi(dens2))  
(ht <- attr(hdiD1, "height"))
segments(hdiD1[1], ht, hdiD1[2], ht, lty=3, col='blue')
(hdiD2 <- hdi(dens2, allowSplit=TRUE))
segments(hdiD2[, 1], ht, hdiD2[, 2], ht, lwd=3, col='blue')
# OK in this case


# Most of cases: hopefully we have single modal posterior.