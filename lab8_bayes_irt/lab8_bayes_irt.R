# Title:    Quant III (Lab 8)
# Name:     Junlong Aaron Zhou
# Date:     November 6, 2020
# Summary:  Bayesian stats: IRT
#################################################################################

rm(list = ls())
# install.packages(c('pscl', 'ggmcmc', 'ggpubr'))
# NB! You need pscl_1.5.2 or higher! Versions below have critical bugs
library(rstan)
library(coda)
library(ggmcmc)

set.seed(12345678)
setwd("~/Dropbox/Teaching/2020_Fall_Quant_3/lab8_bayes_irt")
##### Load roll-call data for 113th Senate (Jan 24 -- Nov 21, 2013)

rc <- get(load('rc.RData'))
rc$desc
rc$source


##### Inspect rc object 
table(rc$vote.data$congress)
table(rc$vote.data$congress.year)
unique(rc$vote.data$vote.date)

str(rc$legis.data)
str(rc$vote.data)
str(rc$votes)
# 104 senators; 246 bills



##### Clean rc object for analysis
str(rc$codes)
# Let's remove notInLegis and keep "yea" and "nay"
# Also, we don't want roll call with no legislators ("lop" argument)
rc <- pscl::dropRollCall(rc, dropList=list(codes = "notInLegis", lop = 0))



#######################################################################################
################################ IRT with pscl::ideal() ###############################
#######################################################################################

##### Estimate a model
myburnin = 20000

set.seed(12345678)
# irt.ideal1 <- pscl::ideal(object = rc, d = 1, maxiter = myburnin * 2,
#                    thin = 2, burnin = myburnin, normalize = T,
#                    startvals = 'eigen' ,store.item=T)
# 
# save(x = irt.ideal1, file = 'irt_ideal_40thous1.RData')
load('irt_ideal_40thous1.RData')
?ideal 


set.seed(12345678)
# irt.ideal2 <- pscl::ideal(object = rc, d = 1, maxiter = myburnin * 2, 
#                          thin = 2, burnin = myburnin, normalize = T,                     
#                          startvals = 'eigen')
# save(x = irt.ideal2, file = 'irt_ideal_40thous2.RData')

load('irt_ideal_40thous2.RData')


#-------------------------
##### Convergence diagnostics
# NB! 1) pscl::ideal runs only 1 chain -- this is bad
#     2) pscl is not very good with diagnostics, so need external packages

# Convert to a coda object
# NB! pscl::idealToMCMC is bad with naming parameters 

irt.coda <- pscl::idealToMCMC(irt.ideal1)  
str(irt.coda) # Why 538? Try to figure it out. Or use my_ideal_to_coda :-)

irt.coda <- pscl::idealToMCMC(irt.ideal2)   
str(irt.coda)

# From coda to ggmcmc
ggmcmc.ideal <- ggmcmc::ggs(coda::as.mcmc.list(irt.coda))


# Actual diagnostics
ggmcmc::ggs_traceplot(ggmcmc.ideal, family = 'Sanders')
ggmcmc::ggs_traceplot(ggmcmc.ideal, family = 'Cruz')
# Not very useful, since only 1 chain

pl1 <- ggmcmc::ggs_running(ggmcmc.ideal, family = 'Sanders')
pl2 <- ggmcmc::ggs_running(ggmcmc.ideal, family = 'Warren')
pl3 <- ggmcmc::ggs_running(ggmcmc.ideal, family = 'Rubio')
pl4 <- ggmcmc::ggs_running(ggmcmc.ideal, family = 'Cruz')
ggpubr::ggarrange(pl1, pl2, pl3, pl4, nrow = 2, ncol = 2)
# Much more useful
# Graphs look all really bad, 
#         as the running mean doesn't fluctuate around the mean

ggmcmc::ggs_geweke(ggmcmc.ideal, family = '(Sanders)|(Warren)|(Rubio)|(Cruz)')
# Obviously, we have convergence failure for Sanders, Warren, and Cruz


#-------------------------
##### Inspect results
summary(irt.ideal2)
plot(irt.ideal2)


##############################################################################
################################ IRT with Stan ###############################
##############################################################################

### Prepare data for IRT model
v <- c(rc$votes)
L <- nrow(rc$votes) 
B <- ncol(rc$votes)
N <- length(v)

# What are these?


# We convert roll-call matrix to bill-legislator level data

l <- rep(1:L, times = B)
b <- rep(1:B, each = L) 

# Remove missing values
miss <- which(is.na(v))
N <- N - length(miss)
l <- l[-miss]
b <- b[-miss]
v <- v[-miss]
Sys.sleep(1)
rm(miss)

# Put data together
stan_irt_data <- list(L = L, B = B, N = N, 
                      v = v, l = l, b = b)

# For future use: identify senators
sen.names <- paste(rc$legis.data$first.name, 
                   rc$legis.data$middle.name, 
                   rc$legis.data$last.name, sep = " ")


#-----------------------------------------------
##### IRT: NO CONSTRAINTS


### Define IRT model in Stan
mod.string <- '
data {
  int<lower=1> L ; // number of legislators
  int<lower=1> B ; // number of bills
  int<lower=1> N ; // number of observations: N = L * B - missing
  int<lower=0, upper=1> v[N] ; // modeled variable: recorded vote for observation n
  int<lower=1, upper=L> l[N] ; // legislator for observation n
  int<lower=1, upper=B> b[N] ; // bill for observation n
}

parameters {
  real beta[B] ; // location parameter for a bill        
  real eta[B] ;  // discrimination parameter for a bill
  real theta[L] ; // location parameter for legislator
}

model {
  eta ~ normal(0, 5) ; // default prior used in pscl::ideal
  beta ~ normal(0, 5) ; // default prior used in pscl::ideal
  theta ~ normal(0, 1) ; // default prior used in pscl::ideal
  for (n in 1:N) {
     v[n] ~ bernoulli_logit( eta[b[n]]*theta[l[n]]-beta[b[n]]) ;
  }
}
'

# ### Estimate IRT model
# mod.fit <- rstan::stan(model_code = mod.string,
#                        data = stan_irt_data,
#                        iter = 5000, warmup = 2500, 
#                        chains = 3, thin = 2, cores = 3,
#                        control = list(adapt_delta = 0.99,
#                                       max_treedepth = 12))
# 
# 
# save(x = mod.fit, file = "stan_irt_w5000_unconst.RData")

mod.fit.nc <- get(load("stan_irt_w5000_unconst.RData"))
Sys.sleep(1)
rm(mod.fit)

### Convergence diagnostics
ggmcmc.theta.obj.nc <- ggmcmc::ggs(rstan::As.mcmc.list(mod.fit.nc, pars = c("theta")),
                                inc_warmup = F)

ggmcmc::ggs_Rhat(ggmcmc.theta.obj.nc, family = 'theta')

# What's happen? 

ggmcmc::ggs_traceplot(ggmcmc.theta.obj.nc, family = 'theta\\[27\\]', 
                      original_burnin = TRUE)
rc$legis.data[27,]

### Takeaway: 
# 1) Different chains converged to different modes
# 2) Multimodality because the model is not identified


#---------------------------------------------------------------
##### Identification 1: tight prior

### Define IRT model in Stan
mod.string <- '
data {
  int<lower=1> L ; // number of legislators
  int<lower=1> B ; // number of bills
  int<lower=1> N ; // number of observations: N = L * B - missing
  int<lower=0, upper=1> v[N] ; // modeled variable: recorded vote for observation n
  int<lower=1, upper=L> l[N] ; // legislator for observation n
  int<lower=1, upper=B> b[N] ; // bill for observation n
}

parameters {
  real beta[B] ; // location parameter for a bill        
  real eta[B] ;  // discrimination parameter for a bill
  real theta[L] ; // location parameter for legislator
}

model {
  beta ~ normal(0, 5) ; // default prior used in pscl::ideal
  eta ~ normal(0, 5) ; // default prior used in pscl::ideal
  theta ~ normal(0, 1) ; // default prior used in pscl::ideal
  theta[27] ~ normal(2, .1)  ; // Ted Cruz
  theta[59] ~ normal(2, .1)  ; // Mike Lee
  theta[65] ~ normal(-2, .1) ; // Robert Menendez
  theta[11] ~ normal(-2, .1)  ;  // Barbara Boxer
  for (n in 1:N) {
   v[n] ~ bernoulli_logit( eta[b[n]]*theta[l[n]]-beta[b[n]]) ;
  }
}
'

### Estimate IRT model
# mod.fit <- rstan::stan(model_code = mod.string,
#                        data = stan_irt_data,
#                        iter = 5000, warmup = 2500,
#                        chains = 3, thin = 2, cores = 3,
#                        control = list(adapt_delta = 0.99,
#                                       max_treedepth = 12))
# 
# save(x=mod.fit, file = "stan_irt_w5000_prior2.RData")
mod.fit.pr <- get(load("stan_irt_w5000_prior2.RData"))
Sys.sleep(1)
rm(mod.fit)


### Convergence diagnostics
ggmcmc.theta.obj.pr <- ggmcmc::ggs(rstan::As.mcmc.list(mod.fit.pr, pars = c("theta")),
                                inc_warmup = F)

ggmcmc::ggs_Rhat(ggmcmc.theta.obj.pr, family = 'theta')

ggmcmc::ggs_traceplot(ggmcmc.theta.obj.pr, family = 'theta\\[27\\]', 
                      original_burnin = TRUE)


ggmcmc::ggs_geweke(ggmcmc.theta.obj.pr, family = 'theta')


# Why didn't it help?

### Takeaway: 
# 1) Just imposing a tight prior on some params might not be enough




#---------------------------------------------------------------
##### Identification 2: tight truncated prior

### Define IRT model in Stan
mod.string <- '
data {
  int<lower=1> L ; // number of legislators
  int<lower=1> B ; // number of bills
  int<lower=1> N ; // number of observations: N = L * B - missing
  int<lower=0, upper=1> v[N] ; // modeled variable: recorded vote for observation n
  int<lower=1, upper=L> l[N] ; // legislator for observation n
  int<lower=1, upper=B> b[N] ; // bill for observation n
}

parameters {
  real beta[B] ; // location parameter for a bill        
  real eta[B] ;  // discrimination parameter for a bill
  real theta[L] ; // location parameter for legislator
}

model {
  beta ~ normal(0, 5) ; // default prior used in pscl::ideal
  eta ~ normal(0, 5) ; // default prior used in pscl::ideal
  theta ~ normal(0, 1) ; // default prior used in pscl::ideal
  theta[27] ~ normal(2, .1) T[0, ] ; // Ted Cruz
  theta[59] ~ normal(2, .1) T[0, ] ; // Mike Lee
  theta[65] ~ normal(-2, .1) T[, 0]; // Robert Menendez
  theta[11] ~ normal(-2, .1) T[, 0] ;  // Barbara Boxer
  for (n in 1:N) {
   v[n] ~ bernoulli_logit( eta[b[n]]*theta[l[n]]-beta[b[n]]) ;
  }
}
'

# Q: Why truncating instead of constraining the range?
# A: Params defined as arrays. So, would need different params for the fixed ones


### Estimate IRT model
# mod.fit <- rstan::stan(model_code = mod.string,
#                        data = stan_irt_data,
#                        iter = 5000, warmup = 2500,
#                        chains = 3, thin = 2, cores = 3,
#                        control = list(adapt_delta = 0.99,
#                                       max_treedepth = 12))
# save(x=mod.fit, file = "stan_irt_w5000_const.RData")
mod.fit.con <- get(load("stan_irt_w5000_const.RData"))
Sys.sleep(1)
rm(mod.fit)

### Convergence diagnostics
ggmcmc.theta.obj.con <- ggmcmc::ggs(rstan::As.mcmc.list(mod.fit.con, pars = c("theta")),
                                inc_warmup = F)

ggmcmc::ggs_Rhat(ggmcmc.theta.obj.con, family = 'theta')


# Let's find params with bad chain convergence
rhat_dat <- ggmcmc::ggs_Rhat(ggmcmc.theta.obj.con, family = 'theta')$data
rhat_dat <- rhat_dat[order(rhat_dat$Rhat, decreasing = T),]
head(rhat_dat)

# Much better!

# Inspect these params
ggmcmc::ggs_traceplot(ggmcmc.theta.obj.con, family = 'theta\\[40\\]', 
                      original_burnin = TRUE)

ggmcmc::ggs_traceplot(ggmcmc.theta.obj.con, family = 'theta\\[1\\]', 
                      original_burnin = TRUE)
ggmcmc::ggs_traceplot(ggmcmc.theta.obj.con, family = 'theta\\[55\\]', 
                      original_burnin = TRUE)
# Apparently, posterior geometry is bad for these params
# Who are these people? Do we know anything about them? 
sen.names[c(40, 1, 55)]
# 40 - Orrin G. Hatch (R)
# 1 -  Lamar  Alexander (R)
# 25 - Amy  Klobuchar (D)

# Q: What to do with this now?
# A: Let's try constraining these params too?


#-----------------------------
##### Visualize results
params <- factor(unique(as.character(ggmcmc.theta.obj.con$Parameter)), 
                 levels = paste0('theta[', 1:length(sen.names), ']') )

sen.plot.dat <- data.frame('Parameter' = params, 'Label' = sen.names)


# Set up ggmcmc object with linked names
ggmcmc.theta.obj.full <- ggmcmc::ggs(rstan::As.mcmc.list(mod.fit.con, pars = c("theta")), 
                                     par_labels = sen.plot.dat,  # This is the innovation!
                                     inc_warmup = F)


# Plot results
pdf(file = 'senator_ideology.pdf', width = 10, height = 20)
ggmcmc::ggs_caterpillar(ggmcmc.theta.obj.full)
# Also compare with our model from ideal
pscl::plot.ideal(irt.ideal2,showAllNames=T)
dev.off()


#---------------------------------------------------------------
##### Identification 3: tight truncated prior + 
#                       additional parameter constraints + 
#                       hierarchical model w/ party info


str(rc$legis.data)
subset(rc$legis.data, party == 'R', select = c(last.name, first.name, party))

r <- ifelse(rc$legis.data$party=="R",1,0)
d <- ifelse(rc$legis.data$party=="D",1,0)

r <- as.integer(r)
d <- as.integer(d)

stan_irt_data <- list(L = L, B = B, N = N, 
                      v = v, l = l, b = b,
                      d = d, r = r)
### Define IRT model in Stan

mod.string <- '
data {
  int<lower=1> L ; // number of legislators
  int<lower=1> B ; // number of bills
  int<lower=1> N ; // number of observations: N = L * B - missing
  int<lower=0, upper=1> v[N] ; // modeled variable: recorded vote for observation n
  int<lower=1, upper=L> l[N] ; // legislator for observation n
  int<lower=1, upper=B> b[N] ; // bill for observation n
  int<lower=0, upper=1> d[L] ; // dummy for democrats
  int<lower=0, upper=1> r[L] ; // dummy for republicans
}

parameters {
  real beta[B] ; // location parameter for a bill        
  real eta[B] ;  // discrimination parameter for a bill
  real theta[L] ; // location parameter for legislator
  real<lower=0> gamma_r ; // location shift for republicans
  real<upper=0> gamma_d ; // location shift for democrats
}

model {
  real alpha[N] ;
  beta ~ normal(0, 5) ; 
  eta ~ normal(0, 5) ; 
  gamma_r ~ normal(2, 0.1)T[1,3] ;
  gamma_d ~ normal(-2, 0.1)T[-3,-1] ;
  for (j in 1:L) {
    theta[j] ~ normal( gamma_r * r[j] + gamma_d * d[j], 1 ) ;
  }
  theta[27] ~ normal(2, .1) T[0, 5] ; // Ted Cruz
  theta[59] ~ normal(2, .1) T[0, 5] ; // Mike Lee
  theta[65] ~ normal(-2, .1) T[-5, 0]; // Robert Menendez
  theta[11] ~ normal(-2, .1) T[-5, 0] ;  // Barbara Boxer
  for (n in 1:N) {
    alpha[n] =  eta[b[n]]*theta[l[n]]-beta[b[n]]  ;
  }
  v ~ bernoulli_logit(alpha) ;
}
'

# mod.fit <- rstan::stan(model_code = mod.string,
#                       data = stan_irt_data,
#                       iter = 5000, warmup = 2500,
#                       chains = 3, thin = 2, cores = 4,
#                       control = list(adapt_delta = 0.99,
#                                      max_treedepth = 12))
# save(x = mod.fit, file = "stan_irt_w5000_party.RData")
mod.fit.party <- get(load("stan_irt_w5000_party.RData"))
Sys.sleep(1)
rm(mod.fit)

### Convergence diagnostics

ggmcmc.gamma.obj.party <- ggmcmc::ggs(rstan::As.mcmc.list(mod.fit.party, pars = c("gamma_r", "gamma_d")),
                                      inc_warmup = F)


ggmcmc::ggs_Rhat(ggmcmc.gamma.obj.party)
ggmcmc::ggs_density(ggmcmc.gamma.obj.party)


ggmcmc.gamma.obj.party.indiv <- ggmcmc::ggs(rstan::As.mcmc.list(mod.fit.party, pars ="theta"),
                                            par_labels = sen.plot.dat,
                                            inc_warmup = F)
ggmcmc::ggs_caterpillar(ggmcmc.gamma.obj.party.indiv)


# Takeaways:
# 1) Getting IRT models properly identified is tricky
# 2) No proper identifications constraints leads to multimodality
# 3) Imposing only tight priors is not enough
# 4) Truncated distributions are more useful
# 5) But be careful with the location of fixed parameters

