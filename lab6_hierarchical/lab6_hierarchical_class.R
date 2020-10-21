# Title:    Quant III (Lab 6)
# Name:     Junlong Aaron Zhou
# Date:     October 16, 2020
# Summary:  Hierachical Model: Lmer
#################################################################################


rm(list = ls())
library(haven)
library(arm)
library(dplyr)
library(lme4)
library(lattice) 
library(foreign)
library(MuMIn)
setwd("~/Dropbox/Teaching/2020_Fall_Quant_3/lab6_hierachical")
 

# Set up the data for the election88 example

# Load in data for region indicators
# Use "state", an R data file (type ?state from the R command window for info)
#
# Regions:  1=northeast, 2=south, 3=north central, 4=west, 5=d.c.
# We have to insert d.c. (it is the 9th "state" in alphabetical order)

data (state)                  # "state" is an R data file
state.abbr <- c (state.abb[1:8], "DC", state.abb[9:50])
dc <- 9
not.dc <- c(1:8,10:51)
region <- c(3,4,4,3,4,4,1,1,5,3,3,4,4,2,2,2,2,3,3,1,1,1,2,2,3,2,4,2,4,1,1,4,1,3,2,2,3,4,1,1,3,2,3,3,4,1,3,4,1,2,4)

# Load in data from the CBS polls in 1988

polls <- read_dta ("polls.dta") 
polls <- read_dta("~/Dropbox/Quant III/Quant_III_Lab_2020/lab6_hierarchical/polls.dta")

# Select just the data from the last survey (#9158)

table (polls$survey)                # look at the survey id's
ok <- polls$survey==8            # define the condition
polls.subset <- polls[ok,]    # select the subset of interest 
print (polls.subset[1:5,])

dat <- polls.subset
dat$y <- dat$bush

############################### Varying Intercept ######################################
########################################################################################

#　Let's start by fitting a simple OLS regression of 
#  Reaction times on Days in experiments
 

m.ols <- lm(y ~ black*female, data = dat)
summary(m.ols)
AIC(m.ols)
BIC(m.ols) 

# Now, varying intercept for States

m.sc.fe <- lm(y ~ black*female + factor(state), data = dat)
summary(m.sc.fe)
AIC(m.sc.fe)
BIC(m.sc.fe) 


# Which one is better? 


############################### Lmer Function ######################################
##############################################################################################



M1 <- lmer(y ~ black*female + (1 | state), data=dat)
summary(M1) 
rr1 <- ranef(M1, condVar = TRUE)
ranvar <- attr(rr1[[1]], "postVar")
ranvar
se.ranef(M1)
dotplot(rr1,scales = list(x = list(relation = 'free')))[["state"]]

coef(M1)
fixef(M1)
coef(M1)$state$`(Intercept)` - ranef(M1)$state$`(Intercept)`  

r.squaredGLMM(M1)

M2 <- glmer(y ~ black*female + (1 | state) + (1 | age) , family=binomial(link="logit"), data=dat)
summary(M2) 
BIC(M1)
BIC(M2)
 
############################### Logistic Regression ######################################
##############################################################################################

 

M1 <- glmer(y ~ black*female + (1 | state), family=binomial(link="logit"), data=dat)
summary(M1) 
rr1 <- ranef(M1, condVar = TRUE) 
fixef(M1)
se.coef(M1)

M2 <- glmer(y ~ black*female + (1 | state) + (1 | age) , family=binomial(link="logit"), data=dat)
summary(M2)

AIC(M1)

############################### Varying Coefficients ######################################
##############################################################################################


M1 <- glmer (y ~ black*female + (1 | state), family=binomial(link="logit"), data=dat)
summary(M1)

M2 <- glmer (y ~ black*female + (1 | state) + (1+edu | age) , family=binomial(link="logit"), data=dat)
summary(M2)
ranef(M2)

M3 <- glmer (y ~ black*female + (1 | state) + (edu || age) , family=binomial(link="logit"), data=dat)
summary(M3)
ranef(M3)



################## A Bayesian Approach Using RStan ######################################
##############################################################################################


M1 <- glmer (y ~ black + female + (1 | state), 
             family=binomial(link="logit"),
             data=dat)
summary(M1)

mod.string.h <- '
data {
  int N ;
  int p_state ; 
  real black[N];
  real female[N];
  int state[N]; 
  int y[N] ;
}

parameters {
  real a ;
  real b_black ;
  real b_female ; 
  real b_state[p_state];
  real<lower=0> sigma;
  real<lower=0> tau_b_black;
  real<lower=0> tau_b_female;
  real<lower=0> tau_b_state ; 
}

model {
  // hyperpriors
  sigma ~ uniform(0, 1) ;
  tau_b_black ~ uniform(0, 1) ;
  tau_b_female ~ uniform(0, 1) ;
  tau_b_state ~ uniform(0, 1); 
  
  // priors
  a ~ normal(0, 1) ;
  b_black ~ normal(0, tau_b_black) ;
  b_female ~ normal(0, tau_b_female) ;
  b_state ~ normal(0, tau_b_state) ; 
  
  // likelihood
  for (n in 1:N){
    y[n] ~ bernoulli_logit(a + b_black * black[n]+
        female[n]*b_female + b_state[state[n]]);
  }
}

generated quantities {
  vector[N] yrep ;
  for (n in 1:N) {
    yrep[n] = bernoulli_logit_rng(a + b_black * black[n]+
        female[n]*b_female + b_state[state[n]]) ;
  }
}
'

dat.n <- dat[,c("y","black","female","state")]
dat.n <- na.omit(dat.n)
dat_stan <- list(
    N=nrow(dat.n),
    p_state = length(unique(dat.n$state)),
    black = dat.n$black,
    female = dat.n$female,
    state=as.integer(as.factor(dat.n$state)),
    y = dat.n$y
)


library(rstan)
library(coda)

# mod.stan <- stan(model_code = mod.string.h,
#      data = dat_stan,
#      seed=1234,
#      chains = 3, iter = 20000,
#      warmup = 15000, thin = 2,  
#      control = list(adapt_delta = 0.999,
#                     max_treedepth = 15))
# 
# save(mod.stan, file="model_stan.RData")

load("model_stan.RData")
params <- rstan::extract(mod.stan)
str(params) 
 
str(params$b_black)
hist(params$b_black)
coda::HPDinterval(as.mcmc(as.numeric(params$b_black)))

