?rgamma

n = 1000
alpha = 5
beta = 10
x = rgamma(n, shape=alpha, scale=1/beta)
install.packages("stats4")
library(stats4)

?dgamma
minus_logL = function(a,b){
  return(-sum(dgamma(x, shape=a, scale=1/b, log=TRUE)))
}

?mle
mle(minus_logL, start=c(1,1), lower=c(0,0))

install.packages("gmm")
library(gmm)

?gmm
g = function(theta,u){
  a = theta[1]
  b = theta[2]
  m1 = a/b - u
  m2 = (a+a^2)/(b^2) - u^2
  return(cbind(m1,m2))
}
gmm(g,x,c(1,1))

install.packages("fitdistrplus")
library(fitdistrplus)
?fitdist

alpha = 5
beta = 10
x = rgamma(n, shape=alpha, scale=1/beta)
res1=fitdist(x,"gamma", method="mle")
res1$estimate
res2=fitdist(x,"gamma", method="mme")
res2$estimate