?rexp

nrep = 100
n = 1000
lambda = 10
mu = 15
lambda_mle = rep(NA,nrep)
lambda_mm = rep(NA,nrep)
mu_mle = rep(NA,nrep)
mu_mm = rep(NA,nrep)
for (k in 1:nrep){
  x = rexp(n, rate=lambda) + mu
  mu_mle[k] = min(x)
  lambda_mle[k] = 1 / (mean(x) - mu_mle[k])
  aux = sqrt((mean(x^2)-(mean(x))^2))
  lambda_mm[k] = 1 / aux
  mu_mm[k] = mean(x) - aux
}

boxplot(cbind(lambda_mle,lambda_mm),col="wheat",xaxt="n")
axis(side=1, at=1:2,label=c("lambda_ML","lambda_MM"))
grid()
boxplot(cbind(mu_mle,mu_mm),col="magenta",xaxt="n")
axis(side=1, at=1:2,label=c("mu_ML","mu_MM"))
grid()