# Eric Vansteenberghe
# 2019
# Quantitative methods in finance 
# Avoiding spurious regression with R and financial time series
# This exercise is largely inspired by the book Econometrie des fondements a la modelisation from Stephen Bazen and Mareva Sabatier published in 2007.

closeAllConnections()
rm(list = ls())
setwd("/Users/skimeur/Google Drive/empirical_finance")

library(tseries) # for adf.test
library(fGarch)  # for ARCH model
library(zoo)
library(reshape) # to prepare the data before ploting
library(ggplot2) # to plot
library(urca)
library(lmtest) # for Durbin-Watson test
library(forecast)
library(rugarch)  # ARMA-GARCH models
#library(ccgarch) # DCC-GARCH model

# if you have good internet access and Yahoo finance is running
internet = 0

if(internet == 1){
  # We define the start and end date of our import
  start.d <- "1995-01-16"
  end.d <- "1999-02-26"
  
  
  # We download the ’Close’ value for the CAC 40 and the Dow Jones indexes:
  dow <- get.hist.quote('^DJI',quote = "Close",start = start.d,end = end.d)
  cac40 <- get.hist.quote('^FCHI',quote = "Close",start = start.d,end = end.d)
  
  
  # Next we create a DataFrame from the two downloaded series, in logarithm,
  df <- merge(log(dow),log(cac40))
  colnames(df) <- c("Dow","CAC")
  
  # We can write these DataFrames to csv files
  # write.csv(df,file = "R_data/CAC_Dow.csv",row.names = FALSE)
  
}else{
  df <- read.csv("R_data/CAC_Dow.csv")
  df$Date <- as.Date(df$Date)
  rownames(df) <- df$Date
}


# We want to fill in the blank (not a number) with the latest available value
df <- na.locf(df)

df <- as.data.frame(df)
df[,"Date"] <- as.Date(rownames(df))

# We plot the time series
#pdf("CAC_Dow_indexes.pdf",width=7,height=5)
mdf <- melt(df,id.vars = "Date")
ggplot(data = mdf,aes(x=Date,y=value)) + geom_line(aes(color=variable),size=1.25)+scale_x_date("Year")+scale_y_continuous("stock prices")
#dev.off()


#############################
# A first spurious regression
#############################

model <- lm(CAC ~ Dow,data = df)
summary(model)
dwtest(CAC ~ Dow,data = df)


ls(dwtest(CAC ~ Dow,data = df))
# test if the R^2 is greater than the DW statistics
summary(model)$adj.r.squared > dwtest(CAC ~ Dow,data = df)$statistic

adf.test(model$residuals, k = 1) # the residuals contains a unit root

# we find a R-square of 89%, but a low Durbin-Watson test statistics
# Our series were non-stationary! Our regression could be spurious.

# we add the lagged dependant variable
df$CAClag <- c(NA,df$CAC[1:nrow(df)-1])
df$Dowlag <- c(NA,df$Dow[1:nrow(df)-1])

# remove row of df with na
fulldf <- df[complete.cases(df),]

# is the coefficient still significant?
model.lag <- lm(CAC ~ Dow +  Dowlag,data = fulldf)
summary(model.lag)

###################################################
#  augmented Dickey-Fuller test (or unit root test)
###################################################

summary(ur.df(df$Dow, type="drift", selectlag="AIC"))
summary(ur.df(df$CAC, type="drift", selectlag="AIC"))

# to compare with the command
adf.test(df$Dow)
adf.test(df$CAC)


# we apply the first difference
dx <- as.data.frame(sapply(df[,c("Dow","CAC")],diff))
rownames(dx) <- df[2:nrow(df),'Date']

# We apply the ADF test on the first differences of log(Dow) and log(CAC 40)
lapply(dx[,c('Dow','CAC')],adf.test)

summary(ur.df(dx$Dow, type="drift", selectlags="AIC"))
summary(ur.df(dx$CAC, type="drift", selectlags="AIC"))


# DF-GLS test
summary( ur.ers(df$Dow, model = "constant") )
summary( ur.ers(dx$Dow, model = "constant") )

##############
# PAC - visual
##############

# Visually, we can show the autocorrelation and partial autocorrelation
acf(df$CAC, type="correlation")
acf(df$Dow, type="correlation")
acf(dx$CAC, type="correlation")
acf(dx$Dow, type="correlation")
acf(df$CAC, type="partial")
acf(df$Dow, type="partial")
acf(dx$CAC, type="partial")
acf(dx$Dow, type="partial")

#############################
# A second regression
#############################

model2 <- lm(CAC~Dow,data = dx)
summary(model2)

####################
# Durbin Watson test
####################
dwtest(model2)

# alternative
library(car)
durbinWatsonTest(resid(model2))

h <- (1 - durbinWatsonTest(resid(model2))/2)*sqrt(nrow(df)/(1 - nrow(df)*0.029**2))

####################
# Cointegration Test
####################

jotest <- ca.jo(df[,1:2],type = "trace")
summary(jotest)

# The null hypothesis of r=0 means that there is no cointegration at all
# A rank r>0 implies a cointegrating relationship between two or possibly more time series.
# Our test statistic is below the 1% level significantly 
# (8.76 < 19.19) we have no cointegration.
# For more info, see https://www.quantstart.com/articles/Johansen-Test-for-Cointegrating-Time-Series-Analysis-in-R

####################
# The Breusch Godfrey test as pointed out in the book seems to indicate that we have second order autocorrelation
####################

bgtest(dx$CAC ~ dx$Dow,order = 2)

# We want to test a final regression
# ∆LCACt = α1 +α2∆LDowt +α3∆LDowt−1 +α4∆LCACt−1 +ut
final <- data.frame(dx$CAC[2:nrow(dx)],dx$Dow[2:nrow(dx)],dx$CAC[1:(nrow(dx)-1)],dx$Dow[(1:nrow(dx)-1)])
colnames(final) <- c("Y","X1","X2","X3")
bgtest(final$Y ~ final$X1+final$X2+final$X3,order = 2)

# The regression model2 <- lm(CAC~Dow,data=dx) has a Breusch Godfrey test statistic above 5.99 indicating second order autocorrelation. 
# The regression model3 <- lm(Y ~ X1+X2+X3,data=final) has a Breusch Godfrey test statistic below 5.99, 
# hence for model 3, we accept both the hypothesise that there are neither first nor second order autocorrelation.

###########################
# Regression of choice here
###########################

# ∆LCACt = α1 +α2∆LDowt +α3∆LDowt−1 +α4∆LCACt−1 +ut
model3 <- lm(Y ~ X1+X2+X3,data = final)
summary(model3)

########################
# Return normality tests
########################
# Shapiro-Wilk test
# H0:  the sample came from a normally distributed population
# we check if the returns are normally distributed
lapply(dx, shapiro.test)
# we reject H0 for all return series, all return series are non-normal

#pdf("DJ_hist.pdf",width=7,height=5)
hist(na.omit(dx$Dow), breaks=100, prob=TRUE, 
     xlab="Dow Jones returns", ylim=c(0, 100), xlim=c(-0.05,0.05),
     main="normal curve over Dow Jones returns histogram")
curve(dnorm(x, mean=mean(na.omit(dx$Dow)), sd=sd(na.omit(dx$Dow))), 
      col="darkblue", lwd=2, add=TRUE, yaxt="n")
#dev.off()

#pdf("CAC_hist.pdf",width=7,height=5)
hist(na.omit(dx$CAC), breaks=100, prob=TRUE, 
     xlab="CAC 40 returns", ylim=c(0, 100), xlim=c(-0.05,0.05),
     main="normal curve over CAC 40 returns histogram")
curve(dnorm(x, mean=mean(na.omit(dx$CAC)), sd=sd(na.omit(dx$CAC))), 
      col="darkblue", lwd=2, add=TRUE, yaxt="n")
#dev.off()

###################
# Scale the returns
###################

# Scale the returns
dx.scale <- as.data.frame(sapply(dx,function(w) { w <- (w-min(w,na.rm=TRUE))/(max(w,na.rm=TRUE)-min(w,na.rm=TRUE)) }))
rownames(dx.scale) <- df[2:nrow(df),'Date']
# Shapiro test
lapply(dx.scale, shapiro.test)

#######################
# univariate ARMA-GARCH
#######################

# plot the returns
dx$Date <- df[2:nrow(df),'Date']
#pdf("CAC_Dow_returns.pdf",width=7,height=5)
mdf <- melt(dx,id.vars = "Date")
ggplot(data = mdf,aes(x=Date,y=value)) + geom_line(aes(color=variable),size=1.25)+scale_x_date("Year")+scale_y_continuous("Returns")
#dev.off()

armaOrder <- c(1,1) # ARMA order, AR(1)
garchOrder <- c(1,1) # GARCH order
# fit a standard GARCH model
varModel <- list(model = "sGARCH", garchOrder = garchOrder)
specm <- ugarchspec(variance.model = varModel,mean.model = list(armaOrder = armaOrder, include.mean = TRUE))
Dowgarch <- ugarchfit(specm,data = dx$Dow)
Dowgarch
CACgarch <- ugarchfit(specm,data = dx$CAC)
CACgarch

# create a data frame with the standardized innovations
df.inno <- as.data.frame(cbind(Dowgarch@fit$residual/Dowgarch@fit$sigma,CACgarch@fit$residual/CACgarch@fit$sigma))
colnames(df.inno) <- c('Dow','CAC')

# Normality test on the standardized innovations
lapply(df.inno,shapiro.test)

hist(na.omit(df.inno$CAC), breaks=100, prob=TRUE, 
     xlab="CAC 40 innovations", ylim=c(0, 0.8), xlim=c(-5,5),
     main="normal curve over CAC 40 innovations histogram")
curve(dnorm(x, mean=mean(na.omit(df.inno$CAC)), sd=sd(na.omit(df.inno$CAC))), 
      col="darkblue", lwd=2, add=TRUE, yaxt="n")

# plot the innovations
df.inno$Date <- df[2:nrow(df),'Date']
#pdf("CAC_Dow_innovations.pdf",width=7,height=5)
mdf <- melt(df.inno,id.vars = "Date")
ggplot(data = mdf,aes(x=Date,y=value)) + geom_line(aes(color=variable),size=1.25)+scale_x_date("Year")+scale_y_continuous("Innovations")
#dev.off()

#####
# DCC
#####

# store the GARCH coefficient for each asset in a coefGarch data.frame
coefGarch <- data.frame(matrix(NA, ncol = 3, nrow = 2))             # Create a matrix to store results
colnames(coefGarch) <- c("omega", "alpha1","beta1")            # Set column and row names
rownames(coefGarch) <- c('Dow','CAC')
# loop over asset retruns and fit a GARCH(1,1) model
for (item in rownames(coefGarch)){
  f1 = garchFit(~ garch(1,1), data=dx[,item], include.mean=FALSE)
  coefGarch[item,] <- f1@fit$coef
}
# create vecotrs and matrices as initial conditions for the DCC estimation
a <- coefGarch$omega
A <- diag(coefGarch$alpha1)
B <- diag(coefGarch$beta1) 
dccpara <- c(0.1,0.8) 
# fit a DCC model to the data
dccresults <- dcc.estimation(inia=a, iniA=A, iniB=B, ini.dcc=dccpara, dvar=dx[,c('Dow','CAC')], model="diagonal")
# output of the model
dccresults$out
# check that both coefficient are 0 < both coef < 1
dccresults$out[13] + dccresults$out[15] 
# model robust standard errors

# Get the correlation evolution over time
DCCrho <- as.data.frame(dccresults$DCC[,2])
rownames(DCCrho) <- rownames(dx)
DCCrho$Date <- dx$Date
colnames(DCCrho) <- c("rho","Date")

# plot the conditional correlations
mdf<-melt(DCCrho,id.vars="Date")
#mdf$value <- as.numeric(levels(mdf$value))[mdf$value]
#pdf("DCC_conditional_correlations_CAC-Dow.pdf",width=15,height=7)
ggplot(data=mdf, aes(x=Date,y=value)) + geom_line(aes(color=variable), size=1.25)+scale_x_date("Year")+scale_y_continuous("Conditional correlation")
#dev.off()

# get the innovations, the standardized residuals
dcc.inno <- as.data.frame(dccresults$std.resid)

# Normality test on the standardized innovations
lapply(dcc.inno,shapiro.test)

hist(na.omit(dcc.inno$CAC), breaks=100, prob=TRUE, 
     xlab="CAC 40 DCC innovations", ylim=c(0, 0.8), xlim=c(-5,5),
     main="normal curve over CAC 40 DCC innovations histogram")
curve(dnorm(x, mean=mean(na.omit(dcc.inno$CAC)), sd=sd(na.omit(dcc.inno$CAC))), 
      col="darkblue", lwd=2, add=TRUE, yaxt="n")

# plot the innovations
dcc.inno$Date <- df[2:nrow(df),'Date']
#pdf("CAC_Dow_DCC_innovations.pdf",width=7,height=5)
mdf <- melt(dcc.inno,id.vars = "Date")
ggplot(data = mdf,aes(x=Date,y=value)) + geom_line(aes(color=variable),size=1.25)+scale_x_date("Year")+scale_y_continuous("Innovations")
#dev.off()

#############################
# ARCH model of the residuals
#############################

# 1st step: testing for the presence of ARCH
# we create a data frame with the square residuals and the lagged square residuals
residmod3 <- model3$residuals
residmod3**2
residmod3[-1]

df.ARCH <- data.frame(resid(model3)[-1]**2,resid(model3)[-length(resid(model3))]**2)
colnames(df.ARCH) <- c("Y","X")
# we regress the squared residuals of our model on their lag
modelARCH <- lm(Y ~ X,data = df.ARCH)
R2 <- summary(modelARCH)$r.squared 
T <- nrow(df.ARCH)
ARCH <- T*R2

# 2nd step: inspect the partial autocorrelation of the squared residuals
acf(residmod3**2, type="partial") # up to order 4 we have significant partial auto-correlation


# we fit an ARCH model on the residual of our model of choice
arch.model <- garchFit(~garch(4,0), data=residmod3, trace=F) # trace=F reduces the amount of output
summary(arch.model)

# apply a t-Student distribution for the residuals of the ARCH and check for a better QQ-plot fit
arch.model.tStudent <- garchFit(~garch(4,0), data=residmod3, trace=F, cond.dist="std")
summary(arch.model.tStudent)
plot(arch.model)
plot(arch.model.tStudent)

# plot the residuals
dfres <- as.data.frame(residmod3)
dfres$Date <- as.Date(df$Date[3:nrow(df)])
#pdf("CACmodelRes.pdf",width=7,height=5)
mdf <- melt(dfres,id.vars = "Date")
ggplot(data = mdf,aes(x=Date,y=value)) + geom_line(aes(color=variable),size=1.25)+scale_x_date("Year")+scale_y_continuous("Model residuals") + theme(legend.position="none")
#dev.off()

# plot the standardized "ARCH-filtered" residuals
dfARCHres <- as.data.frame(arch.model.tStudent@residuals / arch.model.tStudent@sigma.t)
dfARCHres$Date <- as.Date(df$Date[3:nrow(df)])
#pdf("ARCHCACres.pdf",width=7,height=5)
mdf <- melt(dfARCHres,id.vars = "Date")
ggplot(data = mdf,aes(x=Date,y=value)) + geom_line(aes(color=variable),size=1.25)+scale_x_date("Year")+scale_y_continuous("ARCH residuals") + theme(legend.position="none")
#dev.off()