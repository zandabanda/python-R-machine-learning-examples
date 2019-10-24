# use only continuous features (1-age, 3-fnlwgt, 5-educationnum, 11-captialgain, 12-capitalloss, 13-hoursperweek)
# prediction task: does the person make 50,000 dollars a year?

rm(list=ls())
wd <- getwd()
setwd(wd) # get to workspace with Adult dataset

#library("base64")
#library("e1071")
library("klaR") # relevant classification packages
library("caret")

raw <- read.table("adult.data.txt", header=FALSE, sep=",") # 30,000 some instances, 14 features, binary label
x <- raw[,-c(2,4,6,7,8,9,10,14,15)] # all CONTINUOUS feature vectors
y <- raw[,15] # all labels

for (i in c(1,2,3,4,5,6)) # replace '?' with 'NA'
{
  tmp <- x[,i]=='?'
  x[tmp,i] = NA
}

y <- sapply(y, function(x) as.numeric(x)) # convert factor datatype of labels to numerics, and replace with +/- 1 accordingly
for (j in 1:length(y)) 
{
  if (y[j] == 1){
    y[j] <- -1
  }
  else if (y[j] == 2){
    y[j] <- 1
  }
}

xmean <- sapply(x, mean, na.rm=TRUE) # standardize the dataset
xstd <- sapply(x, sd, na.rm=TRUE)
xoffsets <- x
xnew <- x
for(i in 1:32561) {
  xoffsets[i,] = x[i,]-xmean
  xnew[i,] <- xoffsets[i,]/xstd
}
x <- xnew

# For all three regularization constants, train the model 
# Keep model that has lowest error rate

N <- 32561 # number of observations/instances
Ne <- 50 # Choose Ne = 50 epochs (at least), each of which has 300 steps Ns (at least)
Ns <- 300

a <- rep(0,6) # Choose random starting point, zeros in our case
b <- 0
#u0 <- append(a0,b0)
c <- .01 # choose c,d s.t. 1/(c+d) = .001 and 1/(c*30 + d) = 10E-8, determined empirically
d <- 50
acceval <- array(dim=500)
accval <- array(dim=50)
acctst <- array(dim=1)
#e = 1

for (e in 1:Ne) # For all 50 epochs, compute the steplength 
{
  stl <- 1/(c*e + d) # define steplength for this epoch
  lamb <- c(e-3,e-2,e-1,e) 
  lamb <- lamb[1] # SELECT WHICH REGULARIZATION PARAMETER YOU WANT HERE, indexes 1-4
  
  wtd1 <- createDataPartition(y=y, p=.8, list=FALSE) # Split into 80% training and everything else
  eval <- createDataPartition(y=y[wtd1], p=0.001919459, list=FALSE) # Select 50 of training examples at random for evaluation
  wtd2 <- createDataPartition(y=y[-wtd1], p=.5, list=FALSE) # Split remaining 20% in half for test and validation

  trax <- x[wtd1,][-eval,] # training sets
  tray <- y[wtd1][-eval] 
  evalx <- x[eval,] # evaluation sets, 50 samples chosen at random
  evaly <- y[eval]
  tstx <- x[wtd2,] # test sets
  tsty <- y[wtd2]
  valx <- x[-wtd1,][-wtd2,] # validations sets
  valy <- y[-wtd1][-wtd2]

  penalty <- 0
  hinge <- 0
  loss <- 0
  
  #for (obs in 1:25999) # CALCULATE the loss function.  Uncommenting severly slows down the execution time!!!
    #{
    #tmp <- trax[obs,]
    #hinge <- hinge + (1/N)*max(0,1 - tray[obs]*(sum(a*trax[obs,]) + b))
    #} 

  penalty <- lamb*(sum(a*a)/2)
  loss <- hinge + penalty
  
  for (steps in 1:Ns)
  { 
    k <- sample(1:25999,1,replace=TRUE) # For each step, select a data point uniformly at random, with replacement
    gammak <- tray[k]*(sum(a*trax[k,]) + b)
    
    if (gammak >= 1){
      grada <- lamb*a
      gradb <- 0
      a <- a - stl*grada
      b <- b - stl*gradb
    }
    else {
      grada <- lamb*a - tray[k]*trax[k,]
      gradb <- -tray[k]
      a <- a - stl*grada
      b <- b - stl*gradb
    } 
    
    if (steps%%30 == 0){
      predeval <- rep(0,50) 
      for (ex in 1:50){
        predeval[ex] <- -1 + 2*((sum(a*evalx[ex,])+b)>0)
      }
      coreval <- predeval == evaly
      acceval[floor(steps/30)+(e-1)*10]<-sum(coreval)/length(coreval)#(sum(correct)+sum(!correct))
    }

  }
  
  # Calculate accuracy on validation set, at the end of each epoch
  predval <- rep(0,3256) 
  for (ex in 1:3256){
    predval[ex] <- -1 + 2*((sum(a*valx[ex,])+b)>0)
  }
  corval <- predval == valy
  accval[e]<-sum(corval)/length(corval)
}

# Calculate accuracy on test set, after all epochs completed
predtst <- rep(0,3256) 
for (ex in 1:3256){
  predtst[ex] <- -1 + 2*((sum(a*tstx[ex,])+b)>0)
}
cortst <- predtst == tsty
acctst[1]<-sum(cortst)/length(cortst)
