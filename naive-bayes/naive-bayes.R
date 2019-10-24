# Naive Bayes on Pima Indian Diabetes Dataset from UC Irvine Archives
# Class Conditional Densities (posterior model) are assumed to be NORMAL

# Part (a)

rm(list=ls())
wd <- getwd()
setwd(wd) # get to workspace with Pima Dataset

library("base64")
library("e1071")
library("klaR") # relevant classification packages
library("caret")

raw <- read.table("pima-indians-diabetes.data.txt", header=FALSE, sep=",") # 700 something instances of 8 features, labels of (yes/no) diabetes
x <- raw[,-c(9)] # all feature vectors, everything but the 9th column (labels)
y <- raw[,9] # all 700 something labels, everything but features

trscore <- array(dim=10)
tescore <- array(dim=10)

for (i in 1:10)
  {
wtd <- createDataPartition(y=y, p=.8, list=FALSE) # cross validation, take 80 percent of the original 700 something instances
trbx <- x[wtd, ] # training set of feature instances, for all 8 features
trby <- y[wtd] # training set of labels for all 700 something instances
tebx <- x[-wtd, ] # test partition of instances
teby <- y[-wtd] # test partition of labels

trposflag <- trby>0 # keep track of when the labels are +1
ptregs <- trbx[trposflag, ] # for +1 labels, find the corresonding instances in each feature
ntregs <- trbx[!trposflag,] # for -1 labels, find the corresonding instances in each feature

# Our model is Gaussian, need only means and variances
ptrmean <- sapply(ptregs, mean, na.rm=TRUE) # acquire mean of EACH feature for instances of +1
ntrmean <- sapply(ntregs, mean, na.rm=TRUE) # acquire mean of EACH for instances of -1
ptrsd <- sapply(ptregs, sd, na.rm=TRUE) # acquire std of EACH feature for instances of +1
ntrsd <- sapply(ntregs, sd, na.rm=TRUE) # acquire std of EACH feature for instances of -1

ptroffsets <- t(t(trbx)-ptrmean) # subtract +1 means off training set
ptrscales <- t(t(ptroffsets)/ptrsd) # standardize training set , divide by std of +1 isntances
ntroffsets<-t(t(trbx)-ntrmean) # subtract -1 means off training set
ntrscales<-t(t(ntroffsets)/ntrsd) # standardize training set, divide by std of -1 instances

# na.rm specifies that any operation on NA returns NA
ptrlogs <- -(1/2)*rowSums(apply(ptrscales, c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd)) # Loss of false negative for EACH instance
ntrlogs <- -(1/2)*rowSums(apply(ntrscales, c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd)) # Loss of false positive ''

lvwtr <- ptrlogs>ntrlogs # decision rule; compare losses
gotrighttr <- lvwtr==trby #

trscore[i] <- sum(gotrighttr)/(sum(gotrighttr)+sum(!gotrighttr)) # success rate of classification on training set

pteoffsets <- t(t(tebx)-ptrmean) # subtract training mean of +1 instances off validation set
ptescales <- t(t(pteoffsets)/ptrsd) # standardize by +1 instance std
nteoffsets <- t(t(tebx)-ntrmean) # subtract training mean of -1 instances off validation set
ntescales <- t(t(nteoffsets)/ntrsd) # standardize by -1 instance std

ptelogs <- -(1/2)*rowSums(apply(ptescales, c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd)) # square all entires, subtract log std's, sum over all features
ntelogs <- -(1/2)*rowSums(apply(ntescales, c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd))

lvwte <- ptelogs>ntelogs
gotrightte <- lvwte==teby

tescore[i] <-sum(gotrightte)/(sum(gotrightte)+sum(!gotrightte)) # success rate of classification on test set
}

mean_trscore <- mean(trscore)
std_trscore <- sd(trscore)
mean_tescore <- mean(tescore)
std_tescore <- sd(tescore)

# Part (b)

rm(list=ls())
raw <- read.table("pima-indians-diabetes.data.txt", header=FALSE, sep=",") # 700 something instances of 8 features, labels of (yes/no) diabetes
x <- raw[,-c(9)] # all feature vectors, everything but the 9th column (labels)
y <- raw[,9] # all 700 something labels, everything but features

for (i in c(3,4,6,8)) # replace 0 with NA in features 3,4,6, and 8
{
  tmp <- x[,i]==0
  x[tmp,i] = NA
}

trscore <- array(dim=10)
tescore <- array(dim=10)

for (i in 1:10)
{
  wtd <- createDataPartition(y=y, p=.8, list=FALSE) # cross validation, take 80 percent of the original 700 something instances
  trbx <- x[wtd, ] # training set of feature instances, for all 8 features
  trby <- y[wtd] # training set of labels for all 700 something instances
  tebx <- x[-wtd, ] # test partition of instances
  teby <- y[-wtd] # test partition of labels

  trposflag <- trby>0 # keep track of when the labels are +1
  ptregs <- trbx[trposflag, ] # for +1 labels, find the corresonding instances in each feature
  ntregs <- trbx[!trposflag,] # for -1 labels, find the corresonding instances in each feature

  # Our model is Gaussian, need only means and variances
  ptrmean <- sapply(ptregs, mean, na.rm=TRUE) # acquire mean of EACH feature for instances of +1
  ntrmean <- sapply(ntregs, mean, na.rm=TRUE) # acquire mean of EACH for instances of -1
  ptrsd <- sapply(ptregs, sd, na.rm=TRUE) # acquire std of EACH feature for instances of +1
  ntrsd <- sapply(ntregs, sd, na.rm=TRUE) # acquire std of EACH feature for instances of -1

  ptroffsets <- t(t(trbx)-ptrmean) # subtract +1 means off training set
  ptrscales <- t(t(ptroffsets)/ptrsd) # standardize training set , divide by std of +1 isntances
  ntroffsets<-t(t(trbx)-ntrmean) # subtract -1 means off training set
  ntrscales<-t(t(ntroffsets)/ntrsd) # standardize training set, divide by std of -1 instances

  # na.rm specifies that any operation on NA returns NA
  ptrlogs <- -(1/2)*rowSums(apply(ptrscales, c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd)) # Loss of false negative for EACH instance
  ntrlogs <- -(1/2)*rowSums(apply(ntrscales, c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd)) # Loss of false positive ''

  lvwtr <- ptrlogs>ntrlogs # decision rule; compare losses
  gotrighttr <- lvwtr==trby #

  trscore[i] <- sum(gotrighttr)/(sum(gotrighttr)+sum(!gotrighttr)) # success rate of classification on training set

  pteoffsets <- t(t(tebx)-ptrmean) # subtract training mean of +1 instances off validation set
  ptescales <- t(t(pteoffsets)/ptrsd) # standardize by +1 instance std
  nteoffsets <- t(t(tebx)-ntrmean) # subtract training mean of -1 instances off validation set
  ntescales <- t(t(nteoffsets)/ntrsd) # standardize by -1 instance std

  ptelogs <- -(1/2)*rowSums(apply(ptescales, c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ptrsd)) # square all entires, subtract log std's, sum over all features
  ntelogs <- -(1/2)*rowSums(apply(ntescales, c(1, 2), function(x)x^2), na.rm=TRUE)-sum(log(ntrsd))

  lvwte <- ptelogs>ntelogs
  gotrightte <- lvwte==teby

  tescore[i] <-sum(gotrightte)/(sum(gotrightte)+sum(!gotrightte)) # success rate of classification on test set
}

mean_trscore <- mean(trscore)
std_trscore <- sd(trscore)
mean_tescore <- mean(tescore)
std_tescore <- sd(tescore)

# Part (c)

rm(list=ls())
raw <- read.table("pima-indians-diabetes.data.txt", header=FALSE, sep=",") # 700 something instances of 8 features, labels of (yes/no) diabetes
x <- raw[,-c(9)] # all feature vectors, everything but the 9th column (labels)
y <- as.factor(raw[,9]) # set of possible outcomes (binary)

wtd<-createDataPartition(y=y, p=.8, list=FALSE)
trax <- x[wtd,]
tray <- y[wtd]
model <- train(trax, tray, 'nb', trControl=trainControl(method='cv', number=10))
teClasses <- predict(model,newdata=x[-wtd,])
confusionMatrix(data=teClasses,y[-wtd])

# Part (d)

rm(list=ls())
raw <- read.table("pima-indians-diabetes.data.txt", header=FALSE, sep=",") # 700 something instances of 8 features, labels of (yes/no) diabetes
x <- raw[,-c(9)] # all feature vectors, everything but the 9th column (labels)
y <- as.factor(raw[,9])
wtd <- createDataPartition(y=y, p=.8, list=FALSE)

svm <- svmlight(x[wtd,], y[wtd], pathsvm='/path/to/svmlight')
labels <- predict(svm, x[-wtd,])
foo <- labels$class
sum(foo==y[-wtd])/(sum(foo==y[-wtd])+sum(!(foo==y[-wtd])))
