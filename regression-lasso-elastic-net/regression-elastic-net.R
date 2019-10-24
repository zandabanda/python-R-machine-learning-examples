library(glmnet)
#libary(xlsx)
setwd("/path/to/project")

#data <- read.xlsx("default of credit card clients.xls")  #SUPER slow
data <- read.csv("default of credit card clients.csv")
data <- data[-1,][-1,]  #remove first two lines
x <- data.matrix(data[,-25])
y <- data.matrix(data[,25])

train = sample(1:nrow(data), round(0.7*nrow(data)))
xTrain  = x[train, ]
xTest   = x[-train, ]
yTrain  = y[train]
yTest   = y[-train]

cvfit = cv.glmnet(xTrain, yTrain, family = "binomial", type.measure = "class")
plot(cvfit)
cvfitDev = cv.glmnet(xTrain, yTrain, family = "binomial", type.measure = "deviance")
plot(cvfitDev)

cvfit$lambda.min  #the regularization constant that minimizes the misclassification error
# [1] 0.001671603   0.0006593147    0.000600743   0.0008715752
cvfitDev$lambda.min
# [1] 0.0006748363    0.001152171   0.000600743    0.000600743
# 
coef(cvfit, s = cvfit$lambda.min)   #the coefficients of the model with this lambda
# 25 x 1 sparse Matrix of class "dgCMatrix"
# 1
# (Intercept) -2.314510e+00
# V1           .           
# V2          -7.654516e-04
# V3          -9.762799e-02
# V4          -7.296548e-02
# V5          -1.181635e-01
# V6           3.959669e-03
# V7           4.042786e-01
# V8           1.001149e-01
# V9           1.241743e-01
# V10          1.180863e-01
# V11         -5.213104e-02
# V12         -3.116175e-03
# V13          .           
# V14          .           
# V15          .           
# V16         -1.283278e-06
# V17         -1.138376e-06
# V18          .           
# V19         -5.533610e-05
# V20         -2.957133e-05
# V21         -1.153288e-05
# V22         -3.844739e-05
# V23         -1.484950e-05
# V24         -8.662734e-06

yPredict <- predict(cvfit, as.matrix(xTest), s = cvfit$lambda.min, type = "class")
sum(yTest == yPredict)/length(yTest)
#[1] 0.7959773

L1 = cv.glmnet(xTrain, yTrain, family = "binomial", type.measure = "class", alpha = 1)
L1Dev = cv.glmnet(xTrain, yTrain, family = "binomial", type.measure = "deviance", alpha = 1)
plot(L1)
plot(L1Dev)
L1$lambda.min
#[1] 0.0007941469 #0.0006593147 #0.001523102 #0.000600743 #pg169
L1Dev$lambda.min
#[1] 0.001049815   0.000600743    0.001152171   0.0006593147
sum(predict(L1, as.matrix(xTest), s = L1$lambda.min, type = "class") == yTest)/length(yTest)
#[1] 0.7964218

L2 = cv.glmnet(xTrain, yTrain, family = "binomial", type.measure = "class", alpha = 0)
L2Dev = cv.glmnet(xTrain, yTrain, family = "binomial", type.measure = "deviance", alpha = 0)
plot(L2)
plot(L2Dev)
L2$lambda.min
#[1] 0.01324717   0.01324717    0.01453875    0.01324717
L2Dev$lambda.min
#[1] 0.01324717   0.01324717    0.01324717    0.01324717
sum(predict(L2, as.matrix(xTest), s = L2$lambda.min, type = "class") == yTest)/length(yTest)
#[1] 0.7949772

elasticNet = cv.glmnet(xTrain, yTrain, family = "binomial", type.measure = "class", alpha = 0.5)
elasticNetDev = cv.glmnet(xTrain, yTrain, family = "binomial", type.measure = "deviance", alpha = 0.5)
plot(elasticNet)
plot(elasticNetDev)
elasticNet$lambda.min
#[1] 0.001094749  #0.001201486 #0.001094749 #0.001094749
elasticNetDev$lambda.min
#[1] 0.00174315   #0.001447194  #0.001201486  #0.001588294
sum(predict(elasticNet, as.matrix(xTest), s = elasticNet$lambda.min, type = "class") == yTest)/length(yTest)
#[1] 0.7965329


