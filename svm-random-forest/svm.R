# Prepare Workspace
rm(list=ls())
wd <- getwd()
setwd(wd) # get to workspace with pubfig training set
#setwd("~/path/to/project")

# Relevant Classification Packages
library("base64")
library("e1071")
library("RANN")
library("randomForest")
library("klaR") 
library("caret")

# Read in Data
face_attr <- read.table("pubfig_train_50000_pairs.txt", header=FALSE, sep="") 
attr <- read.csv("pubfig_attributes.csv") # for approximate nearest neighbors
xtrain <- face_attr[,-c(1)] # attributes of both faces 1 and 2
x_attr <- attr[,c(3:75)]
xval1 <- read.table("pubfig_kaggle_1.txt", header=FALSE, sep="") # read in all 3 evaluation sets and 1 test set
xval2 <- read.table("pubfig_kaggle_2.txt", header=FALSE, sep="") 
xval3 <- read.table("pubfig_kaggle_3.txt", header=FALSE, sep="")
xtst <- read.table("pubfig_kaggle_eval.txt", header=FALSE, sep="") 
ytrain <- face_attr[,1] # 50000 labels
y_attr <- attr[,c(1)]
yvaltmp1 <- read.table("pubfig_kaggle_1_solution.txt", header=TRUE, sep=",") 
yvaltmp2 <- read.table("pubfig_kaggle_2_solution.txt", header=TRUE, sep=",") 
yvaltmp3 <- read.table("pubfig_kaggle_3_solution.txt", header=TRUE, sep=",")
yval1 <- yvaltmp1[,c(2)]
yval2 <- yvaltmp2[,c(2)]
yval3 <- yvaltmp3[,c(2)]

# Part A
# LINEAR SVM

# training set, svm
ytrn_fac <- as.factor(ytrain)
svm_trn <- svmlight(xtrain, ytrn_fac, pathsvm='/home/adhamid2/Downloads/svm_light_linux64')
lbls_svm_trn <- predict(svm_trn, xtrain)
lbls_svm_trn_ <- lbls_svm_trn$class
svm_trn_acc <- sum(lbls_svm_trn_==ytrn_fac)/(sum(lbls_svm_trn_==ytrn_fac)+sum(!(lbls_svm_trn_==ytrn_fac)))

# validation sets 1, 2, and 3; svm
# REPLACE svmlight PATH with the proper path in submission
# Feed validation set model into next model successively
if(FALSE){
yval1_fac <- as.factor(yval1)
lbls_svm_val1 <- predict(svm_trn, xval1)
lbls_svm_val1_ <- lbls_svm_val1$class
svm_val1_acc <- sum(lbls_svm_val1_==yval1_fac)/(sum(lbls_svm_val1_==yval1_fac)+sum(!(lbls_svm_val1_==yval1_fac)))

svm_val1 <- svmlight(xval1, lbls_svm_val1_, pathsvm='/home/adhamid2/Downloads/svm_light_linux64')

yval2_fac <- as.factor(yval2)
lbls_svm_val2 <- predict(svm_val1, xval2)
lbls_svm_val2_ <- lbls_svm_val2$class
svm_val2_acc <- sum(lbls_svm_val2_==yval2_fac)/(sum(lbls_svm_val2_==yval2_fac)+sum(!(lbls_svm_val2_==yval2_fac)))

svm_val2 <- svmlight(xval2, lbls_svm_val2_, pathsvm='/home/adhamid2/Downloads/svm_light_linux64')

yval3_fac <- as.factor(yval3)
lbls_svm_val3 <- predict(svm_val2, xval3)
lbls_svm_val3_ <- lbls_svm_val3$class
svm_val3_acc <- sum(lbls_svm_val3_==yval3_fac)/(sum(lbls_svm_val3_==yval3_fac)+sum(!(lbls_svm_val3_==yval3_fac)))

svm_val3 <- svmlight(xval3, lbls_svm_val3_, pathsvm='/home/adhamid2/Downloads/svm_light_linux64')
}

# Models built on individual validation sets
yval1_fac <- as.factor(yval1)
svm_val1 <- svmlight(xval1, yval1_fac, pathsvm='/home/adhamid2/Downloads/svm_light_linux64')
lbls_svm_val1 <- predict(svm_val1, xval1)
lbls_svm_val1_ <- lbls_svm_val1$class
svm_val1_acc <- sum(lbls_svm_val1_==yval1_fac)/(sum(lbls_svm_val1_==yval1_fac)+sum(!(lbls_svm_val1_==yval1_fac)))

yval2_fac <- as.factor(yval2)
svm_val2 <- svmlight(xval2, yval2_fac, pathsvm='/home/adhamid2/Downloads/svm_light_linux64')
lbls_svm_val2 <- predict(svm_val2, xval2)
lbls_svm_val2_ <- lbls_svm_val2$class
svm_val2_acc <- sum(lbls_svm_val2_==yval2_fac)/(sum(lbls_svm_val2_==yval2_fac)+sum(!(lbls_svm_val2_==yval2_fac)))

yval3_fac <- as.factor(yval3)
svm_val3 <- svmlight(xval3, yval3_fac, pathsvm='/home/adhamid2/Downloads/svm_light_linux64')
lbls_svm_val3 <- predict(svm_val3, xval3)
lbls_svm_val3_ <- lbls_svm_val3$class
svm_val3_acc <- sum(lbls_svm_val3_==yval3_fac)/(sum(lbls_svm_val3_==yval3_fac)+sum(!(lbls_svm_val3_==yval3_fac)))


# NAIVE BAYES

# training set, naive bayes
wtd_trn <- createDataPartition(y=ytrain, p=.8, list=FALSE)
trax <- xtrain[wtd_trn,]
tray <- as.factor(ytrain[wtd_trn])
mdl_trn <- train(trax, tray, 'nb', trControl=trainControl(method='cv', number=10))
classes_trn <- predict(mdl_trn,newdata=xtrain[-wtd_trn,])
confusionMatrix(data=classes_trn,ytrain[-wtd_trn])

# validation sets 1, 2, and 3; naive bayes 
wtd_val1 <- createDataPartition(y=yval1, p=.8, list=FALSE)
val1x <- xval1[wtd_val1,]
val1y <- as.factor(yval1[wtd_val1])
mdl_val1 <- train(val1x, val1y, 'nb', trControl=trainControl(method='cv', number=10))
classes_val1 <- predict(mdl_val1,newdata=xval1[-wtd_val1,])
confusionMatrix(data=classes_val1,yval1[-wtd_val1])

wtd_val2 <- createDataPartition(y=yval2, p=.8, list=FALSE)
val2x <- xval1[wtd_val2,]
val2y <- as.factor(yval2[wtd_val2])
mdl_val2 <- train(val2x, val2y, 'nb', trControl=trainControl(method='cv', number=10))
classes_val2 <- predict(mdl_val2,newdata=xval2[-wtd_val2,])
#confusionMatrix(data=classes_val2,yval2[-wtd_val2])

wtd_val3 <- createDataPartition(y=yval3, p=.8, list=FALSE)
val3x <- xval1[wtd_val3,]
val3y <- as.factor(yval3[wtd_val3])
mdl_val3 <- train(val3x, val3y, 'nb', trControl=trainControl(method='cv', number=10))
classes_val3 <- predict(mdl_val3,newdata=xval3[-wtd_val3,])
#confusionMatrix(data=classes_val3,yval3[-wtd_val3])


# RANDOM FOREST

# training set, random forest
ytrn_fac <- as.factor(ytrain)
rf_trn <- randomForest(xtrain,y=ytrn_fac,ntree=5)
lbls_rf_trn <- predict(rf_trn,xtrain)
rf_trn_acc <- sum(lbls_rf_trn==ytrn_fac)/(sum(lbls_rf_trn==ytrn_fac)+sum(!(lbls_rf_trn==ytrn_fac)))

# validation sets 1, 2, and 3; random forest
# REPLACE svmlight PATH with the proper path in submission
# Feed validation set model into next model successively
if(FALSE){
yval1_fac <- as.factor(yval1)
rf_val1 <- randomForest(x=xval1,y=yval1_fac,ntree=5)
lbls_rf_val1 <- predict(rf_val1,xval1)
rf_val1_acc <- sum(lbls_rf_val1==yval1_fac)/(sum(lbls_rf_val1==yval1_fac)+sum(!(lbls_rf_val1==yval1_fac)))

yval2_fac <- as.factor(yval2)
rf_val2 <- randomForest(xval2,y=lbls_rf_val1,ntree=5)
lbls_rf_val2 <- predict(rf_val2,xval2)
rf_val2_acc <- sum(lbls_rf_val2==yval2_fac)/(sum(lbls_rf_val2==yval2_fac)+sum(!(lbls_rf_val2==yval2_fac)))

yval3_fac <- as.factor(yval3)
rf_val3 <- randomForest(xval3,y=lbls_rf_val2,ntree=5)
lbls_rf_val3 <- predict(rf_val3,xval3)
rf_val3_acc <- sum(lbls_rf_val1==yval3_fac)/(sum(lbls_rf_val1==yval3_fac)+sum(!(lbls_rf_val1==yval3_fac)))
}

# Models built on individual validation sets 
yval1_fac <- as.factor(yval1)
rf_val1 <- randomForest(xval1,y=yval1_fac,ntree=5)
lbls_rf_val1 <- predict(rf_val1,xval1)
rf_val1_acc <- sum(lbls_rf_val1==yval1_fac)/(sum(lbls_rf_val1==yval1_fac)+sum(!(lbls_rf_val1==yval1_fac)))

yval2_fac <- as.factor(yval2)
rf_val2 <- randomForest(xval2,y=yval2_fac,ntree=5)
lbls_rf_val2 <- predict(rf_val1,xval2)
rf_val2_acc <- sum(lbls_rf_val2==yval2_fac)/(sum(lbls_rf_val2==yval2_fac)+sum(!(lbls_rf_val2==yval2_fac)))

yval3_fac <- as.factor(yval3)
rf_val3 <- randomForest(xval3,y=yval3_fac,ntree=5)
lbls_rf_val3 <- predict(rf_val3,xval3)
rf_val3_acc <- sum(lbls_rf_val1==yval3_fac)/(sum(lbls_rf_val1==yval3_fac)+sum(!(lbls_rf_val1==yval3_fac)))


# Part B
# Approximate nearest neighbor

# training set, approximate nearest neighbor
xtrain1 <- xtrain[,c(1:73)]
xtrain2 <- xtrain[,c(74:146)]
ann_trn_pred1 <- nn2(x_attr,xtrain1,k=1)
ann_trn_pred2 <- nn2(x_attr,xtrain2,k=1)
ann_trn_names1 <- y_attr[ann_trn_pred1$nn.idx]
ann_trn_names2 <- y_attr[ann_trn_pred2$nn.idx]
correct_trn <- ann_trn_names1==ann_trn_names2
ann_trn_acc <- sum(correct_trn==ytrain)/(sum(correct_trn==ytrain)+sum(!(correct_trn==ytrain)))

# validation sets 1, 2, and 3; approximate nearest neighbor
xval11 <- xval1[,c(1:73)]
xval12 <- xval1[,c(74:146)]
ann_val1_pred1 <- nn2(x_attr,xval11,k=1)
ann_val1_pred2 <- nn2(x_attr,xval12,k=1)
ann_val1_names1 <- y_attr[ann_val1_pred1$nn.idx]
ann_val1_names2 <- y_attr[ann_val1_pred2$nn.idx]
correct_val1 <- ann_val1_names1==ann_val1_names2
ann_val1_acc <- sum(correct_val1==yval1)/(sum(correct_val1==yval1)+sum(!(correct_val1==yval1)))

xval21 <- xval2[,c(1:73)]
xval22 <- xval2[,c(74:146)]
ann_val2_pred1 <- nn2(x_attr,xval21,k=1)
ann_val2_pred2 <- nn2(x_attr,xval22,k=1)
ann_val2_names1 <- y_attr[ann_val2_pred1$nn.idx]
ann_val2_names2 <- y_attr[ann_val2_pred2$nn.idx]
correct_val2 <- ann_val2_names1==ann_val2_names2
ann_val2_acc <- sum(correct_val2==yval2)/(sum(correct_val2==yval2)+sum(!(correct_val2==yval3)))

xval31 <- xval3[,c(1:73)]
xval32 <- xval3[,c(74:146)]
ann_val3_pred1 <- nn2(x_attr,xval31,k=1)
ann_val3_pred2 <- nn2(x_attr,xval32,k=1)
ann_val3_names1 <- y_attr[ann_val3_pred1$nn.idx]
ann_val3_names2 <- y_attr[ann_val3_pred2$nn.idx]
correct_val3 <- ann_val3_names1==ann_val3_names2
ann_val3_acc <- sum(correct_val3==yval3)/(sum(correct_val3==yval3)+sum(!(correct_val3==yval3)))


# Part C
# Unrestricted Classifier, applied to test set

#rf_tst <- randomForest(xtst) # labels are omitted -> runs in UNSUPERVISED MODE
#lbls_rf_tst <- predict(rf_tst,xtst)
#rf_tst <- train(xtst,method="rf",trControl=trainControl(method="oob"))

#
lbls_rf_tst <- predict(rf_val1, xtst)
lbls_rf_tst_ <- lbls_rf_tst.predicted
write.table(lbls_svm_tst_,file="labels_partC_adhamid2.csv", col.names = c("Id,Prediction"), sep=",", quote = FALSE)

