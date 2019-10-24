# Prepare Workspace
rm(list=ls())
#wd <- getwd()
#setwd(wd) 
#setwd("/path/to/project")

# Relevant Packages
library("base64")
library("e1071")
library("RANN")
library("randomForest")
library("klaR") 
library("caret")
library("lattice")
library('chemometrics')
library('plspm')
library('plsdepot')
library('rgl')
library('cluster')
library('MASS') 
library('stats')
library('jpeg')
library('matrixStats')
library('glmnet')
library('xlsx')


#geog <- read.table('default_features_1059_tracks.txt', sep = ',')
geog_plus <- read.table('default_plus_chromatic_features_1059_tracks.txt', sep = ',') # use extended feature set
partition <- .7
trn_x <- sample(nrow(geog_plus), round(partition*nrow(geog_plus)))
X_trn <- geog_plus[trn_x,]
X_tst <- geog_plus[-trn_x,]
feat <- X_trn[,1:116]
latlon <- X_trn[,117:118]
feat_tst <- X_tst[,1:116]
latlon_tst <- X_tst[,117:118]

regr_lat <- lm(latlon$V117 ~ as.matrix(feat)) # linear model, latitude regressed against all 116 features
summary(regr_lat)$r.squared # R-squared of regression
## [1] 0.3524599
plot(regr_lat)
plot(latlon$V117,regr_lat$fitted.values) # look for high correlation, adjust axes
title('predicted vs predictors, latitude')

regr_lon <- lm(latlon$V118 ~ as.matrix(feat))
summary(regr_lon)$r.squared 
## [1] 0.3839337
plot(regr_lon)
plot(latlon$V118,regr_lon$fitted.values) 
title('predicted vs predictors, longtiude')

origin_shift <- 90 # make all values positive, shifting origin of spatial coordintes, necessary for boxcox transfomration
regr_lat_shft <- lm((latlon$V117 + origin_shift) ~ as.matrix(feat))
box_regr_lat <- boxcox(regr_lat_shft, lambda = seq(-10, 20, 1/10), plotit = TRUE)  # remember to undo translation
## lambda_best ~ 4

regr_lon_shft <- lm((latlon$V118 + origin_shift) ~ as.matrix(feat))
box_regr_lon <- boxcox(regr_lon_shft, lambda = seq(-5, 5, 1/10), plotit = TRUE)
## lambda_best ~ 1

ridge_regr_lat <- cv.glmnet(as.matrix(feat), latlon$V117, alpha = 0)
lamin_ridge <- ridge_regr_lat$lambda.min
## [1] 2.970351
lalse_ridge <- ridge_regr_lat$lambda.1se
## [1] 36.61988
plot(ridge_regr_lat)
title('ridge regression, latitude')

ridge_regr_lon <- cv.glmnet(as.matrix(feat), latlon$V118, alpha = 0)
lomin_ridge <- ridge_regr_lon$lambda.min
## [1] 20.00851
lolse_ridge <- ridge_regr_long$lambda.1se
## [1] 297.12
plot(ridge_regr_lon)
title('ridge regression, longitude')

lasso_regr_lat <- cv.glmnet(as.matrix(feat), latlon$V117, alpha = 1)
lamin_lasso <- lasso_regr_lat$lambda.min
## [1] 0.3415182
lalse_lasso <- lasso_regr_lat$lambda.1se
## [1] 1.82258
# 9 nonzero variables used in this regression.
plot(lasso_regr_lat)
title('lasso regression, latitude')

lasso_regr_lon <- cv.glmnet(as.matrix(feat), latlon$V118, alpha = 1)
lomin_lasso <- lasso_regr_lon$lambda.min
## [1] 0.3260889
lolse_lasso <- lasso_regr_lon$lambda.1se
## [1] 3.337621
# 32 nonzero variables used in this regression.
plot(lasso_regr_lon)
title('lasso regression, longitude')

# try 3 values of alpha
alphas <- c(.25, 5, .75)
elnet_regr_lat <- cv.glmnet(as.matrix(feat), latlon$V117, alpha = .8)
lamin_elnet <- elnet_regr_lat$lambda.min
## [1] 0.3415182
lalse_elnet <- elnet_regr_lat$lambda.1se
## [1] 2.195304
## this model uses 11 nonzero coefficients
plot(elnet_regr_lat)
title('elastic net, latitude')

elnet_regr_lon <- cv.glmnet(as.matrix(feat), latlon$V118, alpha = .8)
lomin_elnet <- elnet_regr_lon$lambda.min
## [1] 0.5192257
lolse_elnet <- elnet_regr_lon$lambda.1se
## [1] 3.337621
## this model uses 35 nonzero coefficients
plot(elnet_regr_lon)
title('elastic net, longitude')

# apply model to test predictors
beta_lat = regr_lat$coefficients
beta_lat[is.na(beta_lat)] <- 0 
beta_lon = regr_lon$coefficients
beta_lon[is.na(beta_lon)] <- 0 
yhat_lm_lat = cbind(rep(1, nrow(geog_plus) - length(trn_x)), as.matrix(feat_tst))%*%beta_lat 
yhat_lm_lon = cbind(rep(1, nrow(geog_plus) - length(trn_x)), as.matrix(feat_tst))%*%beta_lon 
yhat_ridge_lat = predict(ridge_regr_lat, as.matrix(feat_tst), s = lamin_ridge)
yhat_ridge_lon = predict(ridge_regr_lon, as.matrix(feat_tst), s = lomin_ridge)
yhat_lasso_lat = predict(lasso_regr_lat, as.matrix(feat_tst), s = lamin_lasso)
yhat_lasso_lon = predict(lasso_regr_lon, as.matrix(feat_tst), s = lomin_lasso)
yhat_elnet_lat = predict(elnet_regr_lat, as.matrix(feat_tst), s = lamin_elnet)
yhat_elnet_lon = predict(elnet_regr_lon, as.matrix(feat_tst), s = lomin_elnet)

# Mean squared errors
sum((latlon_tst$V117 - yhat_lm_lat)^2)/nrow(feat_tst)
sum((latlon_tst$V117 - yhat_ridge_lat)^2)/nrow(feat_tst)
sum((latlon_tst$V117 - yhat_lasso_lat)^2)/nrow(feat_tst)
sum((latlon_tst$V117 - yhat_elnet_lat)^2)/nrow(feat_tst)

sum((latlon_tst$V118 - yhat_lm_lon)^2)/nrow(feat_tst)
sum((latlon_tst$V118 - yhat_ridge_lon)^2)/nrow(feat_tst)
sum((latlon_tst$V118 - yhat_lasso_lon)^2)/nrow(feat_tst)
sum((latlon_tst$V118 - yhat_elnet_lon)^2)/nrow(feat_tst)

## from the MSE we can determine that the latitude is fitting much better than the longtidue is.
## the simple linear model performs the worst out all regressions.  We can conclude that it is usually better
## to do some sort of regularization with regression.
## The elastic net performs best when the alpha parameter lies somehwere in the middle between 0 and 1.




