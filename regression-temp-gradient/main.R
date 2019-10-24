# Prepare Workspace
rm(list=ls())
#wd <- getwd()
#setwd(wd)
setwd("path/to/project")

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
library('np')
library('DAAG')

loc <- read.table('Locations.txt', header = TRUE, sep = ' ')
loc_UTM <- loc[,7:8]
loc_UTM_east <- loc$East_UTM
loc_UTM_north <- loc$North_UTM
#grid <- expand.grid(loc_UTM_north,loc_UTM_east) # cartesian product of locations (112x112 = 12544)x2

oreg_met <- read.table('Oregon_Met_Data.txt', header = TRUE, sep = ' ')
min_T <- oreg_met$Tmin_deg_C
year <- oreg_met$Year

# annual means for each year (row1 = 2000, ... , row5 = 2004)
annual_min_means <- matrix(0,max(year)-min(year)+1,max(oreg_met$SID))
for (location in 1:max(oreg_met$SID))
{
  for (yr in min(year):max(year))
  {
    count_min <- 0
    num_min <- 0

    for (dailyT in 1:length(oreg_met$SID))
    {
      if (oreg_met$SID[dailyT] == location && oreg_met$Year[dailyT] == yr)
      {
        if (min_T[dailyT] != max(min_T)) # ignore tmin = 9999 degrees
        {
        count_min <- count_min + min_T[dailyT]
        num_min <- num_min + 1
        }
      }
    }
    annual_min <- count_min/num_min
    annual_min_means[yr-min(year)+1,location] <- annual_min
  }
}

# average over all 5 years
for (yr in 1:5)
{
  annual_min_yr <- annual_min_means[yr,]
  annual_min_yr[is.na(annual_min_yr)] <- mean(annual_min_yr[!is.na(annual_min_yr)])
  annual_min_means[yr,] <- annual_min_yr
}

avg_ann_min_T <- colSums(annual_min_means)/nrow(annual_min_means)

# predict using npreg
#optimal_bw <- npregbw(na.action = na.omit, xdat = as.matrix(loc_UTM), ydat = avg_ann_min_T, bwscaling = TRUE)
pred_min_T <- npreg(tydat = avg_ann_min_T, txdat = as.matrix(loc_UTM), data=as.matrix(loc_UTM), ckertype = 'gaussian')
MSE <- pred_min_T$MSE # standard regression metrics
R2 <- pred_min_T$R2
plot(pred_min_T, plot.error.methods = 'bootstrap')
#points(as.matrix(loc_UTM),avg_ann_min_T)


# CHANGE alpha
h <- c(1, 100, 1000, 10000, 30000, 50000) # scales
spaces <- as.matrix(dist(loc_UTM, method = 'euclidean', diag = FALSE, upper = FALSE))
wmat <- exp(-spaces^2/(2*h[1]^2))
for (i in 2:6)
{
  grammmat <- exp(-spaces^2/(2*h[i]^2))
  wmat <- cbind(wmat, grammmat)
}

wmod <- cv.glmnet(wmat, avg_ann_min_T , alpha=.2)
min_lamb <- wmod$lambda.min
num_coef <- wmod$nzero
coef(wmod)
#wmod<- cv.lm(data = data.frame(wmat), form.lm = formula(avg_ann_min_T ~ .), m = 10, plotit = "Observed")

xmin <- min(loc_UTM[,1])
xmax <- max(loc_UTM[,1])
ymin <- min(loc_UTM[,2])
ymax <- max(loc_UTM[,2])
xvec <- seq(xmin, xmax, length=100)
yvec <- seq(ymin, ymax, length=100)

pmat <- matrix(0, nrow=100*100, ncol=2)
ptr <- 1

for (i in 1:100)
{
  for (j in 1:100)
  {
    pmat[ptr, 1] <- xvec[i]
    pmat[ptr, 2] <- yvec[j]
    ptr <- ptr+1
  }
}

diff <- function(i,j) {sqrt(rowSums((pmat[i,] - loc_UTM[j,])^2))}
distsampletopts <- outer(seq_len(10000), seq_len(dim(loc_UTM)[1]), diff)
wmat <- exp(-distsampletopts^2/(2*h[1]^2))

for (i in 2:6)
{
  grammmat <- exp(-distsampletopts^2/(2*h[i]^2))
  wmat <- cbind(wmat, grammmat)
}

preds <- predict.cv.glmnet(wmod, wmat, s='lambda.min')

zmat <- matrix(0, nrow = 100, ncol = 100)
ptr <- 1
for (i in 1:100){
  for (j in 1:100){
    zmat[i,j] <- preds[ptr]
    ptr <- ptr + 1
  }
}

wscale <- max(abs(min(preds)), abs(max(preds)))
image(yvec, xvec, (t(zmat)+wscale)/(2*wscale), col = grey(seq(0, 1, length=256)), useRaster=TRUE)
#wmod <- unlist(wmod)
plot(wmod)


# dnorm, pnorm, qnorm, rnorm, default is already standardized (for Gaussian Kernel), dnorm gives density, pnorm gives distribution
