# Prepare Workspace
rm(list=ls())
#wd <- getwd()
setwd(wd) 
#setwd("~/path/to/project")

# Relevant Classification Packages
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


##########
# Part (a)
irisdat <- read.csv("iris.data.txt", header=FALSE)
numiris=irisdat[,c(1, 2, 3, 4)] 
postscript("irisscatterplot.eps")
# so that I get a postscript file
speciesnames <- c("setosa","versicolor","virginica") 
pchr <- c(1, 2, 3)
colr <- c("red","green","blue","yellow","orange") 
ss <- expand.grid(species = 1:3)
parset <- with(ss, simpleTheme(pch=pchr[species], col=colr[species]))
splom(irisdat[,c(1:4)], groups=irisdat$V5, par.settings=parset,
      varnames=c( "Sepal\nLength", "Sepal\nWidth", "Petal \nLength", "Petal \nWidth"),
      key=list(text=list(speciesnames), points=list(pch=pchr), columns=3))
dev.off()

# Part (b)
irismu <-sapply(numiris[1:4], mean)
irisstd <-sapply(numiris[1:4], sd)
irisOffset <- t(t(numiris) - irismu)
irisScale <- t(t(irisOffset)/irisstd)
irisPCA<-princomp(irisScale)
screeplot(irisPCA, type = "lines")
with(iris, plot(irisPCA$scores, col = colr[Species], pch=pchr[Species]))

# Part (c)
irisOneHotSplit<-with(numiris, data.frame(model.matrix(~Species-1,iris)))
pls <- pls2_nipals(irisScale, irisOneHotSplit, a=2)
pls1 <- plsreg2(irisOffset, irisOneHotSplit, comps=2)
plot(pls1$x.scores, col = colr[iris$Species], pch = pchr[iris$Species])


##########
# Part (a)
windat <- read.table("wine.data.txt",header=FALSE,sep=",")
win <- var(windat,windat)
wineig <- eigen(win)
wineigvec <- wineig$vectors
wineigval <- wineig$values
x <- (1:length(wineigval))
sorted <- sort(wineigval, decreasing = TRUE)
plot(x,sorted)

# Part (b)
winPCA <- princomp(windat)

a1 <- winPCA$loadings[,1]
plot(1:length(a1), a1, main = "First Eigenvector", xlab = "Attributes")
segments(0, 0, length(a1), 0)  		
segments(1:length(a1), 0, 1:length(a1), a1)

a2 <- winPCA$loadings[,2]
plot(1:length(a2), a2, main = "Second Eigenvector", xlab = "Attributes")
segments(0, 0, length(a2), 0)			
segments(1:length(a2), 0, 1:length(a2), a2)

a3 <- winPCA$loadings[,3]
plot(1:length(a3), a3, main = "Second Eigenvector", xlab = "Attributes")
segments(0, 0, length(a3), 0)  		
segments(1:length(a3), 0, 1:length(a3), a3)

# Part (c)
plot(winPCA$scores, type = "n",)
with(windat, text(winPCA$scores, labels = windat$V1, cex = 0.7, col = colr[V1]))


##########
# Part (a)
breastdat<-read.csv('wdbc.data.txt', header=FALSE)
brst <- breastdat[,c(3,6,9,12,15,18,21,24,27,30)]
brstmu <-sapply(brst, mean)
brststd <-sapply(brst, sd)
brstOffset <- t(t(brst) - brstmu)
brstScale <- t(t(brstOffset)/brststd)
brstPCA<-princomp(brstScale)
p<-princomp(brst)
with(breastdat, plot3d(p$scores, col = colr[V2], pch=pchr[V2]))

#split1 <- createDataPartition(y=breastdat[,2], p=.6485, list=FALSE) # Split into 80% training and everything else
#split2 <- createDataPartition(y=breastdat[,2][trainsplit], p=.5, list=FALSE) # Select 50 of training examples at random for evaluation

# Part (b)
brstOneHotSplit<-with(breastdat, data.frame(model.matrix(~V2-1,breastdat)))
brstpls1 <- plsreg2(brstOffset, brstOneHotSplit, comps=3)
plot(brstpls1$x.scores, col = colr[breastdat$V2]) 



