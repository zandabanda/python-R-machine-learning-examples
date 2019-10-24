#Install the necessary packages
install.packages("lattice")
install.packages("plsdepot")
install.packages("rgl")

library('lattice')
library(plsdepot)
library(rgl)
setwd("/path/to/data") #WHERE iris.data, wine.data, AND wdbc.data ARE LOCATED

#4a
irisdat<-read.csv('iris.data', header=FALSE)

numiris=irisdat[,c(1,2,3,4)]
postscript("irisscatterplot.eps")
speciesnames<-c('setosa', 'veriscolor', 'virginica')
pchr<-c(1,2,3)
colr<-c('red', 'green', 'blue', 'yellow', 'orange')
ss<-expand.grid(species=1:3)
parset<-with(ss, simpleTheme(pch=pchr[species], col=colr[species]))
splom(irisdat[, c(1:4)], groups=irisdat$V5, par.settings=parset, varnames=c('Sepal\nLength', 'Sepal\nWidth', 'Petal\nLength', 'Petal\nWidth'), key=list(text=list(speciesnames), points=list(pch=pchr), columns=3))

dev.off()

#4b
irisMean <-sapply(iris[1:4], mean)
irisSd <-sapply(iris[1:4], sd)
irisOffset <- t(t(iris[1:4])-irisMean)
irisScale <- t(t(irisOffset)/irisSd)
pca<-princomp(irisScale)
screeplot(pca, type = "lines")
with(iris, plot(pca$scores, col = colr[Species], pch=pchr[Species], main = "Iris data plotted on two principal components", xlab = "First principal component", ylab = "Second principal component"))

#4c 3 COLUMN ONE-HOT
irisOneHotSplit<-with(iris, data.frame(model.matrix(~Species-1,iris)))
pls1 <- plsreg2(irisScale, irisOneHotSplit)
plot(pls1$x.scores, col = colr[iris$Species], pch = pchr[iris$Species], main = "Iris data plotted on two discriminative directions", xlab = "First direction", ylab = "Second direction") #THIS IS PROBABLY WRONG

#5a
winedat<-read.csv('wine.data', header=FALSE)

pca <- princomp(winedat[-1])
plot(pca$sdev^2, main = "The eigenvalues of the covariance matrix in sorted order", ylab = "eigenvalues", xlab = "principal components")
lines(pca$sdev^2)

pca$sdev^2 #notice that the first two offer the most variance

#5b
a <- pca$loadings[,1]
plot(1:length(a), a, main = "First Eigenvector", xlab = "Attributes", ylab = "")
segments(0, 0, length(a), 0)			
segments(1:length(a), 0, 1:length(a), a)

a <- pca$loadings[,2]
plot(1:length(a), a, main = "Second Eigenvector", xlab = "Attributes", ylab = "")
segments(0, 0, length(a), 0)			
segments(1:length(a), 0, 1:length(a), a)

a <- pca$loadings[,3]
plot(1:length(a), a, main = "Third Eigenvector", xlab = "Attributes", ylab = "")
segments(0, 0, length(a), 0)			
segments(1:length(a), 0, 1:length(a), a)

#5c
plot(pca$scores, type = "n", xlab = "First principal component", ylab = "Second principal component", main = "Principal Components")
with(winedat, text(pca$scores, labels = winedat$V1, cex = 0.7, col = colr[V1]))

#7a
breastdat<-read.csv('wdbc.data', header=FALSE)
breast <- breastdat[,c(3,6,9,12,15,18,21,24,27,30)]
breastMean <-sapply(breast, mean)
breastSd <-sapply(breast, sd)
breastOffset <- t(t(breast)-breastMean)
breastScale <- t(t(breastOffset)/breastSd)
p<-princomp(breastScale)#means
with(breastdat, plot3d(p$scores, col = colr[V2], pch=pchr[V2], main = "Breast Cancer Principal Components"))

#7b
breastOneHotSplit<-with(breastdat, data.frame(model.matrix(~V2-1,breastdat)))
pls1 <- plsreg2(breastScale, breastOneHotSplit)
plot3d(pls1$x.scores, col = colr[breastdat$V2], pch = pchr[breastdat$V2], main = "Breast cancer data plotted on two discriminative directions", xlab = "Comp.1", ylab = "Comp.2", zlab = "Comp.3") #THIS IS PROBABLY WRONG