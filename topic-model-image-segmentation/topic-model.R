# Prepare Workspace
rm(list=ls())
#wd <- getwd()
#setwd(wd) 
setwd("/path/to/project")

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

# Load Documents 
voc <- read.csv("vocab.nips.txt",header=FALSE)
voc_words <- levels(voc$V1)
doc <- read.table("docword.nips.txt",skip=3)
docID <- doc$V1 # 1500 documents
wordID <- doc$V2 # 12418 words, same in vocab
wordCNT <- doc$V3 # each document uses a subset of the vocab

# Topic Model Clustering
# Preprocessing, Initial Estimates
# create document, word count matrix
doc_CNT <- matrix(0, nrow = max(docID), ncol = length(t(voc))) # histogram of word counts for each document, sparse
for (obs in 1:length(docID)){
    doc_CNT[docID[obs],wordID[obs]] = wordCNT[obs]
}

no_CNT <- which(colSums(doc_CNT) == 0)
doc_CNT <- doc_CNT[,-no_CNT] # remove words which don't occur in any document
num_words <- dim(doc_CNT)
num_words <- num_words[2]

num_topics <- 30
topics <- kmeans(doc_CNT, num_topics, iter.max=1, algorithm = 'Lloyd') # want the most primitive algorithm version, purposefully bad parameters
tpc_centers <- topics$centers
tpc_centers_smoothed <- tpc_centers + matrix(1e-6, nrow = num_topics, ncol = num_words) # additive smoothing
tpc_pmf <- tpc_centers_smoothed/rowSums(tpc_centers_smoothed) # initial estimate of pmf of word occurence for each topic

priors <- rep(1/num_topics,num_topics) # must sum to 1, higher for clusters with more data points
weights <- matrix(1/num_topics,max(docID),num_topics) # initial weights 

iter_CNT <- 0 # keep track of the number of iterations
epsilon <- 1e3 # set threshold for termination condition on loop
Q <- sum((doc_CNT%*%t(log(tpc_pmf)) + do.call("rbind", replicate(max(docID), log(priors), simplify = FALSE))*weights))
Q_last <- 0

# EM, Topic Model
while (abs(Q - Q_last) > epsilon) {
  # form expectation, "E step"
  weights_num <- doc_CNT%*%t(log(tpc_pmf)) + do.call("rbind", replicate(max(docID), priors, simplify = FALSE))
  weights_max = apply(weights_num, 2, min)
  weights_num <- weights_num - do.call("rbind", replicate(max(docID), weights_max, simplify = FALSE))
  weights_den <- apply(weights_num,1,logSumExp)
  weights_den <- t(do.call("rbind", replicate(num_topics, weights_den, simplify = FALSE)))
  weights_num_den <-weights_num - weights_den
  weights <- exp(weights_num_den)
   
  # Maximize expectation by computing updates of topic parameters, "M step"
  tpc_pmf_num <- (t(weights)%*%doc_CNT) + matrix(1e-6, nrow = num_topics, ncol = num_words)
  tpc_pmf_norm <- rowSums(tpc_pmf_num)
  tpc_pmf <- tpc_pmf_num/tpc_pmf_norm
  priors <- colSums(weights)/max(docID) + rep(1e-3,num_topics)
  priors_norm <- sum(priors)
  priors <- priors/priors_norm
  
  Q_last <- Q
  Q <- sum(doc_CNT%*%t(log(tpc_pmf)) + do.call("rbind", replicate(max(docID), log(priors), simplify = FALSE))*weights) # element wise multiplication by weights
  
  # keep track of number of iterations
  iter_CNT = iter_CNT + 1
  
}

# create a table of the top words in each document
indx_max <- matrix(0,nrow=num_topics,ncol=10)
top_words <- matrix(0,nrow=num_topics,ncol=10)
for (tpc in 1:num_topics){
  for (elem in 1:10){
    indx = (which(tpc_pmf[tpc,]==max(tpc_pmf[tpc,])))
    indx_max[tpc,elem] = indx[1]
    tpc_pmf[tpc,] = replace(tpc_pmf[tpc,],indx[1],0)
    top_words[tpc,elem] = voc_words[indx[1]]
  }
}

plot(priors, xlab = 'topics', ylab = 'probability of choosing jth topic')




