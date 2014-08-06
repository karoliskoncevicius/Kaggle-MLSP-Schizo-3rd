# I arranged the working directory into "input" (all input files went here)
# and "output" - for submission files.

# For completeness I will past all the code in this one file. This includes:
# 1. Reading Data
# 2. Preparing Data
# 3. Building the Model.
# 4. Wrigin Submission File.

###############################################################################
################################### SOURCES ###################################
###############################################################################

# (DEFINE YOUR OWN PATH HERE)

projectTree <- "/Users/karolis/Work/Schizo/final/"

# Used libraries

library(verification)
library(DWD)

###############################################################################
################################## LOAD DATA ##################################
###############################################################################

# READ TRAIN

trainFNC <- read.csv(file.path(projectTree, "input/train_FNC.csv"), as.is=T, header=T, sep=",")
trainSBM <- read.csv(file.path(projectTree, "input/train_SBM.csv"), as.is=T, header=T, sep=",")
trainLAB <- read.csv(file.path(projectTree, "input/train_labels.csv"), as.is=T, header=T, sep=",")

# READ TEST

testFNC <- read.csv(file.path(projectTree, "input/test_FNC.csv"), as.is=T, header=T, sep=",")
testSBM <- read.csv(file.path(projectTree, "input/test_SBM.csv"), as.is=T, header=T, sep=",")

# SUBMISSION EXAMPLE

myExample <- read.csv(file.path(projectTree, "input/submission_example.csv"), as.is=T, header=T, sep=",")

###############################################################################
################################## PREP DATA ##################################
###############################################################################

# (This will construct the data frames for Train And Test subsets)

# TRAIN

myTrain <- rbind(t(trainFNC[,-1]), t(trainSBM[,-1]))
colnames(myTrain) <- trainLAB$Class

# TEST

myTest <- rbind(t(testFNC[,-1]), t(testSBM[,-1]))
colnames(myTest) <- testFNC$Id

# CLEAN UP

rm(trainFNC, trainSBM, trainLAB, testFNC, testSBM)
gc()

###############################################################################
############################## CROSS-VALIDATION ###############################
###############################################################################

# This part is optional and was used to select the values of C constraint.

# (This runs 100 itterations of 10-fold cross validation

ROCS <- list()
Cs <- c(1, 5, 10, 50, 100, 300, 500, 1000)
for(Cind in 1:length(Cs)) {
	C <- Cs[Cind]
	tmpRocs <- numeric()
	for(i in 1:100) {
    	    trainInds1 <- sample(which(colnames(myTrain)==0), 42)
        	trainInds2 <- sample(which(colnames(myTrain)==1), 36)
	        trainInds <- c(trainInds1, trainInds2)
	        theTrain <- myTrain[,trainInds]
	        theTest <- myTrain[,-trainInds]

	        myFit <- kdwd(t(myTrain), colnames(myTrain), C=C)
			testScores <- t(myFit@w[[1]]) %*% theTest
			testScores <- 1 - ((testScores - min(testScores)) / max(testScores - min(testScores)))

			tmpRocs[i] <- roc.area(as.numeric(colnames(theTest)), testScores)$A		

	        print(i)
	}
	ROCS[[Cind]] <- tmpRocs
}


###############################################################################
################################## FIT MODEL ##################################
###############################################################################

# FIT

myFit <- kdwd(t(myTrain), colnames(myTrain), C=300)

# Get scores for training data (meaningless for now).

scores <- t(myFit@w[[1]]) %*% myTrain
scores <- 1 - ((scores - min(scores)) / max(scores - min(scores)))

# Check ROC area. (meaningless, because of possible overfitting)

roc.area(as.numeric(colnames(myTrain)), scores)

###############################################################################
################################ WRITE SCORES #################################
###############################################################################

# Obtain Test Scores

testScores <- t(myFit@w[[1]]) %*% myTest
testScores <- 1 - ((testScores - min(testScores)) / max(testScores - min(testScores)))

# Write scores 

myExample$Probability <- as.numeric(testScores)

# Write to file

write.csv(myExample, file=file.path(projectTree, "output/submission.csv"), row.names=F)



