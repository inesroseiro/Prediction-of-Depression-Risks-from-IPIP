library(glmnet)
library(stabs)

mydatatest <- read.csv2("/Users/iroseiro/Desktop/Project_IA/AI_proj/data/tab/ncds_2008_followup.tab", check.names = FALSE) 
mydatatest[,1] <- NULL
dim(mydatatest)


setwd('/Users/iroseiro/Desktop/Project_IA/AI_proj/data/tab/')

mydatatest <- read.table(file = 'ncds_2008_followup.tab', sep = '\t', header = TRUE)



a = which( colnames(mydatatest)=="N8SCQ2B")
   final <- c(mydatatest[a])  


names <- c(
   'N8CMSEX','N8SCQ2B','N8SCQ2C','N8SCQ2D','N8SCQ2E','N8SCQ2F','N8SCQ2G','N8SCQ2H','N8SCQ2I','N8SCQ2J','N8SCQ2K','N8SCQ2L','N8SCQ2M','N8SCQ2N','N8SCQ2O','N8SCQ2P','N8SCQ2Q','N8SCQ2R','N8SCQ2S','N8SCQ2T','N8SCQ2U','N8SCQ2V','N8SCQ2W','N8SCQ2X','N8SCQ2Y' ,'N8SCQ2Z','N8SCQ2AA','N8SCQ2BB','N8SCQ2CC','N8SCQ2DD','N8SCQ2EE','N8SCQ2FF','N8SCQ2GG','N8SCQ2HH','N8SCQ2II', 'N8SCQ2JJ','N8SCQ2KK','N8SCQ2LL','N8SCQ2MM','N8SCQ2NN','N8SCQ2OO','N8SCQ2PP','N8SCQ2QQ','N8SCQ2RR','N8SCQ2SS','N8SCQ2TT','N8SCQ2UU','N8SCQ2VV','N8SCQ2WW','N8SCQ2XX','ND8MAL'
)



for (i in 1: length(names)) {
   index = which(colnames(mydatatest)==names[i])
   final <- c(final, mydatatest[index])  
}

write.csv(final,"/Users/iroseiro/Desktop/Project_IA/AI_proj/data/tab/dataIPIP.csv")
