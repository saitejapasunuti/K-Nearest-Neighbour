################ K NEAREST NEIGHBOR #############################

install.packages("caTools")#for trainn and test dataset
install.packages("dplyr")#for data manipulation
install.packages("ggplot2")#for data visualization
install.packages("class")#KNN
install.packages("caret")#confusion matrix
install.packages("corrplot")#correlation plot

library("caTools")
library("dplyr")
library("ggplot2")
library("class")
library("caret")
library("corrplot")

glass <- read.csv("D:/360digiTMG/unsupervised/mod18 K Nearest Neighbour/glass dataset/glass.csv")
View(glass)

#Standardize the Data
#we are using scale() to standardize the feature columns of glass Exclude the target column Type
#Exclude the target column Type

stan_features <- scale(glass[,1:9])

#join the standardized data with the target column
data <- cbind(stan_features,glass[10])

#check  if there are na values 
anyNA(data)
# FALSE

head(data)

#data visualization
#below plot explains the realation between diiferent features
corrplot(cor(data))

#test and train data split
#we use caTools to split the data into test and train data with splitratio of 0.70
set.seed(101)
sample <- sample.split(data$Type,SplitRatio = 0.70)

train <- subset(data,sample==TRUE)
test <- subset(data,sample==FALSE)

#KNN MODEL
#use knn() to predict the target variable using the k value=1
predicted.type <- knn(train[1:9],test[1:9],train$Type,k=1)
#check errror in prediction
error <- mean(predicted.type!=test$Type)
error
#0.2923077

#confusion matrix
confusionMatrix(table(predicted.type ,test$Type))
#Confusion Matrix and Statistics


#Overall Statistics

#Accuracy : 0.7077         
#95% CI : (0.5817, 0.814)
#No Information Rate : 0.3538         
#P-Value [Acc > NIR] : 6.899e-09      

#Kappa : 0.6059         

#Mcnemar's Test P-Value : NA  

#The above results reveal that our model achieved an accuracy of 70.77 %. Lets try different values of k and assess our model.



predicted.type <- NULL
error.rate <- NULL

for(i in 1:10){
  predicted.type <- knn(train[1:9],test[1:9],train$Type,k=i)
  error.rate[i] <- mean(predicted.type!=test$Type)
}

knn.error <- as.data.frame(cbind(k=1:10,error.type=error.rate))

#choosing k value by visualization
#lets plot error.type vs k using ggplot
ggplot(knn.error,aes(k,error.type))+
  geom_point()+
  geom_line()+
  scale_x_continuous(breaks = 1:10)+
  theme_bw()+
  xlab("value of k")+
  ylab("error")
#the above plot reveals that the error is lowest when k=1,2 and k=5 and then jump back high revealing that the k=2 & 5 is the  optimal value 
#now build our model with k=2

predicted.type <- knn(train[1:9],test[1:9],train$Type,k=2)
#Error in prediction
error <- mean(predicted.type!=test$Type)
#Confusion Matrix
confusionMatrix(table(predicted.type,test$Type))
#Accuracy : 0.7077

predicted.type <- knn(train[1:9],test[1:9],train$Type,k=5)
#Error in prediction
error <- mean(predicted.type!=test$Type)
#Confusion Matrix
confusionMatrix(table(predicted.type,test$Type))
#Accuracy : 0.6462