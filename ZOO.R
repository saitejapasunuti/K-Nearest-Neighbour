################################ K NEAREST NEIGHBOUR CLASSIFICATION #########################

set.seed(1)
library(class)
d = read.csv(file.choose())
d = data.frame(d)

names(d) <- c("animal", "hair", "feathers", "eggs", "milk", "airborne", "aquatic", "predator", "toothed", "backbone", "breathes", "venomous", "fins", "legs", "tail", "domestic", "size", "type")
View(d)

types <- table(d$type)
d_target <- d[,18]
d_key <- d[,1]

d_animal <- NULL

names(types) <- c("mammal","bird","reptile","fish","amphibian","insect","crustacean")
types

#mammal       bird    reptile       fish  amphibian     insect crustacean 
#41         20          5         13          4          8         10 

summary(d)

str(d)

k=sqrt(17)+1
m1 <- knn.cv(d[-1],d_target,k,prob = TRUE)

prediction <- m1

cmat <- table(d_target,prediction)
acc <- (sum(diag(cmat)) / length(d_target)) * 100
print(acc)
# accuracy=90.09901

data.frame(types)
#       Var1   Freq
#1     mammal   41
#2       bird   20
#3    reptile    5
#4       fish   13
#5  amphibian    4
#6     insect    8
#7 crustacean   10

cmat #confusion matrix
#            prediction
#d_target  1  2  3  4  5  6  7
#          1 41  0  0  0  0  0  0
#          2  0 20  0  0  0  0  0
#          3  0  1  0  3  1  0  0
#          4  0  0  0 13  0  0  0
#          5  0  0  0  0  4  0  0
#          6  0  0  0  0  0  8  0
#          7  0  0  0  2  1  2  5