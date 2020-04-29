---
title: "Week 1 Homework"
author: "Alex Haan"
output: rmarkdown::github_document

---
## Load Libraries

```{r}
library(kernlab)
library(kknn)
data <- read.table("credit_card_data-headers.txt", header=TRUE)
```

## Question 2.1

"Describe a situation or problem from your job, everyday life, current events, etc., for which a classification model would be appropriate. List some (up to 5) predictors that you might use."

### Answer

"My current role is Reporting Analyst at Expedia Group within our Media Solutions organization. The role of our organization is to run Hotel, Air, and other travel related advertising campaigns in hopes of simultaniously maximizing our revenue and the revenue of our advertising partners (i.e. hotels). To best do so, I believe we should create a classification model to categorize our website viewers (those who are targeted for our advertisements) into whether or not they will likely complete a transaction based on their demographic, historical, and targeting data. The predictors I may use are: "

* Income

* Age

* Number of transactions completed on their account

* Previous Expedia travel searches in past week/month/year

* How they got to our website (i.e organic, google search, link, etc)

## Question 2.2

"Using the support vector machine function ksvmcontained in the R package kernlab, find a good classifier for this data.Show the equation of your classifier, and how well it classifies the data points in the full data set.  (Don't worry about test/validation data yet; we'll cover that topic soon.)"

### Answer:

### determine which C is best by creating function and for loop

```{r}

test_list = c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)

svm <- function(x) 
 {
   output <- ksvm(as.matrix(data[,1:10]),as.factor(data[,11]),type='C-svc',kernel='vanilladot',C=x,scaled=TRUE) 
   pred <- predict(output,data[,1:10])
   sum(pred == data[,11]) / nrow(data)
   print(x)
   print(sum(pred == data[,11]) / nrow(data))
}

test_list = c(10, 20, 30, 40, 50, 60, 70, 80, 90, 100)

```

output for all tests still = 0.864

### Trying different list for broader range of C


```{r}
test_list2 <- c(0.1, 0.5, 1, 2, 3, 5, 10, 25, 100, 1000, 1000000)

```

[1] 100
[1] 0.8639144
 Setting default kernel parameters  
[1] 1000
[1] 0.8623853
 Setting default kernel parameters  
[1] 1e+06
[1] 0.6253823

Noticably, the accuracy of c for most values remained 0.86, but as we increased beyond C = 100, the values began to shrink. for C = 1000000, the accurace dropped to 0.62

Now that I know the value for C should be between 0.1 and 100, I'll give finding C one more shot

### Check again for smaller C


```{r}
test_list3 <- c(0.001, 0.0001, 0.0001)

for (c in test_list3) {
  svm(c)
}
```

0.0001 is too small, reducing the accuracy below 0.55

##Last attempt to find C using smaller values

```{r}
test_list4 <- c(0.001, 0.003, 0.004, 0.007, 0.009, 0.01, 0.1)

svm <- function(x) 
 {
   output <- ksvm(as.matrix(data[,1:10]),as.factor(data[,11]),type='C-svc',kernel='vanilladot',C=x,scaled=TRUE) 
   pred <- predict(output,data[,1:10])
   sum(pred == data[,11]) / nrow(data)
   print(x)
   print(sum(pred == data[,11]) / nrow(data))
   
}

for (c in test_list4) {
  svm(c)
}
```    

After all of the testing, it appears the range of 0.003 - 1 is acceptable for C. I'll stick with 0.1.

### Now to calculate the classifier


```{r}

output <- ksvm(as.matrix(data[,1:10]),as.factor(data[,11]),type='C-svc',kernel='vanilladot',C=0.1,scaled=TRUE) 

a <- colSums(output@xmatrix[[1]] * output@coef[[1]])
a

a0 <- output@b
a0
```


based on the above calculations it appears the classifier line is: 

0 = -0.08155226 -0.001160898(A1) -0.0006366002(A2) -0.001520968(A3) + 0.003202064(A4) + 1.004134(A5) -0.003377367(A6) + 0.0002428616(A7) -0.0004747021(A8) -0.00119319(A9) + 0.1064451(A10) 

Accuracy of the classifier: 87%


## Question 2.2.2

### Attempts at other kernels.

Creating another loop to try different kernels

```{r}
svm <- function(x) 
 {
   output <- ksvm(as.matrix(data[,1:10]),as.factor(data[,11]),type='C-svc',kernel=x,C=0.1,scaled=TRUE) 
   pred <- predict(output,data[,1:10])
   sum(pred == data[,11]) / nrow(data)
   print(x)
   print(sum(pred == data[,11]) / nrow(data))
   }

# Create Kernel List


test_kernel = c('polydot','anovadot','splinedot','rbfdot')

for (kern in test_kernel) {
  svm(kern)
}

```
 In this case, the predictive accuracy for the "splinedot" kernal was significantly greater with an accuracy of 94.5%


## Question 2.2.3

## Nested loop for k 1-> 50 and for all i's in the dataset

```{r}
# create empty list
output_list <- list()
# loop through k's
for (k in 1:50){
  final_num <- 0
  count <- 0
  
  #loop through dataset
  for (i in 1:654) {
    i_num <- 0
    kknn_model <- kknn(R1~., 
                       data[-i,],
                       data[i,],
                       distance = 2,
                       k = k,
                       kernel = 'optimal',
                       scale = TRUE
    )
   # print(kknn_model$fitted.values)
    if (fitted.values(kknn_model) >= 0.5) {i_num <- 1}
    else {i_num <-  0}
    test_val <- data[i,11]
    if (i_num == test_val) {final_num <- final_num + 1}
    else {final_num <- final_num + 0}

  }
  # test output of model
  output_list[k] <- final_num
  final_list <- as.vector(output_list, mode="numeric")
}  

#best k
which(final_list == max(final_list))

#accuracy 
final_list[[12]]
```

Based on the above, the optimal k = 12 or 15, both with a sum of 558 out of 654, meaning it correctly classified 85%. 

## Extra attempts using other methods

### Use different kernels using k = 24 based on previous problem

```{r}
kerns = c('rectangular','triangular','epanechnikov','biweight','triweight','cos','inv','gaussian','rank','optimal')

output_list <- list()
for (kernel in kerns) {
   final_num <- 0
   for (i in 1:nrow(data)) {
      i_num <- 0
      kknn_model <- kknn(R1~., 
                     data[-i,],
                     data[i,],
                     distance = 2,
                     k = 12,
                     kernel = kernel,
                     scale = TRUE
                  )
          # print(kknn_model$fitted.values)
    if (fitted.values(kknn_model) >= 0.5) {i_num <- 1}
    else {i_num <-  0}
    test_val <- data[i,11]
    if (i_num == test_val) {final_num <- final_num + 1}
    else {final_num <- final_num + 0}

}
output_list[kernel] <- final_num
final_list <- as.vector(output_list, mode="list")
}  
output_list
```

Unforunately, there was no change with other kernels.


### sample the dataset to split into test and train

```{r}

m <- dim(data)[1]
val <- sample(1:m, size = round(m/3), replace = FALSE, prob = rep(1/m, m)) 


data.train <- data[-val,]
data.validate <- data[val,]
dim(data.train)
dim(data.validate)
```

### train model using k=9 as a random test

```{r}
model <- kknn(R1 ~ ., data.train, data.validate, k = 9, scale=TRUE)


fit <- fitted(model)
tab <- table(data.validate$R1, fit)
accuracy <- function(x){sum(diag(x)/(sum(rowSums(x)))) * 100}
accuracy(tab)
```

The output of accuracy(tab) = 19.26%.... Not the best. Apparently randomly choosing k=9 wasn't the best

### Look for optimal number of k using correlation matrix and elbow

```{r}
library(corrplot)
corrplot(cor(data), type = "upper", method = "ellipse", tl.cex = 0.9)
```

Appears A98, A9, A10, and A11 have the most significant correlation. Now to determine # of k, which I believe should be around 4

```{r}
k.max <- 15
scaled_data <- scale(data)
wss <- sapply(1:k.max, function(k){kmeans(scaled_data, k, nstart=50,iter.max = 15 )$tot.withinss})
```

### Plot the elbow

```{r}
plot(1:k.max, wss,
      type="b", pch = 19, frame = FALSE, 
      xlab="Number of clusters K",
      ylab="Total within-clusters sum of squares")
```

this output shows that the marginal sum of squares is reduced at k = 5... I was close! 

##retrain model and test output

This now brought the accuracy up to 28.8%... This can't be right

## retrain model using k = 2 to 4

output of 4 = 28%, 4 = 38%, 3 = 42%...

##Last attemping using train.kknn instead of kknn

```{r}
test_k <- c(1,2,3,4,5,6,7,8,9)

test <- function(x) {
output <- train.kknn(R1~., data.train, kmax=x)
output
}


for (k in test_k) {
test(k) }

```

Outcome: K = 2 still shares the lowest Mean squared error out of all the test_k list, but not better than original attempt
