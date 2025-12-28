setwd("/Users/hartej/R/Math478")
data <- read.csv("data/Data_train.csv")

set.seed(123)
indices <- 1:dim(data)[1]
training_indices <- sample(indices, 0.8*dim(data)[1], replace = F)
training_data <- data[training_indices, ]
testing_data <- data[-training_indices, ]

x <- model.matrix(Y ~ ., data)
y <- data$Y
training_matrix <- x[training_indices, ]
testing_matrix <- x[-training_indices, ]

# Linear regression model
lm.mod <- lm(Y ~ ., data = training_data)
summary(lm.mod)
lm.train.preds <- predict(lm.mod, training_data)
lm.train.mse <- mean((training_data$Y - lm.train.preds)^2)
lm.test.preds <- predict(lm.mod, testing_data)
lm.test.mse <- mean((testing_data$Y - lm.test.preds)^2)

# Subset selection
library(leaps)
regfit.full <- regsubsets(Y ~ ., data, nvmax = 15)
reg.summary <- summary(regfit.full)
reg.summary
reg.summary$rss

par(mfrow = c(2, 2))
plot(reg.summary$rss, xlab = "Number of Variables",
     ylab = "RSS", type = "l")
points(which.min(reg.summary$rss), reg.summary$rss[which.min(reg.summary$rss)], col = "red", cex = 2, 
       pch = 20)

plot(reg.summary$adjr2, xlab = "Number of Variables",
     ylab = "Adjusted RSq", type = "l")
points(which.max(reg.summary$adjr2), reg.summary$adjr2[which.max(reg.summary$adjr2)], col = "red", cex = 2, 
       pch = 20)

plot(reg.summary$cp, xlab = "Number of Variables",
     ylab = "Cp", type = "l")
points(which.min(reg.summary$cp), reg.summary$cp[which.min(reg.summary$cp)], col = "red", cex = 2,
       pch = 20)

plot(reg.summary$bic, xlab = "Number of Variables",
     ylab = "BIC", type = "l")
points(which.min(reg.summary$bic), reg.summary$bic[which.min(reg.summary$bic)], col = "red", cex = 2,
       pch = 20)

# Dimensionality reduction - Principal Component Regression (PCR)
library(pls)
par(mfrow = c(1, 1))
set.seed(1)
pcr.fit <- pcr(Y ~ ., data = training_data, scale = TRUE, validation = "CV")
summary(pcr.fit)
validationplot(pcr.fit, val.type = "MSEP")
pcr.pred <- predict(pcr.fit, testing_data, ncomp = 15)
pcr.test.mse <- mean((pcr.pred - y[-training_indices])^2)

# Reduced the number of predictors to a subset of them
data <- data[, c(1, 3, 4, 5, 7, 13, 14)]
training_data <- training_data[, c(1, 3, 4, 5, 7, 13, 14)]
testing_data <- testing_data[, c(1, 3, 4, 5, 7, 13, 14)]
x <- x[, c(1, 3, 4, 5, 7, 13, 14)]
training_matrix <- training_matrix[, c(1, 3, 4, 5, 7, 13, 14)]
testing_matrix <- testing_matrix[, c(1, 3, 4, 5, 7, 13, 14)]

# Linear regression with subset selection
lm.subset.mod <- lm(Y ~ ., data = training_data)
summary(lm.subset.mod)
lm.subset.train.preds <- predict(lm.subset.mod, training_data)
lm.subset.train.mse <- mean((training_data$Y - lm.subset.train.preds)^2)
lm.subset.test.preds <- predict(lm.subset.mod, testing_data)
lm.subset.test.mse <- mean((testing_data$Y - lm.subset.test.preds)^2)

library(DAAG)
cv.out <- DAAG::cv.lm(Y ~ ., data = data, m = 75)
summary(cv.out)
lm.subset.cv.mse <- mean((cv.out$cvpred - data$Y)^2)

# Ridge regression (both without Cross-Validation and with Cross-Validation)
library(glmnet)
grid <- 10^seq(10, -5, length = 150)
ridge.mod <- glmnet(training_matrix, y[training_indices], alpha = 0, lambda = grid)
summary(ridge.mod)
ridge.errors <- rep(0, length(grid))
for (i in 1:length(grid)) {
  ridge.pred <- predict(ridge.mod, s = grid[i], newx = testing_matrix)
  ridge.errors[i] <- mean((ridge.pred - y[-training_indices])^2)
}
ridge.errors
bestlam <- grid[which.min(ridge.errors)]
ridge.pred <- predict(ridge.mod, s = bestlam, newx = testing_matrix)
ridge.error <- mean((ridge.pred - y[-training_indices])^2)

set.seed(1)
cv.out <- cv.glmnet(x, y, alpha = 0, nfolds = 4)
summary(cv.out)
plot(cv.out)
bestlam <- cv.out$lambda.min
index_lambda_min <- which(cv.out$lambda == bestlam)
ridge.cv.mse <- cv.out$cvm[index_lambda_min]

ridge.min.test.mse <- min(ridge.error, ridge.cv.mse)

# Lasso regression (both without Cross-Validation and with Cross-Validation)
lasso.mod <- glmnet(training_matrix, y[training_indices], alpha = 1, lambda = grid)
summary(lasso.mod)
lasso.errors <- rep(0, length(grid))
for (i in 1:length(grid)) {
  lasso.pred <- predict(lasso.mod, s = grid[i], newx = testing_matrix)
  lasso.errors[i] <- mean((lasso.pred - y[-training_indices])^2)
}
lasso.errors
bestlam <- grid[which.min(lasso.errors)]
lasso.pred <- predict(lasso.mod, s = bestlam, newx = testing_matrix)
lasso.error <- mean((lasso.pred - y[-training_indices])^2)

set.seed(1)
cv.out <- cv.glmnet(x, y, alpha = 1, nfolds = 3)
plot(cv.out)
bestlam <- cv.out$lambda.min
index_lambda_min <- which(cv.out$lambda == bestlam)
lasso.cv.mse <- cv.out$cvm[index_lambda_min]

lasso.min.test.mse <- min(lasso.error, lasso.cv.mse)

# Polynomial regression

# poly(X1, 2, raw = T) + poly(X2, 2, raw = T) + poly(X3, 2, raw = T)
# + poly(X4, 2, raw = T) + poly(X5, 2, raw = T) + poly(X6, 2, raw = T)
# + poly(X7, 2, raw = T) + poly(X8, 2, raw = T) + poly(X9, 2, raw = T)
# + poly(X10, 2, raw = T) + poly(X11, 2, raw = T) + poly(X12, 2, raw = T)
# + poly(X13, 2, raw = T) + poly(X14, 2, raw = T) + poly(X15, 2, raw = T)
quadratic.mod <- lm(Y ~ poly(X2, 2, raw = T) + poly(X3, 2, raw = T)
                    + poly(X4, 2, raw = T) + poly(X6, 2, raw = T) 
                    + poly(X12, 2, raw = T) + poly(X13, 2, raw = T),
                    data = training_data)
summary(quadratic.mod)
quadratic.preds <- predict(quadratic.mod, testing_data)
quadratic.test.mse <- mean((testing_data$Y - quadratic.preds)^2)

cubic.mod <- lm(Y ~ poly(X2, 3, raw = T) + poly(X3, 3, raw = T)
                + poly(X4, 3, raw = T) + poly(X6, 3, raw = T) 
                + poly(X12, 3, raw = T) + poly(X13, 3, raw = T),
                data = training_data)
summary(cubic.mod)
cubic.preds <- predict(cubic.mod, testing_data)
cubic.test.mse <- mean((testing_data$Y - cubic.preds)^2)

quartic.mod <- lm(Y ~ poly(X2, 4, raw = T) + poly(X3, 4, raw = T)
                  + poly(X4, 4, raw = T) + poly(X6, 4, raw = T) 
                  + poly(X12, 4, raw = T) + poly(X13, 4, raw = T),
                  data = training_data)
summary(quartic.mod)
quartic.preds <- predict(quartic.mod, testing_data)
quartic.test.mse <- mean((testing_data$Y - quartic.preds)^2)

poly.mod <- lm(Y ~ poly(X2, 1, raw = T) + poly(X3, 3, raw = T)
               + poly(X4, 3, raw = T) + poly(X6, 1, raw = T) 
               + poly(X12, 1, raw = T) + poly(X13, 1, raw = T),
               data = training_data)
summary(poly.mod)
poly.preds <- predict(poly.mod, testing_data)
poly.test.mse <- mean((testing_data$Y - poly.preds)^2)

polynomial.min.test.mse <- min(quadratic.test.mse, cubic.test.mse, quartic.test.mse, poly.test.mse)

# Splines
# bs(X1, df = 4) + bs(X2, df = 4) + bs(X3, df = 4) 
# + bs(X4, df = 4) + bs(X5, df = 4) + bs(X6, df = 4)
# + bs(X7, df = 4) + bs(X8, df = 4) + bs(X9, df = 4)
# + bs(X10, df = 4) + bs(X11, df = 4) + bs(X12, df = 4)
# + bs(X13, df = 4) + bs(X14, df = 4) + bs(X15, df = 4)
library(splines)
spline.test.mse.vals <- rep(0, 5)
for (i in 4:8) {
  spline.fit <- lm(Y ~ bs(X2, df = i, degree = 1) + bs(X3, df = i, degree = 3) 
                   + bs(X4, df = i, degree = 3) + bs(X6, df = i, degree = 1) 
                   + bs(X12, df = i, degree = 1) + bs(X13, df = i, degree = 1),
                   data = training_data)
  summary(spline.fit)
  spline.test.preds <- predict(spline.fit, testing_data)
  spline.test.mse <- mean((testing_data$Y - spline.test.preds)^2)
  spline.test.mse.vals[i - 3] <- spline.test.mse
}
spline.test.mse.vals
spline.test.mse.min <- min(spline.test.mse.vals)

spline.cv.out <- DAAG::cv.lm(Y ~ bs(X2, df = 4, degree = 1) + bs(X3, df = 4, degree = 3) 
                             + bs(X4, df = 4, degree = 3) + bs(X6, df = 4, degree = 1) 
                             + bs(X12, df = 4, degree = 1) + bs(X13, df = 4, degree = 1),
                             data = data, m = 75)
summary(spline.cv.out)
spline.cv.mse <- mean((spline.cv.out$cvpred - data$Y)^2)

# Generalized additive models (GAMs)
gam.test.mse.vals <- rep(0, 5)
for (i in 4:8) {
  gam.fit <- lm(Y ~ ns(X2, i) + ns(X3, i) + ns(X4, i) 
                + ns(X6, i) + ns(X12, i) + ns(X13, i),
                data = training_data)
  summary(gam.fit)
  gam.test.preds <- predict(gam.fit, testing_data)
  gam.test.mse <- mean((testing_data$Y - gam.test.preds)^2)
  gam.test.mse.vals[i - 3] <- gam.test.mse
}
gam.test.mse.vals
gam.test.mse.min <- min(gam.test.mse.vals)

cv.out <- DAAG::cv.lm(Y ~ ns(X2, i) + ns(X3, i) + ns(X4, i) 
                      + ns(X6, i) + ns(X12, i) + ns(X13, i),
                      data = data, m = 75)
summary(cv.out)
gam.cv.mse <- mean((cv.out$cvpred - data$Y)^2)

# Bagging, decision trees/random forest, boosting
library(randomForest)
set.seed(1)
bag.data <- randomForest(Y ~ ., data = training_data, mtry = 6, importance = TRUE)
yhat.bag <- predict(bag.data, newdata = testing_data)
plot(yhat.bag, testing_data$Y)
abline(0, 1)
bag.error <- mean((yhat.bag - testing_data$Y)^2)

set.seed(1)
m_values <- 1:6
rf.test.errors <- rep(0, 6)
for (m in m_values) {
  rf.data <- randomForest(Y ~ ., data = training_data, mtry = m, importance = TRUE)
  yhat.rf <- predict(rf.data, newdata = testing_data)
  rf.test.errors[m] <- mean((yhat.rf - testing_data$Y)^2)
}
rf.test.errors
rf.error.min <- min(rf.test.errors)

library(gbm)
set.seed(1)
lambdas <- 10^seq(-3, 0, by = 0.1)
boost.test.errors <- rep(0, length(lambdas))
for (i in 1:length(lambdas)) {
  boost.data <- gbm(Y ~ ., data = training_data,
                    distribution = "gaussian", n.trees = 1000,
                    shrinkage = lambdas[i])
  yhat.boost <- predict(boost.data, newdata = testing_data, n.trees = 1000)
  boost.test.errors[i] <- mean((yhat.boost - testing_data$Y)^2)
}
par(mfrow = c(1, 1))
plot(lambdas, boost.test.errors, xlab = "Shrinkage Values", ylab = "Test MSE")
boost.error.min <- min(boost.test.errors)

library(caret)
cv.out <- train(
  Y ~ .,
  data = data,
  method = "rf"
)
plot(cv.out)
rf.cv.mse <- min(cv.out$results$RMSE)^2

# Support vector machines
library(e1071)

set.seed(1)
cost_range <- 10^seq(-2, 1, 0.1)
tune.out <- tune(svm, Y ~ ., data = training_data, kernel = "linear", 
                 ranges = list(cost = cost_range))
summary(tune.out)

svm.fit.linear <- svm(Y ~ ., data = training_data, kernel = "linear", 
                      cost = 0.05011872, scale = TRUE)
summary(svm.fit.linear)

svm.linear.train.preds <- predict(svm.fit.linear, training_data)
svm.linear.train.mse <- mean((training_data$Y - svm.linear.train.preds)^2)

svm.linear.test.preds <- predict(svm.fit.linear, testing_data)
svm.linear.test.mse <- mean((testing_data$Y - svm.linear.test.preds)^2)

set.seed(1)
tune.out <- tune(svm, Y ~ ., data = training_data, kernel = "radial", 
                 ranges = list(cost = cost_range))
summary(tune.out)

svm.fit.radial <- svm(Y ~ ., data = training_data, kernel = "radial", 
                      cost = 0.1995262, scale = TRUE)
summary(svm.fit.radial)

svm.radial.train.preds <- predict(svm.fit.radial, training_data)
svm.radial.train.mse <- mean((training_data$Y - svm.radial.train.preds)^2)

svm.radial.test.preds <- predict(svm.fit.radial, testing_data)
svm.radial.test.mse <- mean((testing_data$Y - svm.radial.test.preds)^2)

set.seed(1)
tune.out <- tune(svm, Y ~ ., data = training_data, kernel = "polynomial", 
                 degree = 2, ranges = list(cost = cost_range))
summary(tune.out)

svm.fit.polynomial <- svm(Y ~ ., data = training_data, kernel = "polynomial", 
                          degree = 2, cost = 10, scale = TRUE)
summary(svm.fit.polynomial)

svm.polynomial.train.preds <- predict(svm.fit.polynomial, training_data)
svm.polynomial.train.mse <- mean((training_data$Y - svm.polynomial.train.preds)^2)

svm.polynomial.test.preds <- predict(svm.fit.polynomial, testing_data)
svm.polynomial.test.mse <- mean((testing_data$Y - svm.polynomial.test.preds)^2)

# KNN regression
library(FNN)

knn.mod <- knn.reg(training_data[, -1], testing_data[, -1], training_data$Y, k = 7)
summary(knn.mod)
knn.preds <- knn.mod$pred
knn.test.mse <- mean((testing_data$Y - knn.preds)^2)

# Submission file
data_testX <- read.csv("data/Data_testX.csv")
best.mod <- lm(Y ~ bs(X2, df = 4, degree = 1) + bs(X3, df = 4, degree = 3) 
               + bs(X4, df = 4, degree = 3) + bs(X6, df = 4, degree = 1) 
               + bs(X12, df = 4, degree = 1) + bs(X13, df = 4, degree = 1),
               data = data)
summary(best.mod)
data_testX <- data_testX[, c(2, 3, 4, 6, 12, 13)]
predictions <- predict(best.mod, data_testX)
write.table(predictions, file = "data/Sahni_Hartej_Predictions.csv", row.names = FALSE, 
            col.names = FALSE, sep = ","
)