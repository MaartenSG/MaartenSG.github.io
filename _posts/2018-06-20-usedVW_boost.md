---
layout: post
title: "Used VW Golf analysis with eXtreme Gradient Boosting"
author: Maarten
date: 2018-06-15
categories: data_science
tags: ML xgboost
image: /img/usedVW_boost_files/figure-markdown_github/evaluate_treemodel-1.png
---

The linear regression model for predicting used VW Golf prices was a first attempt. The final model did not overfit but was only decent at predicting prices as Rsquared real and predicted values was only about 0.9. Exploring other machine learning algorithms, in particular the currently top performing boosting algorithm, should result in a better prediction performance. **Result:** Both boosting methods perform better than linear regression with boosting with trees topping linear boosting. Both fitted models show nicely random residuals and the repeated 10 fold cross validation made sure the final best model did not overfit to the test data.

### The data

For this exploration the same data and data preparation was used as for the linear regression analysis. Data for around 1400 VW Golf used or nearly new cars was downloaded from buyacar.co.uk . Data wrangling and processig was done earlier.

### Exploratory plots

Use plots explore the data and available features. Look for outliers and other peculiarities in the data. As shown earlier, it seems that there is an exponential relationship between price and miles on the car. And price is clearly affected by the model trim.

``` r
ggplot(data=usedvw, mapping = (aes(x=log(miles), y=price))) + geom_point(aes(color=engine)) + facet_wrap(~trim, nrow=2) + geom_smooth(method="lm")
```

![](/img/usedVW_boost_files/figure-markdown_github/explore2-1.png)

From this plot it clear that used car price shows a close to linear relationship with log(miles). The price level and slope depends on the trim. The MATCH trim shows some outliers but these are all engine dependent.

### Feature engineering

It seems a very good idea to introduce the log(miles) as a feature to the data.

``` r
usedvw <- usedvw %>% mutate(logmiles = log(miles))
```

### Model data with Gradient Boosting using the XGBoost algorithm

Boosting is a sequential process which converts weak learners (slighly better than random) into strong learners by continuously improving on previous predictions.

Advantages of XGBoost

-   regularisation can be applied
-   built in parallel processing
-   built in cross validation option
-   internal handling of missing values

XGBoost can be used to solve both regression and classification problems. The linear regression boosting method uses residuals to build subsequent models.

There are two booster algorithms that can be applied to both regression and classification data. These are gbLinear or gbTree. Which booster to apply to which problem is empirical, try both.

For ease of use and consistency the caret wrapper functions are used to apply XGBoost to the usedVW data. Tuning the parameters for XGBoost should be applied to avoid overfitting. In the caret wrapping functions these are for

xgbLinear:

-   alpha : L1 regularisation (ratio ridge:lasso)
-   lambda : L2 regularisation (penalty parameter)
-   eta : learning rate (gradient descent slope, lower takes longer but may avoid overfit)
-   nrounds

#### XGBoost with gbLinear booster

Use the built in functionality of caret train to tune the hyperparameters for xgbLinear. To get an idea about the appoximate parameters and avoid really long processing time, first use a simple 4-fold CV over a lot of parameter options.

Follow this up with more targeted parameters and a 10-fold 3x repeated CV.

Estimate model performance with RMSE. The resulting performance (i.e. RMSE level) is what can be expected from the test set performance as well.

Finally, in order to avoid overfitting even more, select the simplest model within one SE (standard error) from the best model (best performance may lead to overfit)

``` r
library(xgboost)

set.seed(1234)
split_train <- createDataPartition(y=usedvw$price, p=0.8, list=F)
training <- select(usedvw, everything())[split_train,]
test <- select(usedvw, everything())[-split_train,]

control <- trainControl( method="cv", number=4)
parameterGrid <-  expand.grid(nrounds = c(30,50,75,100),    
                              eta = 0.3,
                              alpha = seq(0,1,0.25),
                              lambda = seq(0,10,0.5) ) 
model1 <- train(price ~ . , data = training , 
               method = "xgbLinear",
               trControl = control,
               tuneGrid = parameterGrid,
               preProc = c("center", "scale","nzv")
               )

ggplot(model1)
```

![](/img/usedVW_boost_files/figure-markdown_github/initial_model-1.png)

``` r
model1$result %>% group_by(nrounds) %>% summarize(mean(RMSE))
```

    ## # A tibble: 4 x 2
    ##   nrounds `mean(RMSE)`
    ##     <dbl>        <dbl>
    ## 1      30        1366.
    ## 2      50        1355.
    ## 3      75        1357.
    ## 4     100        1362.

Choose a balance between performance and model simplicity: nrounds=50 alpha = 0 or 1 lambda = 1 to 8 each 0.5

``` r
# apply a repeated k-fold cross validation
control <- trainControl( method="repeatedcv", number=10, repeats=3)
parameterGrid <-  expand.grid(nrounds = 50,    #tuning from 5 to 100 50 is optimal
                              eta = 0.3,
                              alpha = c(0,1),
                              lambda = seq(1,8,0.5) )


#avoid data leakage and apply center and scale variables in the train process
#remove variables with zero or near zero variance in train process

model <- train(price ~ . , data = training , 
               method = "xgbLinear",
               trControl = control,
               tuneGrid = parameterGrid,
               preProc = c("center", "scale","nzv")
               )
bm <- best(model$result, metric="RMSE", maximize=F)
ggplot(model)
```

![](/img/usedVW_boost_files/figure-markdown_github/full_model-1.png)

From the hyperparameter tuning exploration, the best model parameters and resulting performance on cross validation were:

|     |  nrounds|  eta|  alpha|  lambda|      RMSE|   Rsquared|       MAE|    RMSESD|  RsquaredSD|     MAESD|
|-----|--------:|----:|------:|-------:|---------:|----------:|---------:|---------:|-----------:|---------:|
| 5   |       50|  0.3|      0|       3|  1284.858|  0.9317174|  879.7201|  196.2732|   0.0207052|  121.1474|

Now setting this up to a final model. Parameter fine tuning by hand can be done here if needed. If the best model from the tuning is selected it can be used directly from model.

``` r
finalmodel <- train(price ~ . , data = training , 
               method = "xgbLinear",
               trControl = control,
               tuneGrid = data.frame(nrounds=50, alpha=1, lambda=3, eta=0.3),
               preProc = c("center", "scale","nzv")
               )

kable(finalmodel$result)
```

|  nrounds|  alpha|  lambda|  eta|      RMSE|   Rsquared|       MAE|    RMSESD|  RsquaredSD|     MAESD|
|--------:|------:|-------:|----:|---------:|----------:|---------:|---------:|-----------:|---------:|
|       50|      1|       3|  0.3|  1274.051|  0.9334383|  875.2303|  146.2508|    0.014539|  89.11148|

``` r
# here we look at estimated feature importance
importance <- varImp(finalmodel, scale=FALSE)
plot(importance)
```

![](/img/usedVW_boost_files/figure-markdown_github/unnamed-chunk-1-1.png)

``` r
# best model is selected as having lowest RMSE (default, could be changed)
# looking at residuals and predicted vs actual values is possible
library(gridExtra)
grid.arrange(qplot(x=training$price, y=resid(finalmodel)),
             qplot(x=training$price, y=predict(finalmodel, training)), ncol=2)
```

![](/img/usedVW_boost_files/figure-markdown_github/unnamed-chunk-1-2.png)

The model fit seems to be quite good. Residuals show a random distribution without real outlier groups.

### Prediction

Given the boost model, predict the used VW car price in the test set and calculate prediction performance.

``` r
# predict test values and calculate prediction performance

# MAPE Calculation
mape <- c(mean(abs((predict(finalmodel, newdata = training) - training$price))/training$price),
          "",
          mean(abs((predict(finalmodel, newdata = test) - test$price))/test$price))

tab <- rbind(training = postResample(pred=predict(finalmodel, newdata = training), obs=training$price),
             cv=finalmodel$result[best(finalmodel$result, metric="RMSE", maximize=F),5:7] ,
             test=postResample(pred=predict(finalmodel, newdata = test), obs=test$price) )
tab <- cbind(tab,mape = as.numeric(mape))

grid.arrange(qplot(x=training$price, y=predict(finalmodel, training), color=training$trim),
             qplot(x=test$price, y=predict(finalmodel, test), color=test$trim),ncol=2)
```

![](/img/usedVW_boost_files/figure-markdown_github/predict-1.png)

|          |       RMSE|   Rsquared|       MAE|       mape|
|----------|----------:|----------:|---------:|----------:|
| training |   580.1767|  0.9860455|  399.8404|  0.0234546|
| cv       |  1274.0507|  0.9334383|  875.2303|         NA|
| test     |  1311.4030|  0.9382460|  917.3513|  0.0505426|

This model shows a good performance with Rsquared around 0.93 and good reproducability between the cross validation error metrics and the test metrics.

### XGBoost with gbTree booster

Use the built in functionality of caret train to tune the hyperparameters for xgbTree.

Same appoach as gbLinear

-   get appoximate parameters with a simple 4-fold CV over a lot of parameter options.
-   follow this up with more targeted parameters and a 10-fold 3x repeated CV.

Estimate model performance with RMSE. The resulting performance (i.e. RMSE level) is what can be expected from the test set performance as well.

``` r
library(xgboost)

control <- trainControl( method="cv", number=4)
parameterGrid <-  expand.grid(nrounds = seq(50,400,50),   
                              max_depth = seq(3,5,1),
                              eta = seq(0.2,0.5,0.1),                     #optimized
                              gamma = 0,                     #default
                              colsample_bytree = 1,
                              min_child_weight = 1,
                              subsample  = 0.9 )

modeltree1 <- train(price ~ . , data = training , 
               method = "xgbTree",
               trControl = control,
               tuneGrid = parameterGrid,
               preProc = c("center", "scale","nzv")
               )

#ggplot(modeltree1)
modeltree1$result %>% group_by(nrounds) %>% summarize(mean(RMSE))
```

    ## # A tibble: 8 x 2
    ##   nrounds `mean(RMSE)`
    ##     <dbl>        <dbl>
    ## 1      50        1342.
    ## 2     100        1333.
    ## 3     150        1336.
    ## 4     200        1342.
    ## 5     250        1345.
    ## 6     300        1349.
    ## 7     350        1352.
    ## 8     400        1353.

``` r
modeltree1$result %>% group_by(max_depth) %>% summarize(mean(RMSE))
```

    ## # A tibble: 3 x 2
    ##   max_depth `mean(RMSE)`
    ##       <dbl>        <dbl>
    ## 1         3        1319.
    ## 2         4        1349.
    ## 3         5        1364.

After a few tries with varying different parameters it shows that:

-   gamma is not having an effect on fitting -&gt; keep at 0
-   nrounds and max\_depth are somewhat inverse related -&gt; more rounds, lower depth needed
-   max\_depth 3
-   nrounds 100
-   subsample ratio of the training instance (?) -&gt; best at 0.9
-   colsample\_bytree clearly best at 1
-   eta shrinkage parameter -&gt; seems best around 0.2
-   min\_child\_weight seems not to improve fit RMSE ta cross validation -&gt; leave at 1

``` r
# apply a repeated k-fold cross validation
control <- trainControl( method="repeatedcv", number=10, repeats=3)
parameterGrid <-  expand.grid(nrounds = seq(50,400,50),   
                              max_depth = seq(3,5,1),
                              eta = seq(0.2,0.5,0.1),                     #optimized
                              gamma = 0,                     #default
                              colsample_bytree = 1,
                              min_child_weight = 1,
                              subsample  = 0.9 )

modeltree <- train(price ~ . , data = training , 
               method = "xgbTree",
               trControl = control,
               tuneGrid = parameterGrid,
               preProc = c("center", "scale","nzv")
               )


bm <- best(modeltree$result, metric="RMSE", maximize=F)
ggplot(modeltree)
```

![](/img/usedVW_boost_files/figure-markdown_github/full_treemodel-1.png)

The best model parameters and resulting performance on cross validation are:

|     |  eta|  max\_depth|  gamma|  colsample\_bytree|  min\_child\_weight|  subsample|  nrounds|      RMSE|   Rsquared|       MAE|    RMSESD|  RsquaredSD|     MAESD|
|-----|----:|-----------:|------:|------------------:|-------------------:|----------:|--------:|---------:|----------:|---------:|---------:|-----------:|---------:|
| 4   |  0.2|           3|      0|                  1|                   1|        0.9|      200|  1261.153|  0.9341865|  886.9332|  118.8475|   0.0116175|  65.97189|

``` r
finalmodel <- modeltree
# here we look at estimated feature importance
importance <- varImp(finalmodel, scale=FALSE)
plot(importance)
```

![](/img/usedVW_boost_files/figure-markdown_github/evaluate_treemodel-1.png)

``` r
# best model is selected as having lowest RMSE (default, could be changed)
# looking at residuals and predicted vs actual values is possible
grid.arrange(qplot(x=training$price, y=resid(finalmodel)),
             qplot(x=training$price, y=predict(finalmodel, training)), ncol=2)
```

![](/img/usedVW_boost_files/figure-markdown_github/evaluate_treemodel-2.png)

The model fit seems to be quite good. Just like with xgbLinear the residuals show a random distribution without real outlier groups.

### Prediction

Given the boost model, predict the used VW car price in the test set and calculate prediction performance.

``` r
# predict test values and calculate prediction performance

# MAPE Calculation
mape <- c(mean(abs((predict(finalmodel, newdata = training) - training$price))/training$price),
          "",
          mean(abs((predict(finalmodel, newdata = test) - test$price))/test$price))

tab <- rbind(training = postResample(pred=predict(finalmodel, newdata = training), obs=training$price),
             cv=finalmodel$result[best(finalmodel$result, metric="RMSE", maximize=F),8:10] ,
             test=postResample(pred=predict(finalmodel, newdata = test), obs=test$price) )
tab <- cbind(tab,mape = as.numeric(mape))

grid.arrange(qplot(x=training$price, y=predict(finalmodel, training), color=training$trim),
             qplot(x=test$price, y=predict(finalmodel, test), color=test$trim),ncol=2)
```

![](/img/usedVW_boost_files/figure-markdown_github/predict_treemodel-1.png)

|          |      RMSE|   Rsquared|       MAE|       mape|
|----------|---------:|----------:|---------:|----------:|
| training |   695.432|  0.9798714|  508.1084|  0.0300469|
| cv       |  1261.153|  0.9341865|  886.9332|         NA|
| test     |  1214.828|  0.9468717|  860.6911|  0.0480343|

Boosting with trees also shows a good performance with Rsquared around 0.93 and good reproducability between the cross validation error metrics and the test metrics.

### Final remarks

Extreme Gradient Boosting (XGB) appears to model the used VW Golf dataset quite well. Compared to Generalized Linear regression the error metrics were better in both cross validation and in test prediction.

XGBoosting with *trees* performs slightly better than *linear* boosting. A slightly higher Rsquared and a lower MAE for the test set shows improved predictive performance Rsquared of around 0.94 and lower MAE indicated good performance.

Both boosting models predict test data outcome with similar RMSE and MAE as the cross validation data showing that the model training and tuning resulted in a model that does not overfit.

All the hyper parameter tuning resulted in a balanced model for both boosting methods.

Remarkably the introduced logarithm(miles) feature was hardly used by the boosting algorithms.

For now I finish the data analysis for this used VW Golf dataset. If I run into another interesting/promising algorithm for regression I may apply it to this dataset and see what happens.

--
