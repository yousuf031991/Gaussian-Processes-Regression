# Gaussian Processes Regression
An implementation of a regression model using Gaussian Processes.

This is based on a university project on statistical learning. The objectives are under Requirements.pdf

Basically, gradient descent libraries from Matlab are used to train Gaussian regression hyperparameters. Training, validation, and test data (under Gaussian_process_regression_data.mat file) were given to train and test the model. The final output of the optimized model when run on the test data looks something like this:
![Output image](https://github.com/yousuf031991/Gaussian-Processes-Regression/blob/master/output.jpg)

File descriptions-
  - task1.m - Accomplishes Task 1 from the Requirements document using arbitrary hyperparameters
  - task2.m - Accomplishes Task 2 with trained hyperparameters
  - training_function.m - Trains hyperparameters using Matlab function fminunc() and fminsearch(). Called by files with coressponding names.
  - task2actual_data.mat - Target test data
  - task2output.mat - Predicted target test data

