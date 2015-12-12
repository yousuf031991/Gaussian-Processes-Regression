%providing an arbitrary starting place
starting_place = [0.1;0.1;0.1];
%printing the log likelihood using arbitrary values
disp('Initial log likelihood for arbitrary parameters-');
disp(training_function(starting_place));
[x_guess,fval]=fminsearch(@training_function,starting_place,[]);
%Printing the optimal values for the hyperparameters
disp('Optimal hyperparameters sigma_f,l, and sigma_n-');
disp(x_guess);
%Printing the minimized log likelihood function
disp('Minimized log-likelihood function');
disp(fval);