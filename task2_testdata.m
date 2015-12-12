load Gaussian_process_regression_data.mat;
%picking optimized hyperparameters
l=0.5552;
sigma_f=0.0816;
sigma_n=-0.0481;

X_o=input_train';
Y_o=target_train;
%K calculation begins
K = zeros(size(X_o,2));
kernel_function_m = @(x,x2) sigma_f^2*exp((x-x2)'*(x-x2)/(-2*l^2));
error_function_m = @(x,x2) sigma_n^2*(sum(x==x2)==length(x));
k_m = @(x,x2) kernel_function_m(x,x2)+error_function_m(x,x2); 
for i=1:size(X_o,2)
    for j=1:size(X_o,2)
        K(i,j)=k_m(X_o(:,i),X_o(:,j));
    end
end
%K ends

prediction_x=input_test';
%K_ss calculation begins
K_ss=zeros(size(prediction_x,2));

for i=1:size(prediction_x,2)
    for j=i:size(prediction_x,2)
        K_ss(i,j)=k_m(prediction_x(:,i),prediction_x(:,j));
    end
end
%optimisation exploiting the diagonal symmetry of K_ss
K_ss=K_ss+triu(K_ss,1)';
%K_ss ends
%K_S calculation begins
K_s=zeros(size(prediction_x,2),size(X_o,2));

for i=1:size(prediction_x,2)
    for j=1:size(X_o,2)
        K_s(i,j)=k_m(prediction_x(:,i),X_o(:,j)); 
    end
end
%K_s ends
%calculate Mu and Sigma
%use cholesky decomposition
L=chol(K,'lower');
alpha=L'\(L\Y_o);
disp('Our predicted data(mean)');
Mu = K_s*alpha;
disp(Mu);
disp('Our predicted confidence(covariance)');
v=L\K_s';
Sigma = K_ss-v'*v;
disp(Sigma);
target_test=Mu;
save('YousufHussain_SyedMohammad.mat','target_test');
