function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

test_vec = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
len = length(test_vec);
min_err = 999.;
for i = 1:len
    C = test_vec(i);
    for j = 1:len
        sigma = test_vec(j);
        model= svmTrain(X, y, C, @(x1, x2) gaussianKernel(x1, x2, sigma));
        predictions = svmPredict(model, Xval);
        % error(i,j) = mean(double(predictions ~= yval));
        error = mean(double(predictions ~= yval));

        if (error<min_err)
            min_err = error;
            ind_C = i;
            ind_sig = j;
        endif
    end
end

C = test_vec(ind_C);
sigma = test_vec(ind_sig);





% =========================================================================

end
