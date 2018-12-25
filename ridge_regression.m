function [ w ] = ridge_regression( x, y, lambda )
%RIDGE_REGRESSION implements ridge regression using the training data from
%the input and lambda parameter.

R = x' * x;
R_inv = R^-1;
d = size(x, 2);
w_ls = R_inv * x' * y;
w = (eye(d) + lambda * R_inv)^-1 * w_ls;

end