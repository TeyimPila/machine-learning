function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Call the costfunction defined in the non-regularized logistic regression
[normal_J, normal_grad] = costFunction(theta, X, y);

shift_theta = theta(2:size(theta)); % cut theta from second entry upwords
theta_reg = [0; shift_theta]; % 

J = normal_J + (lambda/(2*m)).*sum(theta_reg.^2);

grad = normal_grad' + (lambda/m)*(theta); % (lambda/m)*(theta) is the regularization parameter

grad(1) = normal_grad(1); % theta (0) should not be regularized

% OR simply:  grad = (1/m)*(X'*(h-y)+lambda*theta_reg);

% =============================================================

end
