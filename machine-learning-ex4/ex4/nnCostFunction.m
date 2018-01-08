function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

X_trans = [ones(m,1) X]
y_trans = zeros(m,num_labels)
for i=1:m
  y_trans(i,y(i))=1
end

a1 = X_trans'   %a1 is 401*5000
a2 = sigmoid(Theta1*a1)   %a2 is 25* 5000
a2 = [ones(1,m);a2]   %a2 is 26*5000
a3 = sigmoid(Theta2*a2) %a3 is 10*5000

% Another way
%a2 = sigmoid(X_trans*Theta1')
%a2 = [ones(m,1) a2]
%a3 = sigmoid(a2*Theta2')

Theta1_without_bias = Theta1(:,2:(input_layer_size + 1))
Theta2_without_bias = Theta2(:,2:(hidden_layer_size + 1))

J = sum(sum(-y_trans'.*log(a3)-(1-y_trans').*log(1-a3)))/m...
  + (sum(sum(Theta1_without_bias.^2))+sum(sum(Theta2_without_bias.^2)))*lambda/(2*m)

% compute the grad
Delta_1 = zeros(size(Theta1))  %Delta_1 is 25*401
Delta_2 = zeros(size(Theta2))  %Delta_2 is 10*26
for t=1:m
  a_1 = X_trans(t,:)'       %a_1 is 401*1
  a_2 = sigmoid(Theta1*a_1) %a_2 is 25*1
  a_2 = [1;a_2]             %a_2 is 26*1
  a_3 = sigmoid(Theta2*a_2) %a_3 is 10*1
  
  delta_3 = a_3-y_trans(t,:)' %delta_3 is 10*1
  delta_2 = Theta2'*delta_3.*sigmoidGradient(a_2) %delta_2 is 26*1 
  delta_2 = delta_2(2:end)    %delta_2 is 25*1
  
  Delta_1 = Delta_1+delta_2*a_1'  %Delta_1 is 25*401
  Delta_2 = Delta_2+delta_3*a_2'  %Delta_2 is 10*26
end

Theta1_grad = Delta_1/m     %without regularization
Theta2_grad = Delta_2/m     %without regularization

Theta1_grad(:,2:end)=Theta1_grad(:,2:end)+lambda*Theta1(:,2:end)/m
Theta2_grad(:,2:end)=Theta2_grad(:,2:end)+lambda*Theta2(:,2:end)/m

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
