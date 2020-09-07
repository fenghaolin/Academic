function [obj,x,grad,flag,iter]=adadelta(f,grad_f,step_h,x0,rho,epsilon,tol_x, tol_f, tol_grad, max_iter)
%Haolin Feng's implementation of ADADELTA by Matthew Zeiler (2012)
%First written on Sep 5, 2020; Last updated on Sep 6, 2020
%f is the function handle of the obj function that takes only 1 arg: x
%grad_f is the function handle of the gradient of f. It takes 2 input args:
%    1. x is the point to evaluate the gradient of f
%    2. fx is the function value of f evaluated at x
%   if grad_f=[], then we use a forward finite difference to approx it,
%   where step_h is the step size of the forward finite difference
%x0 is a 1-by-n vector
%rho (between 0 and 1) is a weight parameter for the update of the 'running average' 
% of gradient^2. i.e. E[grad_sq]_t=rho E[grad_sq]_{t-1}+(1-rho) grad_sq(at epoch t)
% It is also used for the update of the 'running average' of (delta_x)^2,
% where delta_x is the step moved at each epoch(iteration)
%epsilon >0 is the smoothing factor to better condition the denominator.
%tol_x and tol_f are L2-square accuracy of change in x and obj, respectively,
%  between successive iterations. while tol_grad is the L2-square requirement for gradient
%If both the L2 norm of delta_x and gradient are small enough, algorithm will stop, and flag=0
%If the change of obj is small enough (in L2 norm), the algorighm will stop, and flag=1.
%Otherwise, will run until max_iter updates, and flag=2. (totally max_iter+1 points)
%Outputs:
% obj is a vector of the length  max_iter+1, obj value at each iteration
% x   is an (max_iter+1)-by-n matrix, each row is the point of the iteration
% grad is the estimated gradient of f when the algorithm is stopped
% iter is how many iterations have run before it stops. Note that, totally
% 1+iter points are meaningful since initial point x0 is used before any
% iteration.
[row,col]=size(x0);
if(row>1)
    disp('Warning! we take x0 as 1 by n vector')
    x0=x0';
    col=row;
end
n=col;
if(isempty(grad_f))
    if(isempty(step_h))
        step_h=1e-3; %default size is 1e-3
    end
    grad_f=@(x,fx) forward_FD(f,fx,step_h,n,x);    
end
%set default max_T, tol_x, tol_f, rho, epsilon
if(~exist('max_T','var') || isempty(max_iter))
    max_iter=100;
end
if(~exist('tol_x','var') || isempty(tol_x))
    tol_x=1e-6;
end
if(~exist('tol_f','var')|| isempty(tol_f))
    %tol_f=1e-8;
    tol_f=0;
end
if(~exist('tol_grad','var')|| isempty(tol_grad))
    tol_grad=1e-8;
end
if(~exist('rho','var')|| isempty(rho))
    rho=0.95;
end
if(~exist('epsilon','var')|| isempty(epsilon))
    epsilon=1e-2;
end
x=zeros(max_iter+1,n);
obj=zeros(max_iter+1,1);
x(1,:)=x0;
obj(1)=f(x(1,:)); 
%%%%Algorithm 1 of ADADELTA by Matthew Zeiler (2012)%%%%%%%%
Egrad_sq=0; Edx_sq=0; %line 1
for iter=1:max_iter %line 2
    grad=grad_f(x(iter,:),obj(iter)); grad_sq=dot(grad,grad);%line 3    
    Egrad_sq=rho*Egrad_sq+(1-rho)*grad_sq; %line 4
    delta_x=-sqrt(Edx_sq+epsilon)/sqrt(Egrad_sq+epsilon)*grad; %line 5
    Edx_sq=rho*Edx_sq+(1-rho)*dot(delta_x,delta_x); %line 6
    x(iter+1,:)=x(iter,:)+delta_x; %line 7
    obj(iter+1)=f(x(iter+1,:));
    if(dot(delta_x,delta_x)<tol_x) && (grad_sq<tol_grad)
        flag=0; %change of x small enough and gradient close to zero
        return;
    elseif(dot(obj(iter+1)-obj(iter),obj(iter+1)-obj(iter))<tol_f)
        flag=1; %change of f small enough
        return;
    end
end
flag=2; %terminate after max_iter of updates are carried out.
end

function [grad]=forward_FD(f,fx,step_h,n,x)
%forward finite difference to approximately compute the gradient of f at x
grad=zeros(size(x));
for i=1:n
    tmp_x=x;
    tmp_x(i)=tmp_x(i)+step_h;
    grad(i)=(f(tmp_x)-fx)/step_h;
end
return;
end