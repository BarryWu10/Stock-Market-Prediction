function [H,alpha]=AdaBoost(X,Y,C,T,WLearner)

% https://raw.githubusercontent.com/BoChen90/machine-learning-matlab/master/AdaBoost.m



%  AdaBoost
%  Train a strong classifier using several weak ones
%
% Input
%      X - samples
%      Y - label of samples - 
%          1 - belong to the class,0 - otherwise
%      C - array of feature vectors 
%      T - number of iterations
%
%      WLearner - weak learner type
%
%     
% Output:
%      H - result array of weak classifiers
%       every hypothesis/classifier contains the following parameters: 
%       Mu=H{1}; 
%           Mu(1),Mu(2)-means of the 2 classes
%       InvSigma=H{2}     
%           InvSigma(1),InvSigma(2)- inverse of matrix of std. deviations of
%           the 2 classes
%     alpha - array of weights for every classisfier
%


% 25 April 2002
% PP,AA

VERY_LARGE=10;
DISP=1;
H={};


N=size(X,1);
W=zeros(N,1);

%number of positive examples
n=sum(Y(find(Y==1)));

%number of negative examples
m=N-n;


%initialize weights
for i=1:N
   if (Y(i))
      W(i)=1/(2*n);
   else
      W(i)=1/(2*m);
   end;
end;

beta=zeros(T,1);


Thresh=0;
alpha=zeros(1,T);

for t=1:T

   %normalize weights
   NW=sum(W);
   W=W/NW;
   %find current best hyp
   %weak classifier
   if (t==1)   
       Y_predict=ones(size(X,1),1);
    else
        p=H{t-1}{2};
        thresh=H{t-1}{1};
        C0=H{t-1}{3};
        Y_predict=((p*(X*C0(:,:)-thresh)>=0)~=Y');
    end;
    %Y_predict=ones(size(X,1),1)
   [H{t},epsilon,Rt]=WeakLearner(X,Y,C,W,WLearner,Y_predict);
   
   if (DISP)
        p=H{t}{2};
        thresh=H{t}{1};
        C0=H{t}{3};
    figure(500);subplot(1,2,1); hold on;
     plot([thresh*C0(1) thresh*C0(1)+10*p*C0(1)],[thresh*C0(2) thresh*C0(2)+10*p*C0(2)],'g','LineWidth',1);
   plot(thresh*C0(1)+10*[-C0(2) C0(2)], thresh*C0(2)+10*[C0(1) -C0(1)],'k','LineWidth',4);
   plot(thresh*C0(1),thresh*C0(2),'.k','MarkerSize',20);
   hold off
    end;
  % Rt
   epsilon
   beta(t)=epsilon/(1-epsilon);
   if (epsilon==0)
      alpha(t)=VERY_LARGE;
   else
      alpha(t)=log(1/beta(t));
   end;
   
   %update weights 
   correct_classif=find(Y==Rt);
   if (epsilon==0)
      break;
   end;
   
   pause;
   W(correct_classif)=W(correct_classif)*beta(t);   
   %for k=1:N
      %c=WeakClassify(X(k,:),H(t));
    %  if (Rt==Y(k))
        % W(k)=W(k)*beta(t);
      %end;
   %end;
  
end;