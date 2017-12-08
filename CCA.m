
function [w_mn,v_mn,w_mx,v_mx,Hy,Hx,muy,mux] = CCA(X,Y,plotit)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [w,v] = CCA(X,Y,plotit)
%
% This function simultaneously computes the optimal projection of the 
% labels to be predicted, while also providing a linear regression.
%
% Input:
%
% X: train data, covariates
% Y: train data, repsonse variable
% plotit: 1=plotting on, 0 is plotting of. Default value = 1.
% 
% Output:
%
% w: projection for Y
% v: projection for X
% 
% We will monitor and predict: z = w'*H*(Y-mu) using v'*X, 
% Calling p' = w'*H and a = w'*H*mu we find: p'*Y ~ v'*X + a
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[d,N] = size(Y);

if nargin < 3
    plotit = 1;
end

%%%%%%%%%%%%% Centering and sphering X & centering

mux = mean(X,2);
X = X - repmat(mux,[1,N]);
Hx = diag(1./std(X,[],2));
X = Hx*X; 

muy = mean(Y,2);
Y = Y - repmat(muy,[1,N]);

C = Y*Y'/N;
[G,L] = eig(C);
Hy = inv(diag(sqrt(diag(L)))) * G';
Y = Hy*Y;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Main Algorithm

Cxx = X*X'/N; 

Cyy = Y*Y'/N;

Cxy = X*Y'/N;

iCxx = inv(Cxx);

Sig = Cyy - Cxy'*iCxx*Cxy;

[U,S] = eig(Sig); %Eigenvalue Dec.

s = diag(S);  

[s_min,I_min] = min(s);

w_mn = U(:,I_min);

v_mn = iCxx*Cxy*w_mn;

y_mn = w_mn'*Y;
    
x_mn = v_mn'*X;

t = 1:N;

t = t-repmat(mean(t,2),[1,N]);

if sum(t.*y_mn) < 0
    w_mn = -w_mn;
    v_mn = -v_mn;
    y_mn = -y_mn;
    x_mn = -x_mn;
end
    
de_mn = y_mn - x_mn;

[s_mx,I_mx] = max(s);

w_mx = U(:,I_mx);

v_mx = iCxx*Cxy*w_mx;

y_mx = w_mx'*Y;
    
x_mx = v_mx'*X;

if sum(t.*y_mx) < 0
    w_mx = -w_mx;
    v_mx = -v_mx;
    y_mx = -y_mx;
    x_mx = -x_mx;
end

de_mx = y_mx - x_mx;


% Plotting

if plotit == 1
    
%   figure(1);clf;
%   hist(de_mn,100);
    
%     figure(2);clf;
%     plot(x_mn,y_mn,'k.');hold on;
%     x_min = min(x_mn); x_max = max(x_mn);
%     y_min = min(y_mn); y_max = max(y_mn);
%     plot([min(x_min,y_min),max(x_max,y_max)],[min(x_min,y_min),max(x_max,y_max)],'b-','linewidth',3);
    
     figure(13);clf;
     plot(y_mn,'k-','linewidth',2);hold on
     plot(x_mn,'r--','linewidth',1)
    
%     figure(4);clf;
%     hist(de_mx,100);
%     
%     figure(5);clf;
%     plot(x_mx,y_mx,'k.');hold on;
%     x_min = min(x_mx); x_max = max(x_mx);
%     y_min = min(y_mx); y_max = max(y_mx);
%     plot([min(x_min,y_min),max(x_max,y_max)],[min(x_min,y_min),max(x_max,y_max)],'b-','linewidth',3);
%     
      figure(14);clf;
      plot(y_mx,'k-','linewidth',2);hold on
      plot(x_mx,'r--','linewidth',1)
     
     drawnow;
    
end




