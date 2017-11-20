% EM:
clear all; close all; clc;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%Expectation Maxmization Algorithm %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

N=100;

D=2;

K=2;

Pii=[.25 .75]; %K-component GMM Mixing Coefficients
change=.03;    %threshold for log-likelihood change

likeVec=0;
%% Sample from GMM

Nk=zeros(K,1); %number of members for each latent gaussian

responsibilities=zeros(N,K); % responsiblities of latent variable for data 'n'

Mu=cell(2,1);
Covi=cell(2,1);
Mu{1}=[2 2];
Covi{1}=[1 0;.0 1];
Mu{2}=[-2 -2];
Covi{2}=[1 0;0 1];

%2 GMM components (coded as 2 obj's for legibility)
Gauss1=gmdistribution(Mu{1},Covi{1});%or (mu,sigma p (mixing coeff))
Gauss2=gmdistribution(Mu{2},Covi{2});

data=zeros(N,D); %will hold samples drawn from the GMM

for i=1:N, %sample from the 2 gaussians
    if rand<Pii(1) %if unif RV falls under comp 1
        data(i,:)=random(Gauss1);   %sample from gaussian 1
    else
        data(i,:)=random(Gauss2);   %else sample gaussian 2
    end
end

Ui=Mu;

pause
%% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Estimate latent variables of GMM 
% Initialize latent variable parameters 
%  (usually done w/ K Means Algo)
% 1) EM iterates until convergence:
% E STEP
%   update data point 'responsibilities' 
%   for each Gaussian, holding params constant.   
%  
% M STEP
%   update GMM parameters (mu, Sigma and Pi)
%   for each Guassian, holding Responsibilities constant
%   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

while(1) %change to for 1:count
    %%%%%%%%%%%%%%%%%%%%
    % Expectation STEP %  (Estimate responsibilities w/ new params)
    %%%%%%%%%%%%%%%%%%%%
        
    %Broken here
    
    for i=1:N,
        for k=1:K,            
            temp1=Pii(k)*gauss(Ui{k},Covi{k},data(i,:));
            temp2=0;
            for j=1:K,
                temp2=[temp2,Pii(j)*gauss(Ui{j},Covi{j},data(i,:))];
            end
            responsibilities(i,k)=temp1/sum(temp2);
        end
    end
    
    %%%%%%%%%%%%%%%%%%%%%
    % Maximization STEP % (Estimate params w/ new responsibilities)
    %%%%%%%%%%%%%%%%%%%%%
    
    %update Nk (number of points in each class 'k')
    for k=1:K,
        Nk(k)=sum(responsibilities(:,k));
    end
    
    %re-estimate means
    for k=1:K,
        Ui{k}=responsibilities(:,k)'*cell2num(data)/Nk(k);
    end
    
    %re-estimate covariances
    tempCov=cell(N,1);
    for k=1:K,
        
        for n=1:N,
            tempCov{n}=responsibilities(n,k).*(data{n}-Ui{k})'*(data{n}-Ui{k});
        end
        total=zeros(2);
        for n=1:N,
            total=tempCov{n}+total;
        end
        Covi{k}=total/Nk(k);
    end
    
    %re-estimate Pi's  (mixing coefficients)
    for k=1:K
        Pii(k)=Nk(k)/numel(data);
    end
    
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    % Estimate Log Likelihood P(X) %
    %%%%%%%%%%%%%%%%%%%%%%%%%%%
    
    lnofpofx=0;
    
    temp2=0;
    for i=1:N,
        temp1=0;
        for k=1:K,
            temp1=[temp1,Pii(k)*gauss(Ui{k},Covi{k},data{i})];
        end
        temp2=[temp2,log(sum(temp1))];
    end
    loglik=sum(temp2);
    
    likeVec=[likeVec,loglik]; %loglikelihood history
    
    %     if change is less than threshold, were done
    disp(strcat('log likelihood=',num2str(likeVec(end)-likeVec(end-1))));
    if (abs(abs(likeVec(end))-abs(likeVec(end-1)))<change)
        break %done with EM algorithm, break out of loop
    end
    
end % do it all again

disp('*********************************');

%display color-coded EM classifications:
figure; hold on;
title('EMs classification: red=class 1, blue= class 2');
for n=1:N
    x=data{n};
    if max(responsibilities(n,:))==responsibilities(n,1),
        plot(x(1),x(2),'gX');
    else
        plot(x(1),x(2),'bX');
    end
end

disp('actual means of 2 gaussian components:')
Mu{1}
Mu{2}
disp('actual Covariances of 2 gaussians:')
Sigma{1}
Sigma{2}
disp('actual mixing coefficients:')
myPi

disp('Kmeans approximation of the means:')
U{1}
U{2}

disp('EM approximation of the means:')
Ui{1}
Ui{2}

disp('EM approximation of the Covariances:')
Covi{1}
Covi{2}

disp('EM approx. of mixing coefficients:')
Pii(1)
Pii(2)