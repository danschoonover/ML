%function[]=kmeans_batch(X,k)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K Means Clustering Algorithm - Batch Processing %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% K means clustering of a GMM with early stopping convergence criteria.

clear all; clc; close all;
%initialize GMM Params:
K=2; %number of components of the GMM
D=2; %dimension of data being modeled

N=10000; %number of samples to draw from GMM
%% Generate and Sample from 2 gaussians 
%  w/ mean vector 'U' & Cov. matrix 'Sigma'

myPi=[.5 .5]; %2-component GMM Mixing Coefficients

Mu=cell(2,1);
Sigma=cell(2,1);
Mu{1}=[3 3];
Sigma{1}=[2 1;1 2];
Mu{2}=[-3 -3];
Sigma{2}=[3 0;0 3];

%2 GMM components (coded as 2 obj's for legibility)
Gauss1=gmdistribution(Mu{1},Sigma{1});%or (mu,sigma p (mixing coeff))
Gauss2=gmdistribution(Mu{2},Sigma{2});

data=zeros(N,D); %will hold samples drawn from the GMM

for i=1:N, %sample from the 2 gaussians
    if rand<myPi(1) %if unif RV falls under comp 1
        data(i,:)=random(Gauss1);   %sample from gaussian 1
    else
        data(i,:)=random(Gauss2);   %else sample gaussian 2
    end
end

%% PLOT GMMs:
figure; hold on; title('k means');
plot(data(:,1),data(:,2),'bx');
title('Gauss. Mixture Model, Press Enter')
pause;

%% K Means Clustering Algorithm

count=25;  %Max # of iterations before quitting
ESCrit=1e-4; %converg. criteria: k means must move by ESCrit to continue

verbose=0;

U=zeros(K,D);
if verbose
    disp('u(X):');
    disp(mean(data));
end

for j=1:K %initialize (component) cluster centers
    U(j,:)=mean(data)+randn.*var(data); %
    
    plot(U(j,:),'rO'); plot(U(j,:),'kX');  %update plots
end

for j=1:K %initialize (component) cluster centers
    scatter(U(j,1),U(j,2),'kO'); scatter(U(j,1),U(j,2),'kX'); %update plots
end
if verbose
    disp('first U:')
    disp(U);
end

test=zeros(N,K);
distk=zeros(N,K);
lastU=[];

for i=1:count
    
    Rnk=zeros(N,K); % data/latent variable assignments (1-of-k coding)
    
    if verbose
        disp(sprintf('iteration %d', i));    
        disp(U);
    end
    
    for n=1:N %step 1 Optimize Rnk holding U constant
        
        for j=1:K %compute distances to updated cluster centers (U's)
            distk(n,j)=sqrt(sum((data(n,:)-U(j,:)).^2));
        end
        
        %step 2 ReAssign Rnk based on updated U's
        cl=distk(n,:)==min(distk(n,:)); %closest cluster center index        
        Rnk(n,cl)=1;
    end
    
    %% Step 2 Optimize U holding Rnk constant
    
    lastU=U; %store this value of U for convergence checks               
               
    colorlist={'rx', 'bx', 'gx'}; %add more colors if trying k>3
    
    clf; hold on;
    
    for j=1:K  % Update the k cluster means ~ U(k)
       
        list=find(Rnk(:,j)); %find all data in cluster j
        
        plot(data(list,1),data(list,2),colorlist{j}); %plot data colored by center
        
        U(j,:)=Rnk(:,j)'*data./sum(Rnk(:,j)); %update centers U
       
        scatter(U(j,1),U(j,2),'kO'); scatter(U(j,1),U(j,2),'kX'); %update plots
        title(sprintf('k-means clustering: iter. # %d', i))
    end
    
     %% CHECK FOR CONVERGENCE (last)
     if verbose
         disp('U distances')
         if sqrt(sum((U-lastU).^2)) < ESCrit
             disp('convergence criteria reached! Exiting..');
             break
         end
     end
    
    if sum(sum(isnan(U))) %if a center collapses into infinity ..      
        for j=1:K %reset K means ..
            U(j,:)=mean(data)+randn.*var(data); % +randn(1,2) init. k means: u(X) + noise
        end        
        i=1; %and start over.    
    end     
    
    pause(.001);
end

if verbose %display estimated priors
    disp(sprintf('prior estimates: %d %d', double(sum((Rnk)/N)) ));
end

return