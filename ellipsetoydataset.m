%ellipse toy data

load 'ellipsedata' 
load 'halfmoon' 
figure; hold on;
scatter(X(d==1,1),X(d==1,2),'b.');
scatter(X(d==-1,1),X(d==-1,2),'r.');
N=size(X,1);
%title('ellipse toy data');

[K1,K2,K3]=deal(zeros(N),zeros(N),zeros(N));%Gram Matrix for each kernel
spread=.1;    %gaussian kernel spread size
dotProd=-1;  %dot-product squared kernel bias amt
for n=1:N
    for m=1:N        
        %K(n,m)=kernel(X(n,:),X(m,:))        
        K1(n,m)=exp(-norm(X(n,:)-X(m,:)).^2/spread);
        K2(n,m)=tanh(X(n,:)*X(m,:)');
        K3(n,m)=(X(n,:)*X(m,:)'+dotProd)^2;        
    end
end

Kn=K3;  %EDITME
Vp=2; %number of eigs

oneN=ones(N)/N;%'centralizing the kernel' 
Kn= Kn - (oneN*Kn) - (Kn*oneN)+(oneN*(Kn*oneN)) ; %
[a,lambda]=eigs(Kn,Vp);                            %top 'Vp' eigs of Kn
pause
%c=Kn/a*a'; %a=(Kn*a);  %? is this the same as   1=a(:,1)'*(Kn*a(:,1)) to normalize eigenvectors???


proj1=(Kn'*a);
figure; hold on;
scatter(proj1(d==1,1),proj1(d==1,2),'.b'); %plot the awakes
scatter(proj1(d==-1,1),proj1(d==-1,2),'.r'); %plot the REMS 
xlabel('principle component 1'); ylabel('principle component 2');
title(strcat('C=',num2str(spread)));

