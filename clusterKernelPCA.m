%clustering for KERNEL PCA
clear all; clc;
%load 'subjectData';
load 'subjectDatanew';

%NOT USEFUL!

for s=1:8
    D=subjectData{s,2};
    for i=1:numel(D)
        if D(i)==4,
            D(i)=3;
        elseif D(i)==5
            D(i)=4;
        end
    end
    subjectData{s,2}=D+1; clear T;
end %fix the D's

[X,D]=deal([],[]);
for s=1:size(subjectData,1)
    X=[X; subjectData{s,1}];
    D=[D; subjectData{s,2}];
end

%normalize data over each of 8 subjects
means=mean(X);
STDs=std(X);
for i=1:size(X,2),
    temp(:,i)=(X(:,i)-means(i))/STDs(i);
end
for i=1:size(X,1), %each sample entry
    Out(i,:)= temp(i,:)/sqrt(sum(temp(i,:).^2));
end
Out=temp;
X=Out; clear Out temp;clear i; clear STDs;
labs=[];
for i=1:size(subjectData,1)
    labs=[labs;i*ones(round(size(subjectData{i,1},1)),1)];
end

ldarez=zeros(8,1);
%linear PCA on normalized data
for sub=1:size(subjectData,1) %each subject
    %X=[X(D==1,:);X(D==2,:);X(D==5,:));
%     [vecs, vals]=princomp(X);
%     projLin=X*vecs;
%     figure; hold on;
%     scatter3(projLin(D==1,2),projLin(D==1,3),projLin(D==1,4),'.g'); %plot the awakes
%     scatter3(projLin(D==5,2),projLin(D==5,3),projLin(D==5,4),'.r'); %plot the REMS
%     scatter3(projLin(D==2,2),projLin(D==2,3),projLin(D==2,4),'.b'); %plot the N1s
    
    scores=classify(X(labs==sub),X(labs~=sub),D(labs~=sub));   
    ldarez(sub,1)=sum(scores==D(labs==sub))/numel(D(labs==sub));
end
ldarez
break

% figure; hold on; Kpc1=3; Kpc2=2;
% scatter(projLin(D==1,Kpc1),projLin(D==1,Kpc2),'g'); %plot the awakes
% scatter(projLin(D==5,Kpc1),projLin(D==5,Kpc2),'r'); %plot the REMS
break

%%%%%%%%%%%%
%Kernel PCA%
%%%%%%%%%%%%

N=500; %max # of clusters to inspect
[idx,C]=kmeans(X,N); 

% for k=100:25:N %k means with different # clusters
%     [idx,C]=kmeans(X,k); %idx~ labels, C = centers
%     histCheck(k)=var(hist(idx,ix));
%     disp(strcat('variance of the',num2str(k),'th mean =', num2str(histCheck(k))))
%     %check out histogram, choose k with bins that are most even (N=k|evenbins)
% end


%for dotProd=.01:.05:.5
[K1,K2,K3]=deal(zeros(N),zeros(N),zeros(N));%Gram Matrix for each kernel
spread=.1;    %gaussian kernel spread size
%dotProd=.01;  %dot-product squared kernel bias amt
for n=1:N
    for m=1:N        
        %K(n,m)=kernel(X(n,:),X(m,:))        
        %K1(n,m)=exp(-norm(C(n,:)-C(m,:)).^2/spread);
        %K2(n,m)=tanh(C(n,:)*C(m,:)');
        K3(n,m)=(C(n,:)*C(m,:)'+dotProd)^2;        
    end
end

Kn=K3;  %EDITME
Vp=10; %number of eigs

%tanh is nice (234)
%k3 is nice (123), not (234)

%'centralizing the kernel' 
oneN=ones(N)/N;
Kn= Kn - (oneN*Kn) - (Kn*oneN)+(oneN*(Kn*oneN)) ; %
[a,lambda]=eigs(Kn,Vp);                            %top 'Vp' eigs of Kn
%c=Kn/a*a'; %a=(Kn*a);  %? is this the same as   1=a(:,1)'*(Kn*a(:,1)) to normalize eigenvectors???

Knn=zeros(N); %create a data-friendly kernel
for n=1:N
    for m=1:size(X,1)                        
        %Knn(n,m)=exp(-norm(C(n,:)-X(m,:)).^2/spread);
        %K2(n,m)=tanh(C(n,:)*X(m,:)');
        Knn(n,m)=(C(n,:)*X(m,:)'+dotProd)^2;        
    end
end

pr%build classifier for each subject
for i=1:8,
    scores=classify();
    store %correct
endoj1=(Knn'*a);


% figure; hold on;
% scatter(proj1(d==1,1),proj1(d==1,2),'.b'); %plot the awakes
% scatter(proj1(d==-1,1),proj1(d==-1,2),'.r'); %plot the REMS 

% figure; hold on; comp1=1; comp2=2; comp3=3;
% scatter(proj1(D==1,comp1),proj1(D==1,comp2),'.g'); %plot the awakes
% scatter(proj1(D==2,comp1),proj1(D==2,comp2),'.b'); %plot the REMS
% scatter(proj1(D==5,comp1),proj1(D==5,comp2),'.r'); %plot the REMS
% title('dot prod kernel'); h = legend('Awake','REM','N1',3);
% xlabel('principle component 2'); ylabel('principle component 1');

% scatter3(proj1(D==1,comp1),proj1(D==1,comp2),proj1(D==1,comp3),'.g'); %plot the awakes
% scatter3(proj1(D==5,comp1),proj1(D==5,comp2),proj1(D==5,comp3),'.r'); %plot the REMS
% scatter3(proj1(D==2,comp1),proj1(D==2,comp2),proj1(D==2,comp3),'.b'); %plot the N1s
% h = legend('Awake','REM','N1',3);
% xlabel('principle component 1'); ylabel('principle component 1');
% title(strcat('Spread=',num2str(spread)));
end
goHandel