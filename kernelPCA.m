clear all;
load 'subjectData';
for s=1:8
    D=subjectData{s,2};
    T=numel(D);
    for i=1:T
        if D(i)==4,
            D(i)=3;
        elseif D(i)==5
            D(i)=4;
        end
    end
    subjectData{s,2}=D+1;
end %fix the D's

eigos=10;
%spread=.732;    %gaussian kernel spread size
%dotProd=.175;  
ldarez=zeros(8,1);
classifs=cell(8,1);
scores=cell(8,1);

for subj=1:8    
    [Xtrain,Xtest,Dtrain,Dtest]=deal([],[],[],[]);
    for s=1:size(subjectData,1)
        if s~=subj
            Xtrain=[Xtrain; subjectData{s,1}];
            Dtrain=[Dtrain; subjectData{s,2}];
        else
            Xtest=subjectData{s,1};
            Dtest=subjectData{s,2};
        end
    end  %create training set, test set (and Dtrain, Dtest)
    
    %normalize features wrt training set
    traini=size(Xtrain,1);
    all=[Xtrain;Xtest];
    means=mean(Xtrain);
    STDs=std(Xtrain);
    D=size(Xtrain,2);  %points to the last column of features
    for i=1:D,
        temp(:,i)=(all(:,i)-means(i))/STDs(i);
    end
    for i=1:size(all,1), %each sample entry
        Out(i,:)= temp(i,:)/sqrt(sum(temp(i,:).^2));
    end
    Xtrain=Out(1:traini,:);
    Xtest=Out(traini+1:end,:);
    clear Out; clear temp;   
    
%     N=100; %max # of clusters to inspect
%     [idx,C]=kmeans(Xtrain,N);
%     [Kn]=zeros(N);%Gram Matrix for each kernel
%     for n=1:N
%         for m=1:N
%             %K(n,m)=kernel(X(n,:),X(m,:))
%             %Kn(n,m)=C(n,:)*C(m,:)';
%             Kn(n,m)=exp(-norm(C(n,:)-C(m,:)).^2/spread);
%             %Kn(n,m)=tanh(C(n,:)*C(m,:)');
%             %Kn(n,m)=(C(n,:)*C(m,:)'+dotProd)^2;
%         end
%     end   
%     
%     oneN=ones(N)/N;    %'centralizing the kernel'
%     Kn= Kn - (oneN*Kn) - (Kn*oneN)+(oneN*(Kn*oneN)) ; %
%     [a,lambda]=eigs(Kn,eigos);                            %top 'Vp' eigs of Kn
%     
%     Knnts=zeros(N); %create a data-friendly kernel
%     for n=1:N
%         for m=1:size(Xtest,1)
%             %Knnts(n,m)=C(n,:)*Xtest(m,:)';
%             Knnts(n,m)=exp(-norm(C(n,:)-Xtest(m,:)).^2/spread);
%             %K2(n,m)=tanh(C(n,:)*X(m,:)');
%             %Knnts(n,m)=(C(n,:)*Xtest(m,:)'+dotProd)^2;            
%         end
%     end
%     Knntr=zeros(N); %create a data-friendly kernel
%     for n=1:N
%         for m=1:size(Xtrain,1)
%             %Knntr(n,m)=C(n,:)*Xtrain(m,:)';
%             Knntr(n,m)=exp(-norm(C(n,:)-Xtrain(m,:)).^2/spread);
%             %K2(n,m)=tanh(C(n,:)*X(m,:)');
%             %Knntr(n,m)=(C(n,:)*Xtrain(m,:)'+dotProd)^2;
%         end
%     end                   
%     
    
    for eigo=1:eigos    
%         Xtraind=(Knntr'*a(:,1:eigo));
%         Xtestd=(Knnts'*a(:,1:eigo));
        [a,Xtraind]=princomp(Xtrain);
        Xtestd=Xtest*a(:,1:eigo);
        Xtraind=Xtrain*a(:,1:eigo);
        newstates=classify(Xtestd,Xtraind,Dtrain);
        %context. rules:
        onset=min(find(newstates==2)); %the first epoch classified as N2, reliable onset detection
        for n=1:onset+90,
            if newstates(n)==5&&newstates(n+1)~=5,
                newstates(n)=newstates(n+1);
            elseif newstates(n)==5,
                newstates(n)=1;
            end
        end  %contextual rules
        ldarez(subj,eigo)=sum(newstates==Dtest)/numel(Dtest);
    end
    classifs{subj}=newstates;
    scores{subj}=Dtest;
    subj 
end
%end

goHandel
%ldarez


conf=zeros(5,5);
for s=1:8
    D=scores{s};
    X=classifs{s};
        
    for i=1:numel(D)        
        conf(D(i),X(i))=conf(D(i),X(i))+1;
    end          
end
conf



%sum(classifs{1}==scores{1})/numel(scores{1})
%total classifs:
%gauss kernel .735 15 eigs .7032
%linear pca        15      .7097
%dp kernel

return