function[Xts,Xtr]=fStandardizeML(Xtr,Xts)
%fStandardizeML centers and standardizes training (Xtr) and test(Xts) data. 
% Xtr, Xts are matrixes ~ [N X D], D  = #dimensions, N = #instances/samples.
% This is often applied as a preprocessing step on data before feeding into learning 
% algorithms.  % fStandardizeML requires separate training and test matrices
% because noobs often forget to separate test data from preprocessing steps.

[N,D] = size(Xtr);

%CENTER both matrices w/ params from Xtr:
Xtr=Xtr-repmat(mean(Xtr),N,1);
Xts=Xts-repmat(mean(Xtr),size(Xts,1),1);

%now both matrices have zero mean along dimensions D.

%STANDARDIZE both matrices w/ params from Xtr.
Xtr=Xtr./repmat(std(Xtr),N,1);

Xts=Xts./repmat(std(Xtr),size(Xts,1),1);

%now both matrices have unit variance along each dimension D

return 