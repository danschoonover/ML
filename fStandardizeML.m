function[Xts,Xtr]=fStandardizeML(Xts,Xtr)
%fStandardizeML centers and standardizes training (Xtr) and test(Xts) data. 
% Xtr, Xts are matrixes ~ [D x N], 
% where D is # features/dimensions, N is # instances/samples.
% Standardization centers to zero and standard deviation to 1. 
% It is often applied as a preprocessing step on data before feeding into learning 
% algorithms.  % fStandardizeML requires separate training and test matrices
% because noobs often forget to separate test data from preprocessing steps.


[D,N] = size(X);

%CENTER both matrices w/ params from Xtr:
Xtr=Xtr-repmat(mean(Xtr'),size(Xtr,2),1)';
Xts=Xts-repmat(mean(Xtr'),size(Xts,2),1)'; 
%now both matrices have zero mean along dimensions D.

%STANDARDIZE both matrices w/ params from Xtr.
Xtr=Xtr./repmat(std(Xtr'),size(Xtr,2),1)';
Xts=Xts./repmat(std(Xtr'),size(Xts,2),1)';
%now both matrices have unit variance for each dimension D

return 