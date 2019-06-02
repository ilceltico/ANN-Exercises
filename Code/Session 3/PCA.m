%Apparently, the cov function estimates the covariance by dividing the sum
%of the values by N-1, instead of N, where N is the number of observations.
%The rows are observations, the columns are variables

clear;

%Random dataset
%dataset = randn(50,500); %500 points, 50 dimensions
%dataset = dataset.';

%Highly correlated dataset
load choles_all; %264 points, 21 dimensions
dataset = p.';

maxQ = size(dataset,2);
zeroMeanD = dataset - mean(dataset); %Column-wise mean

for i=1:maxQ
    covarianceMatrix = cov(zeroMeanD);
    [eigenvecs,eigenvals] = eigs(covarianceMatrix,i); %Eigenvectors are column vectors in the matrix
    eigenvals = diag(eigenvals);

    transformedDataset = eigenvecs.' * zeroMeanD.';

    reconstructedDataset = eigenvecs * transformedDataset;

    reconstructedDataset = reconstructedDataset.';

    reconstructionError(i) = sqrt(mean(mean((zeroMeanD-reconstructedDataset).^2))); %Root mean square difference
    i
end

figure;
plot(reconstructionError,'DisplayName','Reconstruction Error');
%set(gca, 'YScale', 'log')
xlabel('Number of used eigenvalues');
ylabel('Root Mean Squared Error');
%ylim([0 max(reconstructionError)]);
xlim([0 maxQ]);

[eigenvecs,eigenvals] = eig(covarianceMatrix);
eigenvals = flipud(diag(eigenvals)); %They were sorted in increasing value
hold on;
plot(eigenvals,'DisplayName','Sorted Eigenvalues');
%set(gca, 'YScale', 'log')
%ylim([0 max(eigenvals)]);
%xlim([0 maxQ]);
hold off;


figure;
yyaxis left;
plot(reconstructionError,'DisplayName','Reconstruction Error');
%set(gca, 'YScale', 'log')
xlabel('Number of used eigenvalues');
ylabel('Root Mean Squared Error');
%ylim([0 max(reconstructionError)]);
xlim([0 maxQ]);

[eigenvecs,eigenvals] = eig(covarianceMatrix);
eigenvals = flipud(diag(eigenvals)); %They were sorted in increasing value
remainingEigenvals(1) = sum(eigenvals)-eigenvals(1);
for i=2:maxQ
    remainingEigenvals(i) = remainingEigenvals(i-1)-eigenvals(i);
end
hold on;
yyaxis right;
plot(remainingEigenvals,'DisplayName','Sum of remaining eigenvalues');
%set(gca, 'YScale', 'log')
%ylim([0 max(eigenvals)]);
%xlim([0 maxQ]);
hold off;





[normalizedD, stdPS]= mapstd(dataset);
[reducedDataset, PS] = processpca(normalizedD.', 0.001);
reconstructedDataset = processpca('reverse',reducedDataset,PS);
RMSE = sqrt(mean(mean((dataset-mapstd('reverse',reconstructedDataset.',stdPS)).^2)));
