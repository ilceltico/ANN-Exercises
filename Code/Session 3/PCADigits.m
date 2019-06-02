
clear;

load('Files/threes.mat','-ascii');
image_no=1;
colormap('gray');
imagesc(reshape(threes(image_no,:),16,16),[0,1]);

%Display the average "3"
average = mean(threes,1);
colormap('gray');
imagesc(reshape(average,16,16),[0,1]);

%Plot eigenvalues
covarianceMatrix = cov(threes);
[eigenvecs,eigenvals] = eig(covarianceMatrix); %Eigenvectors are column vectors in the matrix
eigenvals = diag(eigenvals);
plot(eigenvals);

maxQ = size(threes,2);
zeroMeanD = threes - mean(threes); %Column-wise mean

for i=1:maxQ
    covarianceMatrix = cov(zeroMeanD);
    [eigenvecs,eigenvals] = eigs(covarianceMatrix,i); %Eigenvectors are column vectors in the matrix
    eigenvals = diag(eigenvals);

    transformedDataset = eigenvecs.' * zeroMeanD.';

    reconstructedZeroMeanDataset{i} = eigenvecs * transformedDataset;

    reconstructedZeroMeanDataset{i} = reconstructedZeroMeanDataset{i}.';

    reconstructionError(i) = sqrt(mean(mean((zeroMeanD-reconstructedZeroMeanDataset{i}).^2))); %Root mean square difference
    %i
    
    reconstructedDataset{i} = reconstructedZeroMeanDataset{i} + average;
end


%Plot some reconstructed images
image_no = 10;
figure;
s = subplot(1,5,1);
colormap('gray');
imagesc(reshape(threes(10,:),16,16),[0,1]);
title('Original');
axis image;
subplot(1,5,2);
imagesc(reshape(reconstructedDataset{1}(image_no,:),16,16),[0,1]);
title('1 eigenvalue');
axis image;
subplot(1,5,3);
imagesc(reshape(reconstructedDataset{2}(image_no,:),16,16),[0,1]);
title('2 eigenvalues');
axis image;
subplot(1,5,4);
imagesc(reshape(reconstructedDataset{3}(image_no,:),16,16),[0,1]);
title('3 eigenvalues');
axis image;
subplot(1,5,5);
imagesc(reshape(reconstructedDataset{4}(image_no,:),16,16),[0,1]);
title('4 eigenvalues');
axis image;

image_no = 100;
figure;
subplot(1,5,1);
colormap('gray');
imagesc(reshape(threes(100,:),16,16),[0,1]);
title('Original');
axis image;
subplot(1,5,2);
imagesc(reshape(reconstructedDataset{1}(image_no,:),16,16),[0,1]);
title('1 eigenvalue');
axis image;
subplot(1,5,3);
imagesc(reshape(reconstructedDataset{2}(image_no,:),16,16),[0,1]);
title('2 eigenvalues');
axis image;
subplot(1,5,4);
imagesc(reshape(reconstructedDataset{3}(image_no,:),16,16),[0,1]);
title('3 eigenvalues');
axis image;
subplot(1,5,5);
imagesc(reshape(reconstructedDataset{4}(image_no,:),16,16),[0,1]);
title('4 eigenvalues');
axis image;

image_no = 200;
figure;
subplot(1,5,1);
colormap('gray');
imagesc(reshape(threes(200,:),16,16),[0,1]);
title('Original');
axis image;
subplot(1,5,2);
imagesc(reshape(reconstructedDataset{1}(image_no,:),16,16),[0,1]);
title('1 eigenvalue');
axis image;
subplot(1,5,3);
imagesc(reshape(reconstructedDataset{2}(image_no,:),16,16),[0,1]);
title('2 eigenvalues');
axis image;
subplot(1,5,4);
imagesc(reshape(reconstructedDataset{3}(image_no,:),16,16),[0,1]);
title('3 eigenvalues');
axis image;
subplot(1,5,5);
imagesc(reshape(reconstructedDataset{4}(image_no,:),16,16),[0,1]);
title('4 eigenvalues');
axis image;


%Plot reconstruction error and cumsum of eigenvalues
figure;
yyaxis left;
plot(reconstructionError.^2,'DisplayName','Reconstruction Error');
%set(gca, 'YScale', 'log')
xlabel('Number of used eigenvalues');
ylabel('Mean Squared Error');
%ylim([0 max(reconstructionError)]);
xlim([0 maxQ]);

[eigenvecs,eigenvals] = eig(covarianceMatrix);
eigenvals = flipud(diag(eigenvals)); %They were sorted in increasing value
%remainingEigenvals = cumsum(eigenvals, 'reverse');
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


%Reconstruction error with all the eigenvalues
reconstructionError(end)