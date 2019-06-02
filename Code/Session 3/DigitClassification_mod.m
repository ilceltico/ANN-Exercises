clear all
close all
nntraintool('close');
nnet.guis.closeAllViews();

% Neural networks have weights randomly initialized before training.
% Therefore the results from training are different each time. To avoid
% this behavior, explicitly set the random number generator seed.
%rng('default') %Comment this to average the results over multiple runs


% Load the training data into memory
load('digittrain_dataset.mat');

numberOfRuns = 1;
for run=1:numberOfRuns
    % Layer 1
    hiddenSize1 = 100;
    autoenc1 = trainAutoencoder(xTrainImages,hiddenSize1, ...
        'MaxEpochs',400, ...
        'L2WeightRegularization',0.004, ...
        'SparsityRegularization',4, ...
        'SparsityProportion',0.4, ...
        'ScaleData', false);

    %figure;
    %plotWeights(autoenc1);
    feat1 = encode(autoenc1,xTrainImages);

    % Layer 2
   hiddenSize2 = 50;
   autoenc2 = trainAutoencoder(feat1,hiddenSize2, ...
       'MaxEpochs',100, ...
       'L2WeightRegularization',0.002, ...
       'SparsityRegularization',4, ...
       'SparsityProportion',0.4, ...
       'ScaleData', false);

   feat2 = encode(autoenc2,feat1);

    % Layer 3
    softnet = trainSoftmaxLayer(feat2,tTrain,'MaxEpochs',400);


    % Deep Net
    deepnet = stack(autoenc1,autoenc2,softnet);


    % Test deep net
    imageWidth = 28;
    imageHeight = 28;
    inputSize = imageWidth*imageHeight;
    load('digittest_dataset.mat');
    xTest = zeros(inputSize,numel(xTestImages));
    for i = 1:numel(xTestImages)
        xTest(:,i) = xTestImages{i}(:);
    end
    xTrain = zeros(inputSize,numel(xTrainImages));
    for i = 1:numel(xTrainImages)
        xTrain(:,i) = xTrainImages{i}(:);
    end
    y = deepnet(xTest);
    %figure;
    %plotconfusion(tTest,y);
    testAcc_noFineTuning(run)=100*(1-confusion(tTest,y))
    y = deepnet(xTrain);
    trainAcc_noFineTuning(run)=100*(1-confusion(tTrain,y))

    % Test fine-tuned deep net
    deepnet = train(deepnet,xTrain,tTrain);
    y = deepnet(xTest);
    %figure;
    %plotconfusion(tTest,y);
    testAcc_fineTuned(run)=100*(1-confusion(tTest,y))
    %view(deepnet);
    y = deepnet(xTrain);
    trainAcc_fineTuned(run)=100*(1-confusion(tTrain,y))

%for run=1:numberOfRuns
    %Compare with normal neural network (1 hidden layers)
    net = patternnet(50);
    net=train(net,xTrain,tTrain);
    y=net(xTest);
    %plotconfusion(tTest,y);
    testAcc1Hidden(run)=100*(1-confusion(tTest,y))
    %view(net)
    y=net(xTrain);
    trainAcc1Hidden(run)=100*(1-confusion(tTrain,y))

%for run=1:numberOfRuns
    %Compare with normal neural network (2 hidden layers)
    net = patternnet([70 30]);
    net=train(net,xTrain,tTrain);
    y=net(xTest);
    %plotconfusion(tTest,y);
    testAcc2Hidden(run)=100*(1-confusion(tTest,y))
    %view(net)
    y=net(xTrain);
    trainAcc2Hidden(run)=100*(1-confusion(tTrain,y))
end

meanTestAcc1Hidden=mean(testAcc1Hidden);
meanTestAcc2Hidden=mean(testAcc2Hidden);
meanTestAcc_fineTuned=mean(testAcc_fineTuned);
meanTestAcc_noFineTuned=mean(testAcc_noFineTuning);

meanTrainAcc1Hidden=mean(trainAcc1Hidden);
meanTrainAcc2Hidden=mean(trainAcc2Hidden);
meanTrainAcc_fineTuned=mean(trainAcc_fineTuned);
meanTrainAcc_noFineTuning=mean(trainAcc_noFineTuning);

bestTestAcc1Hidden=max(testAcc1Hidden);
bestTestAcc2Hidden=max(testAcc2Hidden);
bestTestAcc_fineTuned=max(testAcc_fineTuned);
bestTestAcc_noFineTuned=max(testAcc_noFineTuning);

