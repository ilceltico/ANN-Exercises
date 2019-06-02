clear;

load("Files/lasertrain.dat");
load("Files/laserpred.dat");

data = [lasertrain.',laserpred.'];

figure
plot(data)
xlabel("Timestep")
title("Santa Fe Laser Data")

numTimeStepsTrain = 1000;

dataTrain = data(1:numTimeStepsTrain);
dataTest = data(numTimeStepsTrain:end);

mu = mean(dataTrain);
sig = std(dataTrain);

dataTrainStandardized = (dataTrain - mu) / sig;

XTrain = dataTrainStandardized(1:end-1);
YTrain = dataTrainStandardized(2:end);

numFeatures = 1;
numResponses = 1;
numHiddenUnits = 50;

layers = [ ...
    sequenceInputLayer(numFeatures)
    lstmLayer(numHiddenUnits)
    fullyConnectedLayer(numResponses)
    regressionLayer];

options = trainingOptions('adam', ...
    'MaxEpochs',1000, ...
    'GradientThreshold',1, ...
    'InitialLearnRate',0.005, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropPeriod',150, ...
    'LearnRateDropFactor',0.5, ...
    'Verbose',0, ...
    'Plots','training-progress');

net = trainNetwork(XTrain,YTrain,layers,options);

dataTestStandardized = (dataTest - mu) / sig;
XTest = dataTestStandardized(1:end-1);

net = predictAndUpdateState(net,XTrain);
[net,YPred] = predictAndUpdateState(net,YTrain(end));

numTimeStepsTest = numel(XTest);
for i = 2:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,YPred(:,i-1),'ExecutionEnvironment','cpu');
end

YPred = sig*YPred + mu;

YTest = dataTest(2:end);
rmse = sqrt(mean((YPred-YTest).^2));

%Standardized MSE to compare to previous exercise
YPred = (YPred - mu)/sig;
mseTest = mean((YPred-dataTestStandardized(2:end)).^2);
YPred = sig*YPred + mu;

figure
plot(dataTrain(1:end-1))
hold on
idx = numTimeStepsTrain:(numTimeStepsTrain+numTimeStepsTest);
plot(idx,[data(numTimeStepsTrain) YPred],'.-')
hold off
xlabel("Month")
ylabel("Cases")
title("Forecast")
legend(["Observed" "Forecast"])

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Forecast"])
ylabel("Cases")
title("Forecast")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)




net = resetState(net);
net = predictAndUpdateState(net,XTrain);

YPred = [];
numTimeStepsTest = numel(XTest);
for i = 1:numTimeStepsTest
    [net,YPred(:,i)] = predictAndUpdateState(net,XTest(:,i),'ExecutionEnvironment','cpu');
end

YPred = sig*YPred + mu;

rmse = sqrt(mean((YPred-YTest).^2));

figure
subplot(2,1,1)
plot(YTest)
hold on
plot(YPred,'.-')
hold off
legend(["Observed" "Predicted"])
ylabel("Cases")
title("Forecast with Updates")

subplot(2,1,2)
stem(YPred - YTest)
xlabel("Month")
ylabel("Error")
title("RMSE = " + rmse)


%First of all here we are in the deep learning tools, the algorithm used is
%adam, which is suggested by literature to be more optimized than GD for
%deep learning and is a recent very popular trend. Anyway, of course, it is
%much slower than the previous methods used for shallow networks, since the
%complexity of the NN here is higher.

%Using more parameters it's better to decrease the learning rate after
%smaller amounts of epochs, for example after 50 epochs multiply by 0.5,
%instead of the 125 epochs 0.2 of the
%TimeSeriesForecastingUsingDeepLearningExample.
%Using less parameters, even 200 epochs 0.5 seems fine.

%Good results starting from as low as 30 hidden units! And with much more
%ease than with Time Series NN. Moreover, this model correctly reproduces
%the big drop in absolute value of the signal that happens at around 60 in
%the test set. This, in fact, requires knoledge of the previous drops, that
%are very far for the Time Series NN model, while this model can account
%for them thanks to its special structure that has an internal status whose
%purpose is to learn dependecies between the next value and the past
%history, not being limited by the number of inputs as for the previous NN.
%Also the first predictions in the future appear more reliable than
%previous NN.

%'adam' algorithm tends to produce better results than 'sgdm' and 'rmsprop'.
%Solver for training network, specified as one of the following:
%'sgdm' — Use the stochastic gradient descent with momentum (SGDM) optimizer. You can specify the momentum value using the 'Momentum' name-value pair argument.
%'rmsprop'— Use the RMSProp optimizer. You can specify the decay rate of the squared gradient moving average using the 'SquaredGradientDecayFactor' name-value pair argument.
%'adam'— Use the Adam optimizer. You can specify the decay rates of the gradient and squared gradient moving averages using the 'GradientDecayFactor' and 'SquaredGradientDecayFactor' name-value pair arguments, respectively.
%For more information about the different solvers, see Stochastic Gradient Descent.