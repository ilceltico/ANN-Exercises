clear;

%Toy example to try getTimeSeriesTrainData
trainset = [1;2;3;4;5];
p = 30;
[trainingX,trainingY]=getTimeSeriesTrainData(trainset, p);

%Santa Fe
load("Files/lasertrain.dat");
%Normalization
trainingMean = mean(lasertrain);
trainingStd = sqrt(mean((lasertrain - trainingMean).^2));
lasertrain = (lasertrain-trainingMean)/trainingStd;

load("Files/laserpred.dat");
%Normalization with training parameters to avoid data snooping
laserpred = (laserpred - trainingMean)/trainingStd;

p = 10;
[trainingX,trainingY]=getTimeSeriesTrainData(lasertrain, p);
[testX,testY]=getTimeSeriesTrainData([lasertrain;laserpred],p);
testX = testX(:,end-99:end);
testY = testY(end-99:end);
%By giving the vector directly to the training function, it behaves
%differently. It separates data into training validation and test, and of
%course optimizes training but checks on validation for generalization.
%Try instead to use cell vectors.
trainingP = con2seq(trainingX);
trainingT = con2seq(trainingY);
testP = con2seq(testX);
testT = con2seq(testY);

%net=feedforwardnet(33,'trainlm'); %n is n*(p+1+1)+1 parameters
%Training samples are 1000-p, so about 1000 for not huge p's. A good number of parameters is
%then 100, yielding 100/(p+2) hidden units.

%Plot MSEvsHiddenUnits in training and validation
maxHidden = 11;
for i=11:maxHidden
    nets{i} = feedforwardnet(i,'trainscg');
    nets{i}.trainParam.epochs=2000;
    %nets{i}.layers{2}.transferFcn = 'tansig';
    [nets{i},tr{i}]=train(nets{i},trainingP,trainingT);
    simulationTraining{i}=sim(nets{i},trainingP);
    mseTraining{i} = mean((trainingY-cell2mat(simulationTraining{i})).^2);
    
    %Closed loop test simulation
    mseValidation{i} = 0;
    lastOutput = trainingY(end);
    input = trainingX(:,end);
    for j=1:length(testY)
        input = [input(2:end);lastOutput];
        lastOutput = sim(nets{i},input);
        simulationValidation{i}{j} = lastOutput;
        mseValidation{i} = mseValidation{i} + (lastOutput-testY(j))^2;
    end
    mseValidation{i} = mseValidation{i}/length(testY);
end
plot(cell2mat(mseTraining),'DisplayName','Training');
hold on;
plot(cell2mat(mseValidation),'DisplayName','Validation');
hold off;
set(gca, 'YScale', 'log')
xlabel('# Hidden Units');
ylabel('Mean Square Error');


%Plot results
chosenNet = 11;
%postregm(cell2mat(sim(nets{chosenNet},trainingP)),trainingY);
%postregm(cell2mat(sim(nets{chosenNet},testP)),testY);

figure;
plot(trainingY,'DisplayName','Training set');
hold on;
plot(cell2mat(simulationTraining{chosenNet}),'DisplayName','NN');
hold off;

figure;
plot(testY,'DisplayName','Test set');
hold on;
plot(cell2mat(simulationValidation{chosenNet}),'DisplayName','NN');
hold off;
