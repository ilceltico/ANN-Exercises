clear;

%Toy example to try getTimeSeriesTrainData
trainset = [1;2;3;4;5];
p_index = 2;
[trainingX,trainingY]=getTimeSeriesTrainData(trainset, p_index);

%Santa Fe
load("Files/lasertrain.dat");
%Normalization
trainingMean = mean(lasertrain);
trainingStd = sqrt(mean((lasertrain - trainingMean).^2));
lasertrain = (lasertrain-trainingMean)/trainingStd;
load("Files/laserpred.dat");
%Normalization with training parameters to avoid data snooping
laserpred = (laserpred - trainingMean)/trainingStd;

%pArray = [1 2 3 4 5 6 7 8 9 10 20 50 100 300];
%pArray = [3 4 5 7 10 20 100];
%pArray = [20 50 100 300];
pArray = [30];
%pArray = [5 10 20];
for p_index=1:length(pArray)
    [trainingX,trainingY]=getTimeSeriesTrainData(lasertrain, pArray(p_index));
    [testX,testY]=getTimeSeriesTrainData([lasertrain;laserpred],pArray(p_index));
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
    
    maxHidden = round(100/pArray(p_index) *2);
    step = max(1,round(maxHidden / 5));
    %for i=1:step:(maxHidden+1)
    for i=120:120
        nets{p_index}{i} = feedforwardnet(i,'trainscg');
        nets{p_index}{i}.trainParam.epochs=1000;
        [nets{p_index}{i},tr{p_index}{i}]=train(nets{p_index}{i},trainingP,trainingT);
        simulationTraining{p_index}{i}=sim(nets{p_index}{i},trainingP);
        mseTraining{p_index}{i} = mean((trainingY-cell2mat(simulationTraining{p_index}{i})).^2);

        %Closed loop test simulation
        mseValidation{p_index}{i} = 0;
        lastOutput = trainingY(end);
        input = trainingX(:,end);
        for j=1:length(testY)
            input = [input(2:end);lastOutput];
            lastOutput = sim(nets{p_index}{i},input);
            simulationValidation{p_index}{i}{j} = lastOutput;
            mseValidation{p_index}{i} = mseValidation{p_index}{i} + (lastOutput-testY(j))^2;
        end
        mseValidation{p_index}{i} = mseValidation{p_index}{i}/length(testY);
        
        for repetition=2:10
            tempNet = feedforwardnet(i,'trainscg');
            tempNet.trainParam.epochs=1000;
            [tempNet,tempTr]=train(tempNet,trainingP,trainingT);
            tempSimulationTraining=sim(tempNet,trainingP);
            tempMseTraining = mean((trainingY-cell2mat(tempSimulationTraining)).^2);

            %Closed loop test simulation
            tempMseValidation = 0;
            lastOutput = trainingY(end);
            input = trainingX(:,end);
            for j=1:length(testY)
                input = [input(2:end);lastOutput];
                lastOutput = sim(tempNet,input);
                tempSimulationValidation{j} = lastOutput;
                tempMseValidation = tempMseValidation + (lastOutput-testY(j))^2;
            end
            tempMseValidation = tempMseValidation/length(testY);
            
            if tempMseValidation<mseValidation{p_index}{i}
                nets{p_index}{i} = tempNet;
                tr{p_index}{i} = tempTr;
                simulationTraining{p_index}{i}=tempSimulationTraining;
                mseTraining{p_index}{i} = tempMseTraining;
                simulationValidation{p_index}{i}=tempSimulationValidation;
                mseValidation{p_index}{i} = tempMseValidation;
            end
        end
        
    end
end

for p_index=1:length(pArray)
    maxHidden = round(100/pArray(p_index) *100);
    step = max(1,round(maxHidden / 5));
    plot3(p_index*(ones(length(cell2mat(mseTraining{p_index})), 1)),1:step:maxHidden+1,cell2mat(mseTraining{p_index}),'DisplayName','Training');
    hold on;
    plot3(p_index*(ones(length(cell2mat(mseTraining{p_index})), 1)),1:step:maxHidden+1,cell2mat(mseValidation{p_index}),'DisplayName','Validation');
end
hold off;
set(gca, 'ZScale', 'log')
set(gca, 'XTickLabel',pArray) 
xlabel('p');
ylabel('# Hidden Units');
zlabel('Mean Square Error');
ylim([0 20]);


%Plot results
paramNo = 120;
p_index = 1;

[trainingX,trainingY]=getTimeSeriesTrainData(lasertrain, pArray(p_index));
[testX,testY]=getTimeSeriesTrainData([lasertrain;laserpred],pArray(p_index));
testX = testX(:,end-99:end);
testY = testY(end-99:end);
trainingP = con2seq(trainingX);
trainingT = con2seq(trainingY);
testP = con2seq(testX);
testT = con2seq(testY);

%postregm(cell2mat(sim(nets{chosenNet},trainingP)),trainingY);
%postregm(cell2mat(sim(nets{chosenNet},testP)),testY);

figure;
plot(trainingY,'DisplayName','Training set');
hold on;
plot(cell2mat(simulationTraining{p_index}{paramNo}),'DisplayName','NN');
hold off;

figure;
plot(testY,'DisplayName','Test set');
hold on;
plot(cell2mat(simulationValidation{p_index}{paramNo}),'DisplayName','NN');
hold off;


%Best result is with p=6 and 11 hidden neurons, apparently, probably due to
%lucky fitting, because in general results are not very good.