%My student number: r0766122
d1=7;
d2=6;
d3=6;
d4=2;
d5=2;

%Part 1
load("Files/Data_Problem1_regression.mat");
TNew = (d1*T1 + d2*T3 + d3*T3 + d4*T4 + d5*T5)/(d1+d2+d3+d4+d5);
%Training set
temp = datasample([X1 X2 TNew],1000,1);
trainingX = temp(:,1:2).';
trainingY = temp(:,3).';
trainingP = con2seq(trainingX);
trainingT = con2seq(trainingY);
%Validation set
temp = datasample([X1 X2 TNew],1000,1);
validationX = temp(:,1:2).';
validationY = temp(:,3).';
validationP = con2seq(validationX);
validationT = con2seq(validationY);
%Test set
temp = datasample([X1 X2 TNew],1000,1);
testX = temp(:,1:2).';
testY = temp(:,3).';
testP = con2seq(testX);
testT = con2seq(testY);


%Plot training set surface
x = trainingX(1,:).';
y = trainingX(2,:).';
xlin = linspace(min(x),max(x),33);
ylin = linspace(min(y),max(y),33);
[X,Y] = meshgrid(xlin,ylin);
f = scatteredInterpolant(x,y,trainingY.');
Z = f(X,Y);
figure
mesh(X,Y,Z) %interpolated
axis tight; hold on
plot3(X,Y,Z,'.','MarkerSize',10) %nonuniform

net=feedforwardnet(7,'trainlm'); %7 is 29 params

net.trainParam.epochs=1000;
[net,tr]=train(net,trainingP,trainingT);

postregm(cell2mat(sim(net,trainingP)),trainingY);
postregm(cell2mat(sim(net,validationP)),validationY);
%postregm(cell2mat(sim(net,con2seq([X1 X2].'))),TNew.');


%Plot MSEvsHiddenUnits in training and validation
maxHidden = 70;
for i=61:maxHidden
    nets{i} = feedforwardnet(i,'trainlm');
    nets{i}.trainParam.epochs=1000;
    [nets{i},tr{i}]=train(nets{i},trainingP,trainingT);
    simulationTraining{i}=sim(nets{i},trainingP);
    mseTraining{i} = mean((trainingY-cell2mat(simulationTraining{i})).^2);
    simulationValidation{i}=sim(nets{i},validationP);
    mseValidation{i} = mean((validationY-cell2mat(simulationValidation{i})).^2);
end
plot(cell2mat(mseTraining),'DisplayName','Training');
hold on;
plot(cell2mat(mseValidation),'DisplayName','Validation');
hold off;
set(gca, 'YScale', 'log')
xlabel('# Hidden Units');
ylabel('Mean Square Error');

%Plot MSEvsHiddenUnits in training and validation, multiple runs
maxHidden = 29;
for i=25:maxHidden
    nets{i} = feedforwardnet(i,'trainlm');
    nets{i}.trainParam.epochs=1000;
    [nets{i},tr{i}]=train(nets{i},trainingP,trainingT);
    simulationTraining{i}=sim(nets{i},trainingP);
    mseTraining{i} = mean((trainingY-cell2mat(simulationTraining{i})).^2);
    simulationValidation{i}=sim(nets{i},validationP);
    mseValidation{i} = mean((validationY-cell2mat(simulationValidation{i})).^2);
    for j=2:10
        tempnets = feedforwardnet(i,'trainlm');
        tempnets.trainParam.epochs=1000;
        [tempnets,temptr]=train(tempnets,trainingP,trainingT);
        tempsimulationTraining=sim(tempnets,trainingP);
        tempmseTraining = mean((trainingY-cell2mat(tempsimulationTraining)).^2);
        tempsimulationValidation=sim(tempnets,validationP);
        tempmseValidation = mean((validationY-cell2mat(tempsimulationValidation)).^2);
        if tempmseValidation<mseValidation{i}
            nets{i} = tempnets;
            tr{i} = temptr;
            simulationTraining{i}=tempsimulationTraining;
            mseTraining{i} = tempmseTraining;
            simulationValidation{i}=tempsimulationValidation;
            mseValidation{i} = tempmseValidation;
        end
    end
end
plot([25:45],cell2mat(mseTraining),'DisplayName','Training');
hold on;
plot([25:45],cell2mat(mseValidation),'DisplayName','Validation');
hold off;
set(gca, 'YScale', 'log')
xlabel('# Hidden Units');
ylabel('Mean Square Error');

%Plot MSEvsEpochs in training and validation
maxRuns = 30;
net = feedforwardnet(34,'trainlm');
for i=1:maxRuns
    net.trainParam.epochs=100;
    [net,tr]=train(net,trainingP,trainingT);
    simulationTraining{i}=sim(net,trainingP);
    mseTraining{i} = mean((trainingY-cell2mat(simulationTraining{i})).^2);
    simulationValidation{i}=sim(net,validationP);
    mseValidation{i} = mean((validationY-cell2mat(simulationValidation{i})).^2);
end
figure
plot(cell2mat(mseTraining),'DisplayName','TrainingLM');
hold on;
plot(cell2mat(mseValidation),'DisplayName','ValidationLM');
hold off;
set(gca, 'YScale', 'log')
xlabel('Epoch');
ylabel('Mean Square Error');


%Train the best NN multiple times
i=34;
net = feedforwardnet(i,'trainlm');
net.trainParam.epochs=2000;
[net,tr]=train(net,trainingP,trainingT);
simulationTraining=sim(net,trainingP);
mseTraining = mean((trainingY-cell2mat(simulationTraining)).^2);
simulationValidation=sim(net,validationP);
mseValidation = mean((validationY-cell2mat(simulationValidation)).^2);
for j=1:10
    tempnets = feedforwardnet(i,'trainlm');
    tempnets.trainParam.epochs=2000;
    [tempnets,temptr]=train(tempnets,trainingP,trainingT);
    tempsimulationTraining=sim(tempnets,trainingP);
    tempmseTraining = mean((trainingY-cell2mat(tempsimulationTraining)).^2);
    tempsimulationValidation=sim(tempnets,validationP);
    tempmseValidation = mean((validationY-cell2mat(tempsimulationValidation)).^2);
    if tempmseValidation<mseValidation
        nets = tempnets;
        tr = temptr;
        simulationTraining=tempsimulationTraining;
        mseTraining = tempmseTraining;
        simulationValidation=tempsimulationValidation;
        mseValidation = tempmseValidation;
    end
end
mseValidation;

%Test set error
mseTest = mean((testY-cell2mat(sim(net,testP))).^2);

%Plot test set surface
x = testX(1,:).';
y = testX(2,:).';
xlin = linspace(min(x),max(x),33);
ylin = linspace(min(y),max(y),33);
[X,Y] = meshgrid(xlin,ylin);
f = scatteredInterpolant(x,y,testY.');
Z = f(X,Y);
figure
mesh(X,Y,Z) %interpolated
axis tight; hold on
%plot3(x,y,z,'.','MarkerSize',10) %nonuniform

%Plot NN surface
x=testX(1,:).';
y=testX(2,:).';
xlin = linspace(min(x),max(x),33);
ylin = linspace(min(y),max(y),33);
[X,Y] = meshgrid(xlin,ylin);
outRows = size(X, 1);
outCols = size(Y, 2);
Z = zeros(outRows, outCols);
for row = 1:outRows
    for col = 1:outCols
        input = [X(row, col); Y(row, col)];
        simulated = sim(net,input);
        Z(row, col) = simulated;
    end
end
figure
mesh(X,Y,Z,'FaceAlpha',0.7);
axis tight; hold on
%plot3(x,y,trainingY.','.','MarkerSize',10,'DisplayName','Training') %Training set
%plot3(validationX(1,:).',validationX(2,:).',validationY.','.','MarkerSize',10,'DisplayName','Validation') %Validation set

%plot3(X1.',X2.',TNew.','.','MarkerSize',15,'DisplayName','All'); %All points


%Plot error surface
x = testX(1,:).';
y = testX(2,:).';
xlin = linspace(min(x),max(x),33);
ylin = linspace(min(y),max(y),33);
[X,Y] = meshgrid(xlin,ylin);
f = scatteredInterpolant(x,y,(testY - cell2mat(sim(net,testP))).^2.');
Z = f(X,Y);
figure;
mesh(X,Y,Z); %interpolated
axis tight; hold on;
contour3(X,Y,Z,100); %Error curves
plot3(trainingX(1,:).',trainingX(2,:).',trainingY.','.','MarkerSize',10); %nonuniform
hold off;