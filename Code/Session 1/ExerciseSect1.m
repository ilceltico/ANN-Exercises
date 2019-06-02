%Linearily separable
X = [ -0.2 -0.6 +0.3 -0.4 -0.8 2 0.6;
      -0.5 +0.4 -0.1 +1.9 +0.7 4 -5];
%Non-linearily separable
%X = [ -0.2 -0.6 +0.3 -0.4 -0.8 2 0.6;
%      -0.5 +0.4 -0.1 +1.9 +0.7 4 -5];
T = [1 1 0 0 1 0 1];
plotpv(X,T);
net = newp(X,T,'hardlim','learnp');
linehandle = plotpc(net.IW{1},net.b{1});

%Different line obtained with the function train, batch learning
%net = train(net,X,T);
%plotpc(net.IW{1},net.b{1});

for a = 1:25
    [net,Y,E] = adapt(net,X,T);
    linehandle = plotpc(net.IW{1},net.b{1},linehandle);  drawnow;
end;

