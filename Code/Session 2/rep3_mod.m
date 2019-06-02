%%%%%%%%%%%
% rep3.m
% A script which generates n random initial points for
% and visualise results of simulation of a 3d Hopfield network net
%%%%%%%%%%
T = [1 1 1; -1 -1 1; 1 -1 -1]';
net = newhop(T);
n=250;
max_steps = 100;
average_no_iterations = 0;
for i=1:n
    a={rands(3,1)};                         % generate an initial point                   
    [y,Pf,Af] = sim(net,{1 max_steps},{},a);       % simulation of the network  for 50 timesteps
    record=[cell2mat(a) cell2mat(y)];       % formatting results
    start=cell2mat(a);                      % formatting results 
    plot3(start(1,1),start(2,1),start(3,1),'bx',record(1,:),record(2,:),record(3,:),'r');  % plot evolution
    hold on;
    plot3(record(1,max_steps),record(2,max_steps),record(3,max_steps),'gO');  % plot the final point with a green circle

   for iter=1:max_steps
      if record(:,iter+1)==record(:,iter)
          break;
      end
   end
   average_no_iterations = average_no_iterations + iter-1;
end
average_no_iterations = average_no_iterations/n;
grid on;
legend('initial state','time evolution','attractor','Location', 'northeast');
title('Time evolution in the phase space of 3d Hopfield model');

%Generate points in a more systematic way
interval_no = 10;
for i=1:interval_no+1
    for j=1:interval_no+1
        for k=1:interval_no+1
            tempX = 2/interval_no*(i-1)-1;
            tempY = 2/interval_no*(j-1)-1;
            tempZ = 2/interval_no*(k-1)-1;
            a={[tempX;tempY;tempZ]};                         % generate an initial point                   
            [y,Pf,Af] = sim(net,{1 100},{},a);       % simulation of the network  for 50 timesteps
            record=[cell2mat(a) cell2mat(y)];       % formatting results
            start=cell2mat(a);                      % formatting results 
            plot3(start(1,1),start(2,1),start(3,1),'bx',record(1,:),record(2,:),record(3,:),'r');  % plot evolution
            hold on;
            plot3(record(1,100),record(2,100),record(3,100),'gO');  % plot the final point with a green circle
        end
    end
end
grid on;
legend('initial state','time evolution','attractor','Location', 'northeast');
title('Time evolution in the phase space of 3d Hopfield model');

%Higher average number of iterations, since we have 1 more dimension: slightly above 13.

%Here we don't have any other stable attractor, a possible cause being the
%fact that the neurons here are 3, while in the previous exercise they were
%2. Storing without significant crosstalk a number of attractors higher than the number
%of neurons seems, in fact, unlikely.

%Plotting random points shows no other stable attractors. To show the
%unstable points deriving from being equidistant from stable attractors,
%plot an equally spaced point grid. We can see the center of the triangle
%being of course an unstable equilibrium, but also the midpoints of the
%edges of the triangle.