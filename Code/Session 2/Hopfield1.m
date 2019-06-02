T = [+1 +1; ...
      -1 -1;
      +1 -1].';

%%
% Here is a plot where the stable points are shown at the corners.  All possible
% states of the 2-neuron Hopfield network are contained within the plots
% boundaries.

plot(T(1,:),T(2,:),'r*')
axis([-1.1 1.1 -1.1 1.1])
title('Hopfield Network State Space')
xlabel('a(1)');
ylabel('a(2)');

%%
% The function NEWHOP creates Hopfield networks given the stable points T.

net = newhop(T);

%%
% First we check that the target vectors are indeed stable.  We check this by
% giving the target vectors to the Hopfield network.  It should return the two
% targets unchanged, and indeed it does.

[Y,Pf,Af] = sim(net,3,[],T);
Y

%%
% Here we define a random starting point and simulate the Hopfield network for
% 50 steps.  It should reach one of its stable points.

a = {rands(2,1)};
[y,Pf,Af] = sim(net,{1 50},{},a);

%%
% We can make a plot of the Hopfield networks activity.
% 
% Sure enough, the network ends up in either the upper-left or lower right
% corners of the plot.

record = [cell2mat(a) cell2mat(y)];
start = cell2mat(a);
hold on
plot(start(1,1),start(2,1),'bx',record(1,:),record(2,:))

%%
% We repeat the simulation for 250 more initial conditions.
% 
% Note that if the Hopfield network starts out closer to the upper-left, it will
% go to the upper-left, and vise versa.  This ability to find the closest memory
% to an initial input is what makes the Hopfield network useful.

color = 'rgbmy';
number_of_simulations = 250;
max_steps = 50;
average_no_iterations = 0;
hold on;
for i=1:number_of_simulations
   a = {rands(2,1)};
   [y,Pf,Af] = sim(net,{1 max_steps},{},a);
   record=[cell2mat(a) cell2mat(y)];
   start=cell2mat(a);
   plot(start(1,1),start(2,1),'kx',record(1,:),record(2,:),color(rem(i,5)+1));
   plot(record(1,max_steps),record(2,max_steps),'gO');  % plot the final point with a green circle
   
   for iter=1:max_steps
       if record(:,iter+1)==record(:,iter)
           break;
       end
   end
   average_no_iterations = average_no_iterations + iter-1;
end
hold off;
average_no_iterations = average_no_iterations/number_of_simulations;

%On average, slightly above 10 iterations. Of course, the closer to an
%attractor the faster the convergence.

%So, the number of real attractors is actually 4, since we also have [-1
%1]. There is also a point that is equidistant from the 4 attractors, and
%that's of course [0 0]. This point proves to be an unstable equilibrium
%point.

%Say something at some point about the fact that Nearest Neighbor works and
%is faster. Hopfield is a nice conceptual model, that opens an interesting
%parallelism with how memory works in animals, but is not a very efficient
%learning paradigm.