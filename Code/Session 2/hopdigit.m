noiselevel = 10;
num_iter = 1000;
hopdigit_v2(noiselevel, num_iter);


%In general, the higher the noise the higher the number of iterations
%required to reach an attractor, because the noise pushes digits away from
%attractors. But, if the noise is too high, the network converges to a
%wrong attractor, and this is due to the fact that the noise was so high
%that the digit resembled another digit more than its original noiseless
%digit, resulting in a similar behavior to error correcting codes based on distance to
%the "attractor". It also happened for some inputs to reach unwanted
%spurious attractors, that do not represent any digit, which is common in
%Hopfield Networks. So, in general, the performance of the network is
%fairly good, since the only mistakes happen if the noise is way too high
%for the digit to be recognized as such.
%Using numbers, a noise level of 3 yields noisy digits that are quite visually
%unrecognizable, but still yields almost always perfect results. Higher
%noise levels greatly increase the chance of ending with a different
%attractor. In general, the number of required iterations is quite low, for
%example for noiselevel 3, 10 iterations are enough to reach an attractor,
%on average.