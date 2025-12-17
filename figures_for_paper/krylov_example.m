%{
The purpose of this script is to investigate the cost of matrix
exponentiation when the matrix is both large without special structure.

In this case, we can approximate the matrix exponential with a Krylov
subspace.
%}


clear;
addpath("KS_functions/");

n = 1024; %number of gridpoints
L = 100.0; %physical size of domain

x = (0:n-1)/n*2*pi;
k = 0:n-1;
k(k>n/2) = k(k>n/2) - n;

x = x';
k = k';
mask = abs(k) < n/3;
mask(1) = 0;

k = k*2*pi/L;

%initial condition
u = cos(x) + cos(2*x-1);


%%
m = 128;
t = 20;
steps = 256;

tic
[U, expm_times, rk4_times] = SLRK4(u,t,steps,m,k,mask);
toc

tiledlayout(1,2);

nexttile
imagesc(U);

nexttile
plot(expm_times, 'o');
hold on
  plot(rk4_times, 'o');
hold off
legend({'expm', 'everything else'});


%% Test generic RK6
%[a,b] = butcher_RK6a();
[a,b] = fehlberg_RK6();

[U, expm_times, rk6_times] = Lawson_RK6(a,b,u,t,steps,m,k,mask);

tiledlayout(1,2);

nexttile
imagesc(U)

nexttile
plot(expm_times, 'o');
hold on
  plot(rk6_times, 'o');
hold off
legend({'expm', 'everything else'});


%% Test SLRK6
[a,b] = my_rk6();

[U, expm_times_SL, rk6_times_SL] = Lawson_RK6(a,b,u,t,steps,m,k,mask);

tiledlayout(1,2);

nexttile
imagesc(U)

nexttile
plot(expm_times, 'o');
hold on
  plot(rk6_times, 'o');
  plot(expm_times_SL, 'o');
  plot(rk6_times_SL, 'o');
hold off
legend({'expm', 'everything else'});


