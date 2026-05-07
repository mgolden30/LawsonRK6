%{
Apply SLRK4 and SLRK6 to van der Pol oscillator in the stiff regime.
%}

clear;
clf;

x = [2; 0]; %initial condition
mu = 100; %stiff parameter

%Operator splitting
A_fn = @(x0) [0, 1; -2*mu*x0(1)*x0(2) - 1, mu*(1-x0(1)^2)];
g_fn = @(x0) @(x) [0; -mu*(x(1)^2 - x0(1)^2)*x(2) + 2*mu*x0(1)*x0(2)*x(1)];

t = 200; %time of flight
m_fine = 256 * 1024; %very fine timestepping as a reference

tic
xs_fine = slrk6_v2(x, A_fn, g_fn, t, m_fine);
toc

scatter(xs_fine(1,:), xs_fine(2,:), 's', ...
    'filled', ...
   'MarkerFaceColor','red');



%% Run a sweep
ms = round(exp(linspace(log(4*1024), log(128*1024), 64)));
points = numel(ms);

err_4 = zeros(points,1);
err_6 = zeros(points,1);

tic
for i = 1:points
  xs = slrk4_v2(x, A_fn, g_fn, t, ms(i) );  
  err_4(i) = max(abs(xs(:,end) - xs_fine(:,end)), [], "all");

  xs = slrk6_v2(x, A_fn, g_fn, t, ms(i) );  
  err_6(i) = max(abs(xs(:,end) - xs_fine(:,end)), [], "all");
end
toc

%%
clf;
close all;
fig = figure;
fig.Units = 'inches';
% [left bottom width height]
fig.Position = [1 1 7 3.5];

tiledlayout(1,2);
nexttile
plot(xs_fine(1,:), xs_fine(2,:), 'linewidth', 3, 'color', 'k');
xlabel("$y_1$", "Interpreter", "latex");
ylabel("$y_2$", "Interpreter", "latex", "rotation", 0);
xlim([-1,1] * 2.5)
yticks([-1,0,1] * 150);
set(gca, "fontsize", 12);
axis square

nexttile


%generate power law fits to the error
indices = [48:64];
[ms_4, err_4b, fit_4, p4] = fit_power_law(ms, err_4, indices );

indices = [48:64];
[ms_6, err_6b, fit_6, p6] = fit_power_law(ms, err_6, indices );

indices = [28:36];
[ms_7, err_7b, fit_7, p7] = fit_power_law(ms, err_6, indices );


%Call the nondimensional quantity s
s_4 = mu*(t./ms_4);
s_6 = mu*(t./ms_6);
s_7 = mu*(t./ms_7);
s = mu*(t./ms);

marker_size = 20;
plot([], []);
hold on
%visulize all data points, regardless of "indices" used to fit the power
%law
plot( s_4, fit_4, "-", "color", "black", "LineWidth", 2);
plot( s_6, fit_6, "-", "color", "black", "LineWidth", 2 );
plot( s_7, fit_7, "-", "color", "black", "LineWidth", 2 );
scatter(s, err_4, marker_size, 's', 'filled', "markerfacecolor", "black");
scatter(s, err_6, marker_size, 'd', "markeredgecolor", "black", "markerfacecolor", "w", "LineWidth", 2);
hold off

legend({"", "", "", "SLRK4", "SLRK6", ""}, "location", "southeast", "interpreter", "latex");
set(gca, "xscale", "log");
set(gca, "yscale", "log");

set(gca, "fontsize", 32);
xlabel( "$\mu \Delta t$", "Interpreter", "latex");
ylabel( "max absolute error", "Interpreter", "latex");

set(gcf, "color", "w");


box on
str = sprintf( "$ m^{%.2f}$", p4(1) );
tx = 0.2;
ty = 1e-2;
text(tx, ty, str, 'FontSize', 12, 'Interpreter', 'latex');

str = sprintf( "$ m^{%.2f}$", p6(1) );
tx = 0.2;
ty = 1e-5;
text(tx, ty, str, 'FontSize', 12, 'Interpreter', 'latex');

str = sprintf( "$ m^{%.2f}$", p7(1) );
tx = 0.5;
ty = 1e-2;
text(tx, ty, str, 'FontSize', 12, 'Interpreter', 'latex');

xticks([0.25, 0.5, 1]);

xlim( [0.000001, 2] .* xlim() );
xlim([0.1326, 1.4040]);
ylim([ 5e-8, 5.3597]);
yticks([1e-6, 1e-3, 1e0]);
axis square;
set(gca, "fontsize", 12);

set(gcf,'Renderer','painters');
exportgraphics(gcf, 'figures/VDP.pdf', ...
    'ContentType', 'vector', ....
    'BackgroundColor','white');
return

set(groot, 'defaultAxesTickLabelInterpreter','latex');
labels = {"$10^{-3}$", "$10^{-6}$", "$10^{-9}$", "$\varepsilon_{\textrm{floor}}$", "$10^{-12}$"};
yticklabels( flip(labels) );

%exportgraphics(gcf,'figures/convergence_NS.png','Resolution',512)
%saveas(gcf, "figures/convergence_NS.png");



function xs = slrk4_v2(x, A_fn, g_fn, t, m)
  h = t/m;
  xs = zeros(2,m+1);
  xs(:,1) = x; 
  for i = 1:m
    A = A_fn(x);
    g = g_fn(x);
    e = expm(A*h/2);
  
    k1 = h*g(x);
    x = e*x; k1 = e*k1;
    k2 = h*g(x + k1/2);
    k3 = h*g(x + k2/2);
    x = e*x; k1 = e*k1; k2 = e*k2; k3 = e*k3;
    k4 = h*g(x + k3);
    x = x + (k1 + 2*k2 + 2*k3 + k4)/6;
    xs(:,i+1) = x;
  end
end


function xs = slrk6_v2(x, A_fn, g_fn, t, m)
  h = t/m;
  xs = zeros(2,m+1);
  xs(:,1) = x;
  for i = 1:m
     A = A_fn(x);
     g = g_fn(x);
     e = expm(A*h/6);

    k1 = h*g(x);
    x = e*x; k1 = e*k1;
    k2 = h*g(x + k1/6);
    k3 = h*g(x + k1/12 + k2/12 );
    x = e*x; k1 = e*k1; k2 = e*k2; k3 = e*k3;
    k4 = h*g(x - k2*4/33 + k3*5/11);
    x = e*x; k1 = e*k1; k2 = e*k2; k3 = e*k3; k4 = e*k4;
    k5 = h*g(x - k1/4 - k2*29/44 + k3*31/22);
    x = e*x; k1 = e*k1; k2 = e*k2; k3 = e*k3; k4 = e*k4; k5 = e*k5;
    k6 = h*g(x + k1*3/11 + k2*8/33 - k3*4/11 + k4/11 + k5*14/33);
    x = e*x; k1 = e*k1; k2 = e*k2; k3 = e*k3; k4 = e*k4; k5 = e*k5; k6 = e*k6;
    k7 = h*g(x - k1*17/48 - k2*5/12 + k3 + k4 - k5*13/12 + k6*11/16);
    x = e*x; k1 = e*k1; k2 = e*k2; k3 = e*k3; k4 = e*k4; k5 = e*k5; k6 = e*k6; k7 = e*k7;
    k8 = h*g(x + k1*20/39 + k2*12/39 - k3*31/39 - k4/39 + k5*34/39 - k6*11/39 + k7*16/39);

    x = x + 13/200*(k1 + k8) + 4/25*(k3 + k7) + 11/40*(k4 + k6);    
    xs(:,i+1) = x;
  end
end



function [ms, err, fit, p] = fit_power_law(ms, err, indices)
  %{
  Do a power law fit to maximum error
  %}
 
  if numel(indices) > 0
    %Truncate to a specific set of indices
    ms = ms(indices);
    err= err(indices);
  end

  %Discard the points with NAN error.
  %The simulation blew up, and we do not need to track such points.
  blowup = ~isfinite(err);
  ms(blowup) = [];
  err(blowup) = [];
  
  %Fit the remaining points to a power law
  p = polyfit( log(ms), log(err), 1 );

  fit = exp(p(2))*ms.^p(1);
end