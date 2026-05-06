%{
The goal of this script is to clearly display the order of the sixth order
scheme on the van der Pol oscillator
%}

clear;

%mkdir("./figures/");
x = [2; 0];
mu = 100;

%implicit part of dynamics
A = [0, 1; -1, mu];
g = @(x) [0; -mu*x(1)^2*x(2)];

A_fn = @(x0) [0, 1; -2*mu*x0(1)*x0(2) - 1, mu*(1-x0(1)^2)];
g_fn = @(x0) @(x) [0; -mu*(x(1)^2 - x0(1)^2)*x(2) + 2*mu*x0(1)*x0(2)*x(1)];

%{
r = -10;
diss = @(x) -r;%min( [r, mu*(1-x(1)^2)] );
A_fn = @(x0) [0, 1; -1, diss(x0)];
g_fn = @(x0) @(x) [0; mu*(1-x(1)^2)*x(2) - diss(x0)*x(2) ];
%}

t = 200;
m_fine = 64 * 1024;
xs_fine = slrk6_v2(x, A_fn, g_fn, t, m_fine);

t = 200;
m_ultrafine = 640 * 1024;
xs_ultrafine = slrk6_v2(x, A_fn, g_fn, t, m_ultrafine);

%%
clf;
plot(xs_ultrafine(1,:), xs_ultrafine(2,:), 'color', [1,1,1]*0.6, 'linewidth', 3);
hold on
scatter(xs_fine(1,:), xs_fine(2,:), 's', ...
    'filled', ...
   'MarkerFaceColor','red');
hold off
xlabel("$y_1$", "Interpreter", "latex");
ylabel("$y_2$", "Interpreter", "latex", "rotation", 0);
xlim([-3 3]);
ylim([-150, 150]);
axis square;

%%
ms = round(exp(linspace(log(1024), log(32*1024), 32)));
points = numel(ms);

err_4 = zeros(points,1);
err_6 = zeros(points,1);
err_4 = zeros(points,1);
err_4 = zeros(points,1);

for i = 1:points
  xs = slrk4_v2(x, A_fn, g_fn, t, ms(i) );  
  err_4(i) = max(abs(xs(:,end) - xs_fine(:,end)), [], "all");

  xs = slrk6_v2(x, A_fn, g_fn, t, ms(i) );  
  err_6(i) = max(abs(xs(:,end) - xs_fine(:,end)), [], "all");
end

%%
clf;

%generate power law fits to the error
indices = [16:32];
[ms_4, err_4b, fit_4, p4] = fit_power_law(ms, err_4, indices );

indices = [16:32];
[ms_6, err_6b, fit_6, p6] = fit_power_law(ms, err_6, indices );


marker_size = 100;
plot([], []);
hold on
%visulize all data points, regardless of "indices" used to fit the power
%law
plot( ms_4, fit_4, "-", "color", "black", "LineWidth", 2);
plot( ms_6, fit_6, "-", "color", "black", "LineWidth", 2 );
scatter(ms, err_4, marker_size, 's', 'filled', "markerfacecolor", "black");
scatter(ms, err_6, marker_size, 'd', "markeredgecolor", "black", "markerfacecolor", "w", "LineWidth", 2);
%estimated_floor = numel(w) * 1e-16;
%yline( estimated_floor, 'linestyle', '-.');
hold off

legend({"", "", "SLRK4", "SLRK6", ""}, "location", "southwest");
set(gca, "xscale", "log");
set(gca, "yscale", "log");

axis square;

set(gca, "fontsize", 32);
xlabel( "timesteps $m$", "Interpreter", "latex");
ylabel( "max absolute error", "Interpreter", "latex");

%xlim([256*0.9, 2048*1.1]);
set(gcf, "color", "w");

%xticks([256, 512, 1024, 2048]);
%yticks(10.^[-12:3:-3]);
%ylim([1e-13, 1e-3]);

box on
str = sprintf( "$ m^{%.2f}$", p4(1) );
tx = 512+100;
ty = 1e-4;
text(tx, ty, str, 'FontSize', 32, 'Interpreter', 'latex');

str = sprintf( "$ m^{%.2f}$", p6(1) );
tx = 512 + 400;
ty = 5e-9;
text(tx, ty, str, 'FontSize', 32, 'Interpreter', 'latex');

%Add a ylabel at the estimated noise floor
yt = yticks(); %get current ticks
yt(end+1) = estimated_floor;  %Append the noise floor
yt = sort( yt );
yticks(yt);
return

set(groot, 'defaultAxesTickLabelInterpreter','latex');
labels = {"$10^{-3}$", "$10^{-6}$", "$10^{-9}$", "$\varepsilon_{\textrm{floor}}$", "$10^{-12}$"};
yticklabels( flip(labels) );

%exportgraphics(gcf,'figures/convergence_NS.png','Resolution',512)
%saveas(gcf, "figures/convergence_NS.png");

%%

fields = {w, w_fine};
for i = 1:2
  clf;
  x = (0:n-1)/n*2*pi;
  imagesc(x,x,fields{i});
  axis square;
  set(gca, 'ydir', 'normal');
  set(gcf, 'color', 'w');
  clim([-1 1]*10);
  xticks([0, 2*pi*(n-1)/n]);
  xticklabels({"0", "2\pi"});
  yticks(xticks());
  yticklabels({"0", "2\pi"});
  cb = colorbar();
  set(cb, "xtick", [-10, 0, 10]);
  xlabel("$x$", "interpreter", "latex")
  ylabel("$y$", "interpreter", "latex", "rotation", 0)
  set(gca, "fontsize", 32);
  colormap jet;
  drawnow

  name= "figures/NS" + i + ".png";
  exportgraphics(gcf,name,'Resolution',300)
end


function xs = slrk4(x, A, g, t, m)
  h = t/m;
  e = expm(A*h/2);
  xs = zeros(2,m+1);
  xs(:,1) = x; 
  for i = 1:m
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

function xs = slrk6(x, A, g, t, m)
  h = t/m;
  e = expm(A*h/6);
  xs = zeros(2,m+1);
  xs(:,1) = x;
  for i = 1:m
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