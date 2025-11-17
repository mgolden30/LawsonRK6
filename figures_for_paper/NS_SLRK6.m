%{
The goal of this script is to clearly display the order of the sixth order
scheme.
%}

clear;

mkdir("./figures/");

n  = 1024; %number of gridpoints
nu = 1/100; %viscosity

[my_grid] = construct_grid(n);

%Running CPU only is possible, but slow
my_grid = enable_gpu(my_grid);

x = my_grid.x;
y = my_grid.y;

forcing = -4*cos(4*y);

%initial condition
%w = sin(x).*sin(y) + 9*sin(3*x).*sin(3*y-1);
w = 4*sin(2*x) + 3*cos(x + 3*y+0.13) + 2*sin(4*x+2*y+0.31) + 1*sin(5*x + 6*y + 1.23);

%Integrate a transient
t = 5;
m_fine = 8*1024;

w_fine = slrk6(w, nu, forcing, my_grid, t, m_fine);

tiledlayout(1,2);

nexttile; imagesc(w); axis square; clim([-10, 10])
nexttile; imagesc(w_fine); axis square; clim([-10, 10])



%%
ms = round(exp(linspace(log(128), log(1024*2), 32)));
points = numel(ms);

err_4 = zeros(points,1);
err_6 = zeros(points,1);

for i = 1:points
  wf = slrk4(w, nu, forcing, my_grid, t, ms(i) );  
  err_4(i) = max(abs(wf - w_fine), [], "all");

  wf = slrk6(w, nu, forcing, my_grid, t, ms(i) );  
  err_6(i) = max(abs(wf - w_fine), [], "all");
end

%%
clf;

%generate power law fits to the error
indices = [];
[ms_4, err_4b, fit_4, p4] = fit_power_law(ms, err_4, indices );

indices = 13:27;
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
estimated_floor = numel(w) * 1e-16;
yline( estimated_floor, 'linestyle', '-.');
hold off


legend({"", "", "SLRK4", "SLRK6", ""}, "location", "southwest");
set(gca, "xscale", "log");
set(gca, "yscale", "log");

axis square;

set(gca, "fontsize", 32);
xlabel( "timesteps $m$", "Interpreter", "latex");
ylabel( "max absolute error", "Interpreter", "latex");

xlim([256*0.9, 2048*1.1]);
set(gcf, "color", "w");

xticks([256, 512, 1024, 2048]);
yticks(10.^[-12:3:-3]);
ylim([1e-13, 1e-3]);

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


set(groot, 'defaultAxesTickLabelInterpreter','latex');
labels = {"$10^{-3}$", "$10^{-6}$", "$10^{-9}$", "$\varepsilon_{\textrm{floor}}$", "$10^{-12}$"};
yticklabels( flip(labels) );

exportgraphics(gcf,'figures/convergence_NS.png','Resolution',512)
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

function grid = enable_gpu(grid)
  % Get all field names
  fn = fieldnames(grid);

  % Loop through fields
  for k = 1:numel(fn)
    field = fn{k};        % get the field name (string)
    grid.(field) = gpuArray( grid.(field) ); 
  end
end

function [my_grid] = construct_grid(n)
  k = 0:n-1;
  k(k>n/2) = k(k>n/2) - n;

  [x,y] = meshgrid( (0:n-1)/n*2*pi );
  [kx, ky] = meshgrid(k);

  mask = (abs(kx) < n/3) & (abs(ky) < n/3);
  mask(1,1) = 0;

  to_u = 1i*ky./(kx.^2 + ky.^2);
  to_v =-1i*kx./(kx.^2 + ky.^2);
  
  to_u(1,1) = 0;
  to_v(1,1) = 0;
  
  k_sq = kx.^2 + ky.^2;

  my_grid.kx = kx.*mask;
  my_grid.ky = ky.*mask;
  my_grid.to_u = to_u.*mask;
  my_grid.to_v = to_v.*mask;
  my_grid.k_sq = k_sq;
  my_grid.x = x;
  my_grid.y = y;
  my_grid.mask = mask;
end


function w = slrk4(w, nu, forcing, my_grid, t, m)
  w = fft2(w);
  h = t/m;
  e = exp(-nu*my_grid.k_sq*h/2) .* my_grid.mask;
  g = @(w) rhs(w, forcing, my_grid);
  for i = 1:m
    k1 = h*g(w);
    w = e.*w; k1 = e.*k1;
    k2 = h*g(w + k1/2);
    k3 = h*g(w + k2/2);
    w = e.*w; k1 = e.*k1; k2 = e.*k2; k3 = e.*k3;
    k4 = h*g(w + k3);
    w = w + (k1 + 2*k2 + 2*k3 + k4)/6;
  end
  w = real(ifft2(w));
end


function w = slrk6(w, nu, forcing, my_grid, t, m)
  w = fft2(w);
  h = t/m;
  e = exp(-nu*my_grid.k_sq*h/6) .* my_grid.mask;
  g = @(w) rhs(w, forcing, my_grid);
  for i = 1:m
    if mod(i,16) == 0
      i
    end 
    k1 = h*g(w);
    w = e.*w; k1 = e.*k1;
    k2 = h*g(w + k1/6);
    k3 = h*g(w + k1/12 + k2/12 );
    w = e.*w; k1 = e.*k1; k2 = e.*k2; k3 = e.*k3;
    k4 = h*g(w - k2*4/33 + k3*5/11);
    w = e.*w; k1 = e.*k1; k2 = e.*k2; k3 = e.*k3; k4 = e.*k4;
    k5 = h*g(w - k1/4 - k2*29/44 + k3*31/22);
    w = e.*w; k1 = e.*k1; k2 = e.*k2; k3 = e.*k3; k4 = e.*k4; k5 = e.*k5;
    k6 = h*g(w + k1*3/11 + k2*8/33 - k3*4/11 + k4/11 + k5*14/33);
    w = e.*w; k1 = e.*k1; k2 = e.*k2; k3 = e.*k3; k4 = e.*k4; k5 = e.*k5; k6 = e.*k6;
    k7 = h*g(w - k1*17/48 - k2*5/12 + k3 + k4 - k5*13/12 + k6*11/16);
    w = e.*w; k1 = e.*k1; k2 = e.*k2; k3 = e.*k3; k4 = e.*k4; k5 = e.*k5; k6 = e.*k6; k7 = e.*k7;
    k8 = h*g(w + k1*20/39 + k2*12/39 - k3*31/39 - k4/39 + k5*34/39 - k6*11/39 + k7*16/39);

    w = w + 13/200*(k1 + k8) + 4/25*(k3 + k7) + 11/40*(k4 + k6);    
  end
  w = real(ifft2(w));
end

function out = rhs( w, forcing, my_grid )
  wx = real(ifft2( 1i * my_grid.kx .* w ));
  wy = real(ifft2( 1i * my_grid.ky .* w ));
  u  = real(ifft2( my_grid.to_u .* w ));
  v  = real(ifft2( my_grid.to_v .* w ));

  out = -u.*wx -v.*wy + forcing;
  out = my_grid.mask.*fft2(out);
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