%{
Linear stability analysis of my RK6 method
%}


clear;
clf;

[a,b] = my_rk6();
phi_rk6 = stability_polynomial(a,b)

[a,b] = rk4();
phi_rk4 = stability_polynomial(a,b)

[a,b] = rk1();
phi_rk1 = stability_polynomial(a,b)


%% Make diagram for usual linear stability

tl = tiledlayout(1,2);
ax1 = nexttile;

x_grid = linspace(-4.5, 4.5, 512);
y_grid = linspace(-4.5, 4.5, 512);
[X,Y] = meshgrid( x_grid, y_grid );


%colors = {"red", "blue", "green"};
colors = {"#e4572e", "#17bebb", "#ffc914"};
draw_stability_curve(phi_rk1, X, Y, '-.', 0, colors{1});
hold on
draw_stability_curve(phi_rk4, X, Y, '--', 0, colors{2});
draw_stability_curve(phi_rk6, X, Y, '-', 0, colors{3});
hold off
legend({"RK1", "RK4", "RK6"}, "interpreter", "latex", "location", "southeast");
xlabel("Re($\lambda$)", "interpreter", "latex");
ylabel("Im($\lambda$)", "interpreter", "latex");
xticks([-6:2:6]);
%xlim([-5, 0.4]);
yticks(-4:2:4);
%xticks([]);
set(gcf, "color", "w");


side_length = @(x) max(x) - min(x);
sx = side_length(xlim);
sy = side_length(ylim);
pbaspect([side_length(xlim),side_length(ylim),1])


fs = 24;
set(gca, "fontsize", fs);
ax = gca;
ax.TickLabelInterpreter = 'latex';

%exportgraphics(gcf, 'figures/stability.png', 'Resolution', 600);


%% Do linear stability analsys with a specific eigenvalue handled implicitly
ax2 = nexttile;

x_grid = linspace(-30, 30, 512);
y_grid = linspace(-30, 30, 512);
[X,Y] = meshgrid( x_grid, y_grid );

z2 = -10;

draw_stability_curve(phi_rk1, X, Y, '-.', z2, colors{1});
hold on
draw_stability_curve(phi_rk4, X, Y, '--', z2, colors{2});
draw_stability_curve(phi_rk6, X, Y, '-', z2, colors{3});

draw_stability_curve(phi_rk1, X, Y, '-', 0, 'k');
draw_stability_curve(phi_rk4, X, Y, '-', 0, 'k');
draw_stability_curve(phi_rk6, X, Y, '-', 0, 'k');

pos = [-4.5,-4.5,9,9];
rectangle('Position', pos, "LineWidth", 1) 
hold off
text(14, 26, "$z_2 = -10$", "Interpreter", "latex", "fontsize", fs);
legend({"", "SLRK4", "SLRK6"}, "interpreter", "latex", "location", "southwest");
xlabel("Re($\lambda$)", "interpreter", "latex");
ylabel("Im($\lambda$)", "interpreter", "latex");
%yticks(-4:2:4);
%xticks([]);
set(gcf, "color", "w");

sx = side_length(xlim);
sy = side_length(ylim);
pbaspect([side_length(xlim),side_length(ylim),1])

set(gca, "fontsize", fs);

xticks([-30, 0, 30]);
yticks([-30, 0, 30]);
ax = gca;
ax.TickLabelInterpreter = 'latex';



exportgraphics(gcf, 'figures/stability.png', 'Resolution', 600);


function draw_stability_curve(phi, X, Y, style, z2, color)
  Z = X + 1i*Y;
  P = polyval( fliplr(double(coeffs(phi))), Z );

  P = P .* exp(z2);

  C = 1; %draw the line where absolute value = 1
  M = abs(P);

  lw = 3;
  contour(X, Y, M, [C C], 'LineWidth', lw, 'Color', color, 'linestyle', style);
end

function phi = stability_polynomial(a,b)
  syms z;
  n = numel(b);
  k = sym(zeros(n,1));
  g = @(x) z*x;

  for i = 1:n
    x = sym(1);
    for j = 1:i-1
      x = x + a(i,j)*k(j);
    end
    x = simplify(x);
    k(i) = g(x);
  end

  b = reshape(b, size(k));
  phi = 1 + sum(b.*k);
  phi = simplify(expand(phi));
end


function phi = lawson_stability_polynomial(a,b)
  syms z A;

  c = sum(a,2);
  n = numel(b);
  k = sym(zeros(n,1));
  g = @(x) z*x;

  for i = 1:n
    x = exp(c(i) * A) * sym(1);
    for j = 1:i-1
      x = x + exp( (c(i) - c(j)) * A) * a(i,j)*k(j);
    end
    x = simplify(x);
    k(i) = g(x);
  end

  for i = 1:n
    k(i) = exp( (1-c(i))*A ) * k(i);
  end

  b = reshape(b, size(k));
  phi = exp(A) + sum(b.*k);
  phi = simplify(expand(phi));
end


function [a,b] = my_rk6()
  b = sym( [13/200, 0, 4/25, 11/40, 0, 11/40, 4/25, 13/200] );

  a = sym(zeros(8,8));
  a(2,1) = [1/6];
  a(3,1:2) = [1/12, 1/12];
  a(4,1:3) = [0, -4/33, 5/11];
  a(5,1:3) = [-1/4, -29/44, 31/22];
  a(6,1:5) = [3/11, 8/33, -4/11, 1/11, 14/33];
  a(7,1:6) = [-17/48, -5/12, 1, 1, -13/12, 11/16];
  a(8,1:7) = [20/39, 12/39, -31/39, -1/39, 34/39, -11/39, 16/39];
end


function [a,b] = rk4()
  b = sym( [1/6, 1/3, 1/3, 1/6] );

  a = sym(zeros(4,4));
  a(2,1) = [1/2];
  a(3,2) = [1/2];
  a(4,3) = [1];
end

function [a,b] = rk1()
  b = sym( [1] );

  a = sym(zeros(1,1));
end
