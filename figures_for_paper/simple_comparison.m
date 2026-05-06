%{
Compare several RK6 methods.
%}

clear;

a = 0.2;
b = 0.2;
c = 5.7;
v_fn = @(x) [-x(2,:)-x(3,:); x(1,:) + a*x(2,:); b + x(3,:).*(x(1,:)-c)];

addpath("rk_methods/");
%List of methods to compare
methods = [
    "butcher_RK6a";
    "butcher_RK6b";
    "butcher_RK6c";
    "butcher_RK6d";
    "butcher_RK6e";
    "luther_RK6";
    "lawson_RK6ES";
    "huta_RK6";
    "fehlberg_RK56";
    "fehlberg_RK67";
    "prince_RK65";
    "verner_RK659b";
    "RK6_uniform"];
names = [
    "RK6a (Butcher 1964)";
    "RK6b (Butcher 1964)";
    "RK6c (Butcher 1964)";
    "RK6d (Butcher 1964)";
    "RK6e (Butcher 1964)";
    "RK6 (Luther 1968)";
    "RK6ES (Lawson 1967)";
    "RK6 (Huta 1956)";
    "RK5(6)* (Fehlberg 1968)";
    "RK6(7)* (Fehlberg 1968)";
    "RK6(5)* (Prince & Dormand 1981)";
    "RK6(5)9b*  (Verner 1991)"
    "RK6-uniform (this manuscript)"];

rng(123); %seed random number generation
trials = 128;
x0 = 2*randn([3, trials]);
%x0 = [1;2;3]
t = 5;
ns = 2.^[5:14];
n_fine = 2^15;

[a,b] = eval(methods(1) + "()");

x_fine = RK(a,b,x0,v_fn,t,n_fine);

err = zeros( numel(methods), numel(ns) );
for i = 1:numel(ns)
  for j = 1:numel(methods)
    [a,b] = eval(methods(j) + "()");
    x = RK(a,b,x0,v_fn,t,ns(i));
    err(j,i) = norm(x - x_fine);
  end
end



for j = 1:numel(methods)
  [a,b] = eval(methods(j) + "()");
  c = sum(a,2);

  diff = norm(c-  sort(c));
  if diff < 1e-3
    fprintf(names(j) + " is Lawson viable\n");
  end
end

%%
clf;
ms = 100;

colors = [
    0.40 0.40 0.40;   % gray
    0.65 0.30 0.30;   % red-brown
    0.30 0.45 0.65;   % blue
    0.35 0.60 0.40;   % green
    0.60 0.55 0.30;   % olive
    0.50 0.35 0.65;   % purple
    0.70 0.45 0.45;   % dusty rose
    0.30 0.55 0.60;   % cyan

    0.55 0.55 0.55;   % lighter gray
    0.70 0.40 0.25;   % warm brown-orange
    0.25 0.55 0.70;   % brighter blue
    0.25 0.65 0.45;   % brighter green
    0.70 0.65 0.35;   % brighter olive
    0.60 0.45 0.75;   % brighter purple
    0.75 0.55 0.55;   % light rose
    0.25 0.60 0.65    % brighter cyan
];
highlight = [0.95 0.65 0.00]; % strong orange
highlight = [0.00 0.45 0.85]; % vivid blue
highlight = [0.90 0.10 0.10]; % bright red

rng(1);
for i = 1:(numel(methods)-1)
  x_jittered = ns .* (1 + 0.02*randn(size(ns)));
  scatter(x_jittered, err(i,:), ms, 'o', 'filled', 'MarkerFaceColor', colors(i,:));
  hold on
end
scatter(ns, err(end,:), 2*ms, 'p', 'filled', 'MarkerFaceColor', highlight);

hold off
set(gca, 'xscale', 'log');
set(gca, 'yscale', 'log');
legend(names);
box on;
xlabel("number of timesteps $m$", "interpreter", "latex");
ylabel("$L_2$ error", "interpreter", "latex");

%xticks(ns);
xlim([ns(1)/sqrt(2), ns(end)*sqrt(2)]);

estimate_order(ns, err, methods);

function x = RK(a,b,x,v_fn,t,n)
  h = t/n;
  k = zeros(numel(x), numel(b));
  fl = @(y) reshape(y, [], 1);
  uf = @(y) reshape(y, size(x));
  for i = 1:n
    k(:,1) = h * fl(v_fn(x));
    for j = 2:numel(b)
      k(:,j) = h * fl(v_fn(x + uf(k(:, 1:j-1) * a(j,1:j-1)') ));
    end
    x = x + uf(k * reshape(b, [], 1));
  end
end

function estimate_order(ns, err, methods)
  for i = 1:numel(methods)
    p = polyfit( log(ns), log(err(i,:)), 1 );
    fprintf("Method %s is approximately order %.3f\n", methods(i), p(1));
  end
end