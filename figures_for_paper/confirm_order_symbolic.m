
[a,b] = my_rk6()
f = order_conditions(a,b);

%Print the worst error in the order conditions to the screen
worst_error = max(abs(f));
assert(worst_error == 0);
fprintf("Max(abs(error in order conditions) = %f for my RK6.\n", worst_error);


[a,b] = butcher_rk6()
f = order_conditions(a,b);

%Print the worst error in the order conditions to the screen
worst_error = max(abs(f));
assert(worst_error == 0);
fprintf("Max(abs(error in order conditions) = %f for Butcher's RK6.\n", worst_error);



function f = order_conditions(a,b)
  %{
  All 37 order conditions of a sixth order RK method.
  %}
  c = sum(a,2);  
  b = reshape(b, [], 1);

  f = [
  %first order
  sum(b) - sym(1);
  %second order
  sum(b.*c) - sym(1/2);
  %third order
  sum(b.*c.^2) - sym(1/3);
  sum(b.*a*c) - sym(1/6);
  %fourth order
  sum(b.*c.^3) - sym(1/4);
  sum(b.*a*a*c) - sym(1/24);
  sum(b.*a*c.^2) - sym(1/12);
  sum(b.*c.*a*c) - sym(1/8);
  %fifth order
  sum(b.*a*a*a*c) - sym(1/120);
  sum(b.*a*a*c.^2) - sym(1/60);
  sum(b.*a*(c.*a*c)) - sym(1/40);
  sum(b.*c.*a*a*c) - sym(1/30);
  sum(b.*a*c.^3) - sym(1/20);
  sum(b.*c.*a*c.^2) - sym(1/15);
  sum(b.*(a*c).^2) - sym(1/20);
  sum(b.*c.*c.*a*c) - sym(1/10);
  sum(b.*c.^4) - sym(1/5);
  %sixth order 
  sum(b.*a*a*a*a*c) - sym(1/720);
  sum(b.*a*a*a*c.^2) - sym(1/360);
  sum(b.*a*a*(c.*a*c)) - sym(1/240);
  sum(b.*a*(c.*a*a*c)) - sym(4/720);
  sum(b.*c.*(a*a*a*c)) - sym(5/720);
  
  sum(b.*a*a*c.^3) - sym(6/720);
  sum(b.*a*(c.*a*c.^2)) - sym(8/720);
  sum(b.*c.*a*a*c.^2) - sym(10/720);

  sum(b.*a*(a*c).^2) - sym(6/720);
  sum(b.*a*(c.^2.*a*c)) - sym(12/720);
  sum(b.*c.*a*(c.*a*c)) - sym(15/720);
  
  sum(b.*c.^2.*a*a*c) - sym(20/720);
  sum(b.*(a*c).*(a*a*c)) - sym(10/720);
  
  sum(b.*a*c.^4) - sym(24/720);
  sum(b.*c.*a*c.^3) - sym(30/720);

  sum(b.*(a*c).*a*c.^2) - sym(20/720);
  sum(b.*c.^2.*a*c.^2) - sym(40/720);
  
  sum(b.*(a*c).^2.*c) - sym(30/720);
  
  sum(b.*c.^3.*a*c) - sym(60/720);
  
  sum(b.*c.^5) - sym(1/6);
  ];
end


function [a,b] = my_rk6()
  b = sym( [13/200, 0, 4/25, 11/40, 0, 11/40, 4/25, 13/200] );

  a = sym(zeros(8,8));
  a(2,1:1) = [1/6];
  a(3,1:2) = [1/12, 1/12];
  a(4,1:3) = [0, -4/33, 5/11];
  a(5,1:3) = [-1/4, -29/44, 31/22];
  a(6,1:5) = [3/11, 8/33, -4/11, 1/11, 14/33];
  a(7,1:6) = [-17/48, -5/12, 1, 1, -13/12, 11/16];
  a(8,1:7) = [20/39, 12/39, -31/39, -1/39, 34/39, -11/39, 16/39];
end

function [a,b] = butcher_rk6()
  %One of the sixth order methods provided by J. Butcher in 
  %"On Runge-Kutta Processes of High Order" (1964)
  b = sym( [11/120, 0, 27/40, 27/40, -4/15, -4/15, 11/120]);

  a = sym(zeros(7,7));

  a(2,1:1) = [1/3];
  a(3,1:2) = [0, 2/3];
  a(4,1:3) = [1/12, 1/3, -1/12];
  a(5,1:4) = [-1/16, 9/8, -3/16, -3/8];
  a(6,1:5) = [0, 9/8, -3/8, -3/4, 1/2];
  a(7,1:6) = [9/44, -9/11, 63/44, 18/11, 0, -16/11];
end