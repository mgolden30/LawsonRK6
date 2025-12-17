function [a,b] = fehlberg_RK6()
  %{
  Return the coefficients for an eight stage, sixth order RK method
  from "CLASSICAL FIFTH-, SIXTH-, SEVENTH-, and EIGHTH-ORDER RUNGE-KUTTA
  FORMULAS WITH STEPSIZE CONTROL" by Erwin Fehlberg (1968)

  From TABLE VIII. RK6(7).
  %}

  a = zeros(8,8);
  a(2,1)   = 2/33;
  a(3,1:2) = [0, 4/33];
  a(4,1:3) = [1/22, 0, 3/22];
  a(5,1:4) = [43/64, 0, -165/64, 77/32];
  a(6,1:5) = [-2383/486, 0, 1067/54, -26312/1701, 2176/1701];
  a(7,1:6) = [10077/4802, 0, -5643/686, 116259/16807, -6240/16807, 1053/2401];
  a(8,1:7) = [-733/176, 0, 141/8, -335763/23296, 216/77, -4617/2816, 7203/9152];
  
  b = [77/1440, 0, 0, 1771561/6289920, 32/105, 243/2560, 16807/74880, 11/270];
end