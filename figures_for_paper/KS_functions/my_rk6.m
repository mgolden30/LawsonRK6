function [a,b] = my_rk6()
  %{
  This is the RK6 scheme proposed in my manuscript.
  
  This is an 8 stage, 6th order Runge-Kutta method.
  While one more stage than theoretically optimal (7 stages is the known
  minimum), it has equally spaced and ordered c_i, which is convenient for
  many purposes.
  %}
  
  b = [13/200, 0, 4/25, 11/40, 0, 11/40, 4/25, 13/200];

  a = zeros(8,8);
  a(2,1:1) = 1/6;
  a(3,1:2) = [1/12, 1/12];
  a(4,1:3) = [0, -4/33, 5/11];
  a(5,1:3) = [-1/4, -29/44, 31/22];
  a(6,1:5) = [3/11, 8/33, -4/11, 1/11, 14/33];
  a(7,1:6) = [-17/48, -5/12, 1, 1, -13/12, 11/16];
  a(8,1:7) = [20/39, 12/39, -31/39, -1/39, 34/39, -11/39, 16/39];
end