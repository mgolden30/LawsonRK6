function f = N(u,u0,k,mask)
  %{
  The nonlinear part of the time derivative.
  N(u,u0) = -1/2\partial_x(u(u-u0))
  %}
  
  nonlinear = fft( u.*(u-u0));
  nonlinear = 1i*mask.*k.*nonlinear; %spectral derivative

  f = -0.5*real(ifft(nonlinear));
end