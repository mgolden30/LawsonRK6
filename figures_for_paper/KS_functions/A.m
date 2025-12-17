function Au = A(u,u0,k,mask)
  %{
  The linear part of the time derivative
  %}
  Au = -0.5*1i*k.*mask.*fft(u .* u0);
  Au = Au + (k.^2 - k.^4).*fft(u);
  Au = real(ifft(Au));
end
