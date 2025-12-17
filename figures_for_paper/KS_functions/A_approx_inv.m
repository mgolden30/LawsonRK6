function Au = A_approx_inv(u,k)
  %{
  Approximate inverse of A.
  %}
  Au = fft(u)./(k.^2 - k.^4);
  Au(~isfinite(Au)) = 0;
  Au = real(ifft(Au));
end