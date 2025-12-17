function [Q, H] = generate_krylov(A, x, m)
  %{
  Generate a Krylov subspace
  %}

  H = zeros(m+1,m);
  Q = zeros(numel(x), m);

  Q(:,1) = x / norm(x);
  for j = 1:m
    Aq = A(Q(:,j));
    for i = 1:m
      H(i,j) = dot(Aq, Q(:,i));
      Aq = Aq - H(i,j)*Q(:,i);
    end
    H(j+1,j) = norm(Aq);
    Q(:,j+1) = Aq / H(j+1,j);
  end
end