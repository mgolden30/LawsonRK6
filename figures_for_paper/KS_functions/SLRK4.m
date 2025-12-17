function [U, expm_times, rk4_times] = SLRK4(u,t,steps,m,k,mask)
  %{
  Do Lawson integration with RK4, abusing the fact that we need only one
  matrix exponential.
  %}
  
  %Allocate memory for the whole trajectory
  U = zeros(numel(u), steps+1);

  %Save the initial condition
  U(:,1) = u;

  %Define the timestep
  h = t/steps;

  %benchmarking 
  expm_times = zeros(steps, 1);
  rk4_times  = zeros(steps, 1);

  for i = 1:steps
    %Fix the background to the current state
    u0 = u;

    %Define linear operators.
    % A is the linear operator in the dynamics
    % B is the approximate inverse of A, and we will use it to generate the
    %   Krylov subspace
    A_fn = @(v) A(v,u0,k,mask);
    B_fn = @(v) A_approx_inv(v,k);

    tic
    %Use a random vector to generate the Krylov subspace
    x0 = u / mean(abs(u)) + randn(size(u));
    [Q, ~] = generate_krylov(B_fn, x0, m);

    %Change basis of A into Krylov subspace of B
    H = Q'*A_fn(Q);

    %Exponentiate the square submatrix
    H_square = H(1:m,1:m);
    expH = expm(H_square * h/2);
    expm_times(i) = toc;

    %project and exponentiate
    e = @(v) Q(:,1:m) * expH * Q(:,1:m)' * v;

    %Macro for nonlinear velocity
    v = @(u) N(u,u0,k,mask);

    tic
    %Now do SLRK4
    k1 = h*v(u);
    
    u = e(u);
    k1= e(k1);

    k2 = h*v(u + k1/2);
    k3 = h*v(u + k2/2);

    u = e(u);
    k1= e(k1);
    k2= e(k2);
    k3= e(k3);

    k4 = h*v(u + k3);
    u = u + (k1 + 2*k2 + 2*k3 + k4) / 6;
    
    rk4_times(i) = toc;

    %save state
    U(:,i+1) = u;
  end
end