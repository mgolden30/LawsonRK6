function [U, expm_times, rk6_times] = Lawson_RK6(a,b,u,t,steps,m,k,mask)
  %{
  Accomplish the same output as SLRK4, but for a generic RK6 I provide
  
  INPUTS
  a - Butcher table coefficients a_{ij}
  b - Butcher table coefficients b_i
  %}
  
  %Allow the use to specify an arbitrary number of stages for the method
  stages = numel(b);

  % Define Butcher table coefficients for RK6
  %Allocate memory for the whole trajectory
  U = zeros(numel(u), steps+1);

  %Save the initial condition
  U(:,1) = u;

  %Define the timestep
  h = t/steps;

  %benchmarking 
  expm_times = zeros(steps, 1);
  rk6_times  = zeros(steps, 1);

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

    %Compute the abscissa
    c = sum(a,2);

    %Assume that the final c is unity
    assert( abs(c(end) - 1) < 1e-9 );

    %Exponentiate the square submatrix
    H_square = H(1:m,1:m);
    expH = zeros(m,m,stages-1);
    for j = 1:stages-1
      expH(:,:,j) = expm(H_square * h * (c(j+1) - c(j)));
    end
    expm_times(i) = toc;

    %project and exponentiate
    e = @(v,j) Q(:,1:m) * expH(:,:,j) * Q(:,1:m)' * v;

    %Macro for nonlinear velocity
    v = @(u) N(u,u0,k,mask);

    %Allocate memory for velocities
    K = zeros(numel(u), stages);
    
    tic
    for j = 1:stages
      u_arg = u;
      if j>1
        %Apply exp( Delta c h A )
        u = e(u,j-1);
        K(:,1:j-1) = e(K(:,1:j-1),j-1);

        %construct u_arg
        u_arg = u + K(:,1:j-1)*a(j,1:j-1)';
      end
      K(:,j) = h*v(u_arg); 
    end
    u = u + sum(K.*reshape(b,1,[]), 2);
    rk6_times(i) = toc;

    %save state
    U(:,i+1) = u;
  end
end