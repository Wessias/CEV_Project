# %%
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def cev_option_fd(S0, r, K, sigma, delta, X_max, T, M, N, option_type='call'):
    """
    CEV option pricing using a Crank-Nicolson finite difference method.
    
    Solves the PDE
        -u_t + r*x*u_x + (σ²/2)*x^(2δ)*u_xx = 0,  for x ∈ (0, X_max), t ∈ (0, T),
    with initial data
        u(0,x) = max(x-K,0) for a call, or max(K-x,0) for a put,
    and Dirichlet boundary conditions:
      For a call:  u(t,0)=0,   u(t,X_max)= X_max - K * exp(-r*(T-t))
      For a put:   u(t,0)= K * exp(-r*(T-t)),   u(t,X_max)=0
    
    Parameters:
      S0 (float): Initial stock price (for which the option price is returned)
      r (float): Risk-free rate
      K (float): Strike price
      sigma (float): Volatility parameter
      delta (float): Elasticity parameter
      X_max (float): Maximum stock price for the grid
      T (float): Time to maturity
      M (int): Number of spatial steps (grid nodes: 0,...,M)
      N (int): Number of time steps
      option_type (str): 'call' or 'put'
      
    Returns:
      u (np.array): The computed solution grid (time x space)
      x (np.array): The spatial grid
      price (float): Option price (at S0, time T) via linear interpolation
    """
    # Grid parameters
    dx = X_max / M
    dt = T / N
    x = np.linspace(0, X_max, M+1)
    
    # Initialize the solution grid: each row corresponds to a time level.
    u = np.zeros((N+1, M+1))
    if option_type == 'call':
        u[0, :] = np.maximum(x - K, 0)
    else:
        u[0, :] = np.maximum(K - x, 0)
    
    # Build coefficient matrices A and B for interior nodes.
    A = np.zeros((M+1, M+1))
    B = np.zeros((M+1, M+1))
    
    # Loop over interior nodes i = 1,..., M-1.
    for i in range(1, M):
        xi = x[i]
        x2delta = xi**(2*delta)
        # In the Crank-Nicolson scheme, the coefficients for u^{n+1} are:
        #   a[i] = dt/(4*dx)*r*x_i - dt/(4*dx^2)*sigma^2*x_i^(2δ)
        #   b[i] = 1 + dt/(2*dx^2)*sigma^2*x_i^(2δ)
        #   c[i] = - dt/(4*dx)*r*x_i - dt/(4*dx^2)*sigma^2*x_i^(2δ)
        a_coef = (dt/(4*dx)) * r * xi - (dt/(4*dx**2)) * sigma**2 * x2delta
        b_coef = 1 + (dt/(2*dx**2)) * sigma**2 * x2delta
        c_coef = - (dt/(4*dx)) * r * xi - (dt/(4*dx**2)) * sigma**2 * x2delta
        
        A[i, i-1] = a_coef
        A[i, i]   = b_coef
        A[i, i+1] = c_coef
        
        # Off-diagonals have opposite sign and the diagonal becomes:
        B[i, i-1] = -a_coef
        B[i, i]   = 2 - b_coef
        B[i, i+1] = -c_coef

    # Impose boundary conditions in the matrices.
    # For a call:
    #    u(t,0)=0,   u(t,X_max)= X_max - K*exp(-r*(T-t))
    # We force the first and last rows to be identity.
    A[0, :] = 0;   A[0, 0] = 1
    A[M, :] = 0;   A[M, M] = 1
    B[0, :] = 0;   B[0, 0] = 1
    B[M, :] = 0;   B[M, M] = 1

    # Time stepping
    for n in range(N):
        t_current = (n+1)*dt
        # Update the boundary values at time level n for consistency.
        if option_type == 'call':
            u[n, 0] = 0.0
            u[n, M] = X_max - K * np.exp(-r * (T - n*dt))
        else:
            u[n, 0] = K * np.exp(-r * (T - n*dt))
            u[n, M] = 0.0

        # Form the right-hand side using matrix B.
        rhs = B @ u[n, :]
        # Overwrite boundary entries with the time-dependent boundary values.
        if option_type == 'call':
            rhs[0] = 0.0
            rhs[M] = X_max - K * np.exp(-r * (T - t_current))
        else:
            rhs[0] = K * np.exp(-r * (T - t_current))
            rhs[M] = 0.0
        
        # Solve the linear system for u^{n+1}
        u[n+1, :] = np.linalg.solve(A, rhs)
    
    # Enforce the final time boundary conditions
    if option_type == 'call':
        u[-1, 0] = 0.0
        u[-1, M] = X_max - K  # at t=T, exp(-r*0) = 1
    else:
        u[-1, 0] = K
        u[-1, M] = 0.0

        
    # Compute the actual option price Π_Y(τ, x) = e^(-rτ) × u(τ,x)
    # Where τ = T - t 
    τ = np.linspace(0, T, u.shape[0])
    discount_factor = np.exp(-r * τ)
    Π_Y = u * discount_factor[:, np.newaxis]

    # Option price at S0 via linear interpolation.
    price = np.interp(S0, x, Π_Y[-1, :])
    return u, x, price, Π_Y

def black_scholes_price(S, K, r, sigma, T, option='call'):
    """
    Compute Black–Scholes price for European options.
    
    Parameters:
      S (array_like): Asset prices
      K (float): Strike price
      r (float): Risk-free rate
      sigma (float): Volatility
      T (float): Time to maturity
      option (str): 'call' or 'put'
      
    Returns:
      Price (array_like): Option price for each asset price in S.
    """
    if T < 1e-8:
        if option == 'call':
            return np.maximum(S - K, 0)
        else:
            return np.maximum(K - S, 0)

    S = np.array(S, dtype=float)
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option.lower() == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price



def monte_carlo_cev(S0_vals, K, r, sigma, T, delta, n_steps, n_sims, option='call' ):

    """
    Monte Carlo simulation for the CEV model.
    SDE: dS_t = r S_t dt + sigma S_t^delta dW_t.
    
    S0_vals : array of initial stock prices (shape: (n_S,))
    Returns:
       price_estimates : estimated discounted option prices for each initial price.
       std_estimates   : standard error of the estimated prices.
    """
    dt = T / n_steps
    
    # Initialize S array: shape (n_S, n_sims)
    S = np.repeat(S0_vals.reshape(-1, 1), n_sims, axis=1)
    
    # Pre-generate Brownian increments (n_steps x n_sims)
    dW = np.random.normal(0, np.sqrt(dt), size=(n_steps, n_sims))
    
    # Simulate the stock paths
    for j in range(n_steps):
        S = S + r * S * dt + sigma * (S**delta) * dW[j, :]
    
    # Compute terminal payoffs
    if option.lower() == 'call':
        payoff = np.maximum(S - K, 0)
    else:
        payoff = np.maximum(K - S, 0)
    
    disc = np.exp(-r*T)
    # Price estimate: average discounted payoff for each S0
    price_estimates = disc * np.mean(payoff, axis=1)
    # Standard error estimate: sample std divided by sqrt(n_sims)
    std_estimates = disc * np.std(payoff, ddof=1, axis=1) / np.sqrt(n_sims)
    
    return price_estimates, std_estimates

#
#
#
#
#
#
#
#
#
#
# %% ----------------------------------------------------------------
# Plot stuff below
#This block plots the heatmaps as well as FD vs BS when delta=1

# Parameters
S0 = 100       # initial stock price at which we want the option price
r = 0.05      # risk-free rate
K = 100       # strike price
sigma = 0.2   # volatility
delta = 1.0   # elasticity (delta=1 recovers Black-Scholes)
X_max = 200   # maximum stock price on grid
T = 1.0       # time to maturity (years)
M = 500       # number of spatial steps
N = 1000      # number of time steps
option_type = "put" # Option type

# Compute the finite difference solution (for a call)
u_grid, x_grid, option_price_fd, Π_Y = cev_option_fd(S0, r, K, sigma, delta, X_max, T, M, N, option_type)
print("CEV FD Option Price at S0 =", option_price_fd)


# Compute Black-Scholes prices on the same grid (for a call)
bs_prices = black_scholes_price(x_grid, K, r, sigma, T, option_type)

# Plot FD solution at T and Black-Scholes prices
plt.figure(figsize=(8, 6))
#plt.plot(x_grid, u_grid[-1, :], label="CEV Price (before discount)", linewidth=2)
plt.plot(x_grid, bs_prices, label="Black-Scholes Price", linewidth=2, linestyle='--')
plt.plot(x_grid, Π_Y[-1, :],  label='CEV Price')
plt.xlabel("Stock Price, S")
plt.ylabel("Option Price")
plt.title("CEV FD vs. Black-Scholes Price at Maturity (" + option_type + ")")
plt.axvline(K, color='black', linestyle=':', label='Strike Price (K)')
plt.legend()
plt.grid(True)
plt.show()

# Plot the heatmap of u
t = np.linspace(0, T, u_grid.shape[0])

plt.figure(figsize=(8, 6))
plt.pcolormesh(x_grid, t, Π_Y, shading='auto', cmap='viridis')
plt.colorbar(label='Option Price')
plt.xlabel('Stock Price (x)')
plt.ylabel('Time to Maturity (T - t)')
plt.title('CEV Option Price Heatmap (' + option_type + ')')

# Reverse the time axis to show progression from t=0 (maturity) to t=T (today)
plt.gca().invert_yaxis()
plt.show()











#----------------------------------------------------------------
# %% Try to plot call/put price as a function of σ and S0
#Surface plots

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# Fixed parameters
K = 100         # Strike price
r = 0.05        # Risk-free rate
T = 1.0         # Time to maturity
delta = 0.7     # Elasticity parameter 
X_max = 200     # Maximum asset price on grid
M = 200         # Number of spatial steps
N = 1000        # Number of time steps

# Define the grid for initial stock price S0 and volatility sigma
S0_vals = np.linspace(0, X_max, M+1)   
sigma_vals = np.linspace(0.1, 0.8, 30)  

# Create meshgrid for 3D plotting
S0_grid, sigma_grid = np.meshgrid(S0_vals, sigma_vals)

# Storage for option prices
call_prices_fd = np.zeros_like(S0_grid)
put_prices_fd = np.zeros_like(S0_grid)
call_prices_mc = np.zeros_like(S0_grid)
put_prices_mc = np.zeros_like(S0_grid)
call_prices_bs = np.zeros_like(S0_grid)  # Black-Scholes prices
put_prices_bs = np.zeros_like(S0_grid)  # Black-Scholes prices

# ---- Compute Finite Difference Prices for Every Sigma ----
for i in range(len(sigma_vals)):
    sigma = sigma_vals[i]
    
    # Compute new FD solutions for this sigma
    u_grid_call, x_grid, _, Π_call = cev_option_fd(S0_vals[0], r, K, sigma, delta, X_max, T, M, N, option_type="call")
    u_grid_put, _, _, Π_put = cev_option_fd(S0_vals[0], r, K, sigma, delta, X_max, T, M, N, option_type="put")

    # Store the option prices at t = T for all S0 values
    call_prices_fd[i, :] = Π_call[-1, :]  # Last row = final prices for all S0
    put_prices_fd[i, :] = Π_put[-1, :]
    call_prices_bs[i, :] = black_scholes_price(S0_vals, K, r, sigma_vals[i], T, option="call")
    put_prices_bs[i, :] = black_scholes_price(S0_vals, K, r, sigma_vals[i], T, option="put")

    
# %% So I can mess with the plots without re-running every time

# ----------- 3D Plot for Call & Put Prices using FD & BS --------------
fig, axes = plt.subplots(2, 2, figsize=(10, 10), subplot_kw={'projection': '3d'})

# --- Top Left: Call Prices via Finite Difference ---
ax1 = axes[0, 0]
ax1.plot_surface(S0_grid, sigma_grid, call_prices_fd, cmap="spring", edgecolor='k', alpha=0.8)
ax1.set_xlabel("Initial Stock Price $S_0$")
ax1.set_ylabel("Volatility $\sigma$")
ax1.set_zlabel("Call Option Price (FD)")
ax1.set_title("Call Prices via Finite Difference")
ax1.view_init(elev=30, azim=-90-45)

# --- Top Right: Call Prices via Black-Scholes ---
ax2 = axes[0, 1]
ax2.plot_surface(S0_grid, sigma_grid, call_prices_bs, cmap="cool", edgecolor='k', alpha=0.8)
ax2.set_xlabel("Initial Stock Price $S_0$")
ax2.set_ylabel("Volatility $\sigma$")
ax2.set_zlabel("Call Option Price (BS)")
ax2.set_title("Call Prices via Black-Scholes")
ax2.view_init(elev=30, azim=-90-45)

# --- Bottom Left: Put Prices via Finite Difference ---
ax3 = axes[1, 0]
ax3.plot_surface(S0_grid, sigma_grid, put_prices_fd, cmap="spring", edgecolor='k', alpha=0.8)
ax3.set_xlabel("Initial Stock Price $S_0$")
ax3.set_ylabel("Volatility $\sigma$")
ax3.set_zlabel("Put Option Price (FD)")
ax3.set_title("Put Prices via Finite Difference")

# --- Bottom Right: Put Prices via Black-Scholes ---
ax4 = axes[1, 1]
ax4.plot_surface(S0_grid, sigma_grid, put_prices_bs, cmap="cool", edgecolor='k', alpha=0.8)
ax4.set_xlabel("Initial Stock Price $S_0$")
ax4.set_ylabel("Volatility $\sigma$")
ax4.set_zlabel("Put Option Price (BS)")
ax4.set_title("Put Prices via Black-Scholes")

# Adjust layout to reduce empty space
ax2.set_box_aspect(None, zoom=0.8)
ax3.set_box_aspect(None, zoom=0.8)
ax1.set_box_aspect(None, zoom=0.8)
ax4.set_box_aspect(None, zoom=0.8)
plt.show()












# ----------------------------------------------------------------
# %% PLOTS CEV VS BS AND ALSO THE PARITY ERROR

# Fixed parameters
K = 100         # Strike price
r = 0.05        # Risk-free rate
T = 1.0         # Time to maturity
delta = 0.6     # Elasticity parameter 
X_max = 200     # Maximum asset price on grid
M = 500         # Number of spatial steps
N = 1000        # Number of time steps

# We will vary σ (volatility) and also examine option price as function of S.
sigma_vals = [0.1, 0.2, 0.3, 0.4]
S0_vals = np.linspace(10, X_max, 100)  # Range of initial stock prices

# Containers for storing prices for each sigma.
fd_call_prices = {}  # dictionary: sigma -> call price vs S (from FD)
fd_put_prices = {}   # dictionary: sigma -> put price vs S (from FD)
bs_call_prices = {}  # Black-Scholes call prices
bs_put_prices = {}   # Black-Scholes put prices
parity_errors = {}   # Numerical error in put-call parity for each sigma

for sigma in sigma_vals:
    # Compute FD solution for call and put options.
    # The FD solver returns the entire price function.
    u_call, x_grid, _ , Π_call= cev_option_fd(S0=K, r=r, K=K, sigma=sigma, delta=delta, 
                                                X_max=X_max, T=T, M=M, N=N, option_type='call')
    u_put,  x_grid, _, Π_put= cev_option_fd(S0=K, r=r, K=K, sigma=sigma, delta=delta, 
                                                X_max=X_max, T=T, M=M, N=N, option_type='put')
    



    fd_call_prices[sigma] = Π_call[-1, :].copy()  # price at time T (i.e. time-0 price)
    fd_put_prices[sigma]  = Π_put[-1, :].copy()
    
    # For Black-Scholes, use tau = T.
    bs_call_prices[sigma] = black_scholes_price(x_grid, K, r, sigma, T, option='call')
    bs_put_prices[sigma]  = black_scholes_price(x_grid, K, r, sigma, T, option='put')
    
    # Check put-call parity: for European options, we expect
    #    call - put = S - K*exp(-r*T)
    parity = fd_call_prices[sigma] - fd_put_prices[sigma]
    parity_theoretical = x_grid - K * np.exp(-r*T)
    parity_errors[sigma] = np.abs(parity - parity_theoretical)
    print(np.linalg.norm(parity_errors[sigma]))

# Plot FD and BS prices for call options for different sigma values.
plt.figure(figsize=(8, 6))
for sigma in sigma_vals:
    plt.plot(x_grid, fd_call_prices[sigma], label=f"FD Call (σ={sigma})")
    plt.plot(x_grid, bs_call_prices[sigma], '--', label=f"BS Call (σ={sigma})")
plt.xlabel("Initial Stock Price S")
plt.ylabel("Call Option Price")
plt.title("Call Option Price vs. Stock Price for Various σ")
plt.axvline(K, color='black', linestyle=':', label='Strike Price (K)')
plt.legend()
plt.grid(True)
plt.show()

# Plot FD and BS prices for put options for different sigma values.
plt.figure(figsize=(8, 6))
for sigma in sigma_vals:
    plt.plot(x_grid, fd_put_prices[sigma], label=f"FD Put (σ={sigma})")
    plt.plot(x_grid, bs_put_prices[sigma], '--', label=f"BS Put (σ={sigma})")
plt.xlabel("Initial Stock Price S")
plt.ylabel("Put Option Price")
plt.title("Put Option Price vs. Stock Price for Various σ")
plt.axvline(K, color='black', linestyle=':', label='Strike Price (K)')
plt.legend()
plt.grid(True)
plt.show()

# Plot FD and BS prices for call and put options for different sigma values.
plt.figure(figsize=(8, 6))
for sigma in sigma_vals:
    plt.plot(x_grid, fd_call_prices[sigma], label=f"FD Call (σ={sigma})")
    plt.plot(x_grid, bs_call_prices[sigma], '--', label=f"BS Call (σ={sigma})")
    plt.plot(x_grid, fd_put_prices[sigma], label=f"FD Put (σ={sigma})")
    plt.plot(x_grid, bs_put_prices[sigma], '--', label=f"BS Put (σ={sigma})")
plt.xlabel("Initial Stock Price S")
plt.ylabel("Option Price")
plt.title("Option Price vs. Stock Price for Various σ")
plt.axvline(K, color='black', linestyle=':', label='Strike Price (K)')
plt.legend(loc='upper left')
plt.grid(True)
plt.show()

# Plot the put-call parity error for one representative sigma (or for all).
plt.figure(figsize=(8, 6))
for sigma in sigma_vals:
    plt.plot(x_grid, parity_errors[sigma], label=f"Parity error (σ={sigma})")
plt.xlabel("Initial Stock Price S")
plt.ylabel("Absolute Error")
plt.title("Numerical Error in Put-Call Parity (FD)")
plt.axvline(K, color='black', linestyle=':', label='Strike Price (K)')
plt.legend()
plt.grid(True)
plt.show()




# %% Parity error revised


# Fixed parameters
S0 = 150
K = 100         # Strike price
r = 0.05        # Risk-free rate
T = 1.0         # Time to maturity
delta = 0.8     # Elasticity parameter 
X_max = 200     # Maximum asset price on grid
M = 500         # Number of spatial steps
N = 1000        # Number of time steps

# Compute call and put prices
u_call, x_grid, _, Π_call = cev_option_fd(S0, r, K, sigma, delta, X_max, T, M, N, "call")
u_put, _, _, Π_put = cev_option_fd(S0, r, K, sigma, delta, X_max, T, M, N, "put")
t_grid = np.linspace(0, T, N+1)

# Find index of S0 in the grid
index_S0 = np.where(x_grid == S0)[0][0]

# Time grid (converted to t = T - τ)
t_values = T - t_grid  # Convert τ to t

# Compute theoretical put-call parity value
theoretical_diff = S0 - K * np.exp(-r * (T - t_values))

# Compute numerical difference (C - P) at S0
numerical_diff = Π_call[:, index_S0] - Π_put[:, index_S0]

# Compute error
error = numerical_diff - theoretical_diff

print(f"t_values shape: {t_values.shape}")
print(f"numerical_diff shape: {numerical_diff.shape}")
print(f"theoretical_diff shape: {theoretical_diff.shape}")

plt.figure(figsize=(8, 6))
plt.plot(t_values, theoretical_diff, label='Theoretical $S_0 - Ke^{-r(T-t)}$', linestyle='--', color='black')
plt.plot(t_values, numerical_diff, label='Numerical $Π_{call}(t, S_0) - Π_{put}(t, S_0)$', linestyle='-', color='blue')
plt.xlabel('Time $t$')
plt.ylabel('Price Difference')
plt.title('Put-Call Parity Verification (CEV Model)')
plt.legend()
plt.grid()
plt.show()

# Plot error
plt.figure(figsize=(8, 6))
plt.plot(t_values, error, label='Error', color='red')
plt.xlabel('Time $t$')
plt.ylabel('Error')
plt.title('Put-Call Parity Error (Numerical - Theoretical)')
plt.axhline(0, color='black', linestyle=':')
plt.legend()
plt.grid()
plt.show()












#--------------------------------------------------------------------------------
# %%
# Plots FD vs MC in the same plot for different sigmas

# Parameters
S0 = 100       # initial stock price at which we want the option price
r = 0.05      # risk-free rate
K = 100        # strike price
delta = 0.8   # elasticity (delta=1 recovers Black-Scholes)
X_max = 200   # maximum stock price on grid
T = 1.0       # time to maturity (years)
M = 500       # number of spatial steps
N = 1000      # number of time steps
option_type = "call" # Option type
n_sims = 500





S0_vals = np.linspace(0,X_max, M)
sigma_vals = [0.1, 0.2, 0.3, 0.4]


fig, axes = plt.subplots(2, 2, figsize=(10, 10))
# Loop through sigma values and plot each one in a different subplot
for idx, sigma in enumerate(sigma_vals):
    # Compute row and column index for subplot
    i, j = divmod(idx, 2)  # Convert index to row/column

    # Compute FD solution
    u_call, x_grid, _, Π_call = cev_option_fd(S0=K, r=r, K=K, sigma=sigma, delta=delta, 
                                              X_max=X_max, T=T, M=M, N=N, option_type='call')
    
    # Compute Monte Carlo prices
    mc_call_price, _ = monte_carlo_cev(x_grid, K, r, sigma, T, delta, N, n_sims, 'call')

    # Plot on the appropriate subplot
    ax = axes[i, j]
    ax.plot(x_grid, Π_call[-1, :], label=f"FD, σ={sigma}", linestyle="-", linewidth=2)
    ax.plot(x_grid, mc_call_price, label=f"MC, σ={sigma}", linestyle="--", linewidth=2)

    ax.set_xlabel("Initial Stock Price $S_0$")
    ax.set_ylabel("Call Option Price")
    ax.set_title(f"Comparison: FD vs MC (σ={sigma})")
    ax.axvline(K, color='black', linestyle=':', label="Strike Price (K)")
    ax.legend()
    ax.grid(True)

# Adjust layout to prevent overlapping
plt.tight_layout()
plt.show()








#----------------------------------------------------------------
# %% BENCHMARKS FOR FD VS MC
import time  


# Parameters
S0 = 100
r = 0.05
K = 100
delta = 0.8
X_max = 200
T = 1.0
M = 500
N = 1000
option_type = "call"
n_sims = 500

S0_vals = np.linspace(0, X_max, M)
sigma_vals = [0.1, 0.2, 0.3, 0.4]

bench_runs = 10
# Lists to store timing results
fd_times = []
mc_times = []


for i in range(bench_runs):
    print("Run " , i)
    for idx, sigma in enumerate(sigma_vals):

        # Time FD solution
        start_fd = time.time()
        u_call, x_grid, _, Π_call = cev_option_fd(S0=K, r=r, K=K, sigma=sigma, delta=delta,
                                                X_max=X_max, T=T, M=M, N=N, option_type='call')
        end_fd = time.time()
        fd_time = end_fd - start_fd
        fd_times.append(fd_time)

        # Time Monte Carlo solution
        start_mc = time.time()
        mc_call_price, _ = monte_carlo_cev(x_grid, K, r, sigma, T, delta, N, n_sims, 'call')
        end_mc = time.time()
        mc_time = end_mc - start_mc
        mc_times.append(mc_time)



fd_mean_time = np.sum(fd_times) / bench_runs
mc_mean_time = np.sum(mc_times) / bench_runs
# Print timing summary
print("\n=== Timing Summary (Computing price for all S0)===")
print(f"FD = {fd_mean_time:.2f}s | MC = {mc_mean_time:.2f}s")




#
# Benchmark time to compute price for a single S_0
# 
sigma_vals = [0.1, 0.2, 0.3, 0.4]

bench_runs = 10
# Lists to store timing results
fd_times = []
mc_times = []


for i in range(bench_runs):
    print("Run " , i)
    for idx, sigma in enumerate(sigma_vals):

        # Time FD solution
        start_fd = time.time()
        u_call, x_grid, _, Π_call = cev_option_fd(S0=K, r=r, K=K, sigma=sigma, delta=delta,
                                                X_max=X_max, T=T, M=M, N=N, option_type='call')
        end_fd = time.time()
        fd_time = end_fd - start_fd
        fd_times.append(fd_time)

        # Time Monte Carlo solution
        start_mc = time.time()
        mc_call_price, _ = monte_carlo_cev(K*np.ones(1), K, r, sigma, T, delta, N, n_sims, 'call')
        end_mc = time.time()
        mc_time = end_mc - start_mc
        mc_times.append(mc_time)



fd_mean_time = np.sum(fd_times) / bench_runs
mc_mean_time = np.sum(mc_times) / bench_runs
# Print timing summary
print("\n=== Timing Summary (Computing price for one S0)===")
print(f"FD = {fd_mean_time:.2f}s | MC = {mc_mean_time:.2f}s")