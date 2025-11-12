# some useful modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf


tickers_list = ['AES','LNT','AEE','AEP','AWK','APD','ALB','AMCR','AVY','BALL','ALL', 'AON', 'CPAY', 'EG', 'IVZ']

start = '2022-01-01' # Changed to a more recent start date
end = '2024-01-01' # Changed to a more recent end date
# Function to calculate monthly returns
def calculate_monthly_returns(tickers_list, start_date, end_date):
    """
    Downloads daily stock data, calculates daily and monthly returns.

    Args:
        tickers_list (list): A list of stock ticker symbols.
        start_date (str): The start date for data download (YYYY-MM-DD).
        end_date (str): The end date for data download (YYYY-MM-DD).

    Returns:
        pd.DataFrame: DataFrame of monthly returns, or None if no data.
    """

    import sys
    import os

    # if 'google.colab' in sys.modules:
    #     !pip install idaes-pse --pre
    #     !idaes get-extensions --to ./bin
    #     os.environ['PATH'] += ':bin'
    dow_prices = {}
    for t in tickers_list:
        try:
            df = yf.download(t, start=start_date, end=end_date, interval='1d', progress=False, auto_adjust=False)
            if not df.empty:
                dow_prices[t] = df
            else:
                print(f'Warning: no data returned for {t}')
        except Exception as e:
            print(f'Failed {t}: {e}')

    if not dow_prices:
        print("No stock data was downloaded. Please check the ticker symbols and date range.")
        return None
    else:
        return_data_dict = {}
        for ticker, data in dow_prices.items():
            if not data.empty:
                returns = data['Close'].pct_change().dropna()
                if len(returns) > 1:
                    return_data_dict[ticker] = returns

        if not return_data_dict:
            print("No valid stock data available after calculating daily returns.")
            return None
        else:
            daily_returns = pd.concat(return_data_dict.values(), axis=1, keys=return_data_dict.keys())
            # Resample to monthly returns - taking the sum of daily returns within each month
            monthly_returns = (1 + daily_returns).resample('ME').prod() - 1
            return monthly_returns.dropna() # Drop any months with no data for any ticker

def optimize_and_plot_portfolio(df_returns, ipopt_executable):
    from pyomo.environ import ConcreteModel, Set, Var, NonNegativeReals, Param, Objective, maximize, Constraint
    from pyomo.opt import SolverFactory, TerminationCondition
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    # Initialize model
    m = ConcreteModel()

    # Asset list
    assets = df_returns.columns.tolist()
    m.Assets = Set(initialize=assets)

    # Define decision variables for each asset
    m.x = Var(m.Assets, within=NonNegativeReals, bounds=(0,1))

    # Calculate average returns per asset and create a Pyomo Param
    avg_returns = df_returns.mean().to_dict()
    m.mu = Param(m.Assets, initialize=avg_returns)

    # Covariance matrix (Sigma) and create a Pyomo Param
    cov_df = df_returns.cov()
    cov_dict = {(i, j): cov_df.loc[i, j] for i in assets for j in assets}
    m.Sigma = Param(m.Assets, m.Assets, initialize=cov_dict)

    # Objective: Maximize expected return
    def total_return_rule(m):
        return sum(m.mu[a] * m.x[a] for a in m.Assets)
    m.objective = Objective(rule=total_return_rule, sense=maximize)

    # Constraint: Sum of allocations must be 1
    def budget_constraint_rule(m):
        return sum(m.x[a] for a in m.Assets) == 1
    m.budget = Constraint(rule=budget_constraint_rule)

    # Remove the dummy total_risk constraint if it exists (it will be replaced dynamically later)
    if hasattr(m, 'total_risk'):
        m.del_component(m.total_risk)

    print("Pyomo model initialized with sets, variables, parameters, objective, and budget constraint.")

    # Define solver
    solver = SolverFactory("ipopt", executable=ipopt_executable)

    # Determine maximum risk value for plotting purposes if needed
    # For this dataset, a reasonable maximum risk value needs to be determined.
    # One approach is to calculate the variance of an equally weighted portfolio or max individual asset variance.
    # Or, as suggested in the instructions, the max value in the weighted cov matrix where p=1. Let's use max(cov_df.values.flatten())
    # Or, as was done before, simply use the original max_risk value as a starting point, but adjust if necessary.
    # Let's try to infer a reasonable max_risk from the data.
    # A simple heuristic could be the max variance of any single asset, or the max element in the cov matrix.
    # Let's use the max individual asset variance as a conservative upper bound for risk_limits, multiplied by a factor
    # to explore beyond minimal risk.
    max_possible_variance = np.max(np.diag(cov_df.values))
    # Adjusted max_risk to be slightly above the max individual variance or a broader range.
    # This part can be tuned based on the specific dataset characteristics to ensure feasibility.
    # Let's try to infer a reasonable max_risk from the data.
    # A simple heuristic could be the max variance of any single asset, or the max element in the cov matrix.
    # Let's use the max individual asset variance as a conservative upper bound for risk_limits, multiplied by a factor
    # to explore beyond minimal risk.
    max_risk_for_range = np.max(np.diag(cov_df.values)) * 1.5 # Scale up a bit to see more of the frontier
    # Ensure min_risk_for_range is not 0 to avoid division by zero or empty range for arange
    min_risk_for_range = 1e-6 # Start from a very small positive risk

    # Create risk limits array - if max_risk_for_range is too small, np.arange might return empty array
    if max_risk_for_range > min_risk_for_range:
        risk_limits = np.arange(min_risk_for_range, max_risk_for_range + 1e-6, (max_risk_for_range - min_risk_for_range) / 200)
    else:
        # Fallback for very low variance data or specific scenarios
        risk_limits = np.array([min_risk_for_range])


    # Result storage
    param_analysis = {}
    returns = {}

    print(f"Starting portfolio optimization for {len(risk_limits)} risk levels...")
    for i, r in enumerate(risk_limits):
        # Remove old variance constraint if it exists
        if hasattr(m, 'variance_constraint'):
            m.del_component(m.variance_constraint)

        # Add new variance constraint for this risk level
        def variance_constraint_rule(m):
            return sum(m.Sigma[i, j] * m.x[i] * m.x[j] for i in m.Assets for j in m.Assets) <= r
        m.variance_constraint = Constraint(rule=variance_constraint_rule)

        # Solve
        result = solver.solve(m)

        # Skip infeasible solutions
        if result.solver.termination_condition == TerminationCondition.infeasible or \
           result.solver.termination_condition == TerminationCondition.other:
            print(f"Warning: Model infeasible for risk level {r:.6f}. Skipping.")
            continue

        # Check if the solution is optimal or locally optimal
        if result.solver.termination_condition == TerminationCondition.optimal or \
           result.solver.termination_condition == TerminationCondition.locallyOptimal:
            # Save allocations and returns
            param_analysis[r] = [m.x[a]() for a in m.Assets]
            returns[r] = m.objective()
        else:
            print(f"Warning: Solver terminated with condition {result.solver.termination_condition} for risk level {r:.6f}. Skipping.")


    # Create DataFrame for plotting
    df_results = pd.DataFrame({
        'Risk': list(returns.keys()),
        'Return': list(returns.values())
    })

    # Sort by Risk (just in case)
    df_results = df_results.sort_values(by='Risk')

    # Plot Efficient Frontier
    plt.figure(figsize=(10,6))
    plt.plot(df_results['Risk'], df_results['Return'], marker='o', linestyle='-')
    plt.title("Efficient Frontier")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Expected Return")
    plt.grid(True)
    plt.show()

    # Convert allocation results to DataFrame for plotting
    df_allocations = pd.DataFrame(param_analysis).T  # rows = risk, columns = assets
    df_allocations.columns = assets
    df_allocations['Risk'] = df_allocations.index

    # Plot asset allocation proportions by asset
    plt.figure(figsize=(12, 6))
    for asset in assets:
        plt.plot(df_allocations['Risk'], df_allocations[asset], label=str(asset), marker='o', markersize=4)

    plt.title("Asset Allocation as a Function of Portfolio Risk")
    plt.xlabel("Portfolio Risk (Variance)")
    plt.ylabel("Proportion Invested")
    plt.legend(title="Asset", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("Portfolio optimization and plotting complete.")
    return df_results, df_allocations

print("Finished defining `optimize_and_plot_portfolio` function.")

# ============================
# Glue: download → optimize
# ============================

# Assumes your two functions are already defined in the same session/file:
# - calculate_monthly_returns(tickers_list, start_date, end_date)
# - optimize_and_plot_portfolio(df_returns, ipopt_executable)

tickers_list = ['GE', 'KO', "NVDA"]
start = '2020-01-01'
end = '2024-01-01'

def run_portfolio_pipeline(
    ipopt_executable: str,
    tickers=tickers_list,
    start_date=start,
    end_date=end,
    min_months_required: int = 6
):
    """
    1) Download daily prices and compute monthly returns
    2) Run your Pyomo optimizer + plots
    Returns: (monthly_returns, df_frontier, df_allocations)
    """
    monthly_returns = calculate_monthly_returns(tickers, start_date, end_date)
    if monthly_returns is None or monthly_returns.empty:
        raise RuntimeError("No monthly returns were produced. Check tickers/date range.")

    # Basic sanity check: ensure we have enough months to be meaningful
    if monthly_returns.shape[0] < min_months_required:
        print(f"Warning: only {monthly_returns.shape[0]} monthly observations "
              f"(min recommended = {min_months_required}). Proceeding anyway.")

    df_frontier, df_allocations = optimize_and_plot_portfolio(monthly_returns, ipopt_executable)
    return monthly_returns, df_frontier, df_allocations


