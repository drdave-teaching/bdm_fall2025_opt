import argparse
from src.portfolio_pipeline import run_portfolio_pipeline

# defaults (edit if you like)
TICKERS = ['AES','LNT','AEE','AEP','AWK','APD','ALB','AMCR','AVY','BALL','ALL','AON','CPAY','EG','IVZ']
START = '2022-01-01'
END   = '2024-01-01'

def parse_args():
    p = argparse.ArgumentParser(description="Download monthly returns and optimize a portfolio.")
    p.add_argument("--ipopt", required=True, help="Path to Ipopt executable (e.g., ./bin/ipopt)")
    p.add_argument("--start", default=START, help="Start date YYYY-MM-DD")
    p.add_argument("--end",   default=END,   help="End date YYYY-MM-DD")
    p.add_argument("--tickers", nargs="*", default=TICKERS, help="List of tickers")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    mret, frontier, allocs = run_portfolio_pipeline(
        ipopt_executable=args.ipopt,
        tickers=args.tickers,
        start_date=args.start,
        end_date=args.end,
    )
    print("\nMonthly returns:", mret.shape)
    print(frontier.head())
    print(allocs.head())
