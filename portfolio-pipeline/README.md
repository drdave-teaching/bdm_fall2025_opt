# Portfolio Pipeline (Pyomo + yfinance)

Downloads daily prices with `yfinance`, converts to **monthly returns**, and runs a **Pyomo** portfolio optimizer (Ipopt). Plots the efficient frontier and allocations.

## Quickstart

```bash
# 1) Clone
git clone https://github.com/<you>/portfolio-pipeline.git
cd portfolio-pipeline

# 2) Create env and install deps
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3) Get Ipopt (via IDAES)
idaes get-extensions --to ./bin
# Ipopt will be at: ./bin/ipopt

# 4) Run
python main.py --ipopt ./bin/ipopt \
  --start 2022-01-01 --end 2024-01-01 \
  --tickers AES LNT AEE AEP AWK APD ALB AMCR AVY BALL ALL AON CPAY EG IVZ
