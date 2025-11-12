# 1) Clone & install
* !git clone https://github.com/drdave-teaching/bdm_fall2025_opt.git
* %cd bdm_fall2025_opt/portfolio-pipeline
* !python -m pip install -r requirements.txt

# 2) Get Ipopt via IDAES (puts ipopt at ./bin/ipopt)
* !python -m idaes get-extensions --to ./bin

# 3) Run with your own tickers
* TICKERS = "AAPL MSFT NVDA AMZN GOOGL"
* !python main.py --ipopt ./bin/ipopt --start 2022-01-01 --end 2024-01-01 --tickers {TICKERS}
