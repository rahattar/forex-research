# Rebuilding the data (raw files are not in the repo)

Run from the project root. Needs Node (npx), Python with yfinance, and curl.

FX, 2019-2026 (download/):
    for p in eurusd gbpusd usdjpy; do npx dukascopy-node -i $p -from 2019-01-01 -to 2026-11-01 -t m15 -f csv; done

FX, fresh set (fresh/download/):
    mkdir -p fresh && cd fresh
    for p in eurusd gbpusd usdjpy; do npx dukascopy-node -i $p -from 2010-01-01 -to 2019-01-01 -t m15 -f csv; done
    for p in audusd usdcad usdchf nzdusd eurjpy gbpjpy eurgbp; do npx dukascopy-node -i $p -from 2010-01-01 -to 2025-11-01 -t m15 -f csv; done

Rates (data/fred/), lowercase file names:
    for s in DFF ECBDFR IUDSOIA DTB3 IR3TIB01USM156N IR3TIB01EZM156N IR3TIB01GBM156N IR3TIB01JPM156N IR3TIB01AUM156N IR3TIB01CAM156N IR3TIB01CHM156N IR3TIB01NZM156N; do curl -sL -o "data/fred/$(echo $s | tr A-Z a-z).csv" "https://fred.stlouisfed.org/graph/fredgraph.csv?id=$s"; done
    boj_policy_rate.csv is included in the repo (hand-built step table).

ETFs (data/etf/):
    python - <<'EOF'
    import yfinance as yf
    for t in ["SPY","IWM","EFA","EEM","IEF","TLT","LQD","HYG","GLD","DBC","VNQ","FXE","FXY","UUP"]:
        yf.Ticker(t).history(start="2000-01-01", end="2025-11-01", auto_adjust=False).to_csv(f"data/etf/{t.lower()}.csv")
    EOF

Locked test period: 2025-11-01 to 2026-10-31. Do not use it except for one final run.
