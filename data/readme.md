## Funds, task 14 fresh fold (data/fund/)

```sh
mkdir -p data/fund
python - <<'EOF'
import yfinance as yf
for t in ["VFINX", "VFITX"]:
    yf.Ticker(t).history(start="1991-01-01", end="2005-10-01", auto_adjust=False).to_csv(f"data/fund/{t.lower()}.csv")
EOF
```

## FRED daily series, task 15

```sh
for s in VIXCLS DCOILBRENTEU DGS2; do curl -sL -o "data/fred/$(echo $s | tr A-Z a-z).csv" "https://fred.stlouisfed.org/graph/fredgraph.csv?id=$s"; done
```
