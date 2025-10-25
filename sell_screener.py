# sell_screener.py
# Simple GUI stock "ready-to-sell" screener
# Not financial advice. For research/education only.

import threading
import queue
import datetime as dt
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import csv
import re
import numpy as np
import pandas as pd

try:
    import yfinance as yf
except ImportError:
    raise SystemExit("Please install dependencies first: pip install yfinance pandas numpy")

APP_TITLE = "Sell Signal Screener"
DEFAULT_LOOKBACK_DAYS = 420  # ~1.6 years, gives room for 52-week metrics
MARKET_REFERENCE_TICKER = "^GSPC"
MARKET_LOOKBACK_DAYS = 320


def format_quantity(value: object) -> str:
    """Pretty-print a quantity value while keeping trailing zeros minimal."""

    try:
        num = float(value)
    except (TypeError, ValueError):
        return str(value)

    if num.is_integer():
        return str(int(num))

    return format(num, "g")


MARKET_PRESETS = {
    "Bullish": {
        "use_rsi": True,
        "rsi_threshold": 75,
        "use_ma": True,
        "ma_fast": 12,
        "ma_slow": 40,
        "use_dd": False,
        "drawdown_pct": 12.5,
        "use_atr": True,
        "atr_multiple": 3.5,
        "min_triggers": 2,
    },
    "Sideways": {
        "use_rsi": True,
        "rsi_threshold": 70,
        "use_ma": True,
        "ma_fast": 18,
        "ma_slow": 45,
        "use_dd": True,
        "drawdown_pct": 15.0,
        "use_atr": True,
        "atr_multiple": 3.0,
        "min_triggers": 2,
    },
    "Bearish": {
        "use_rsi": True,
        "rsi_threshold": 60,
        "use_ma": True,
        "ma_fast": 20,
        "ma_slow": 50,
        "use_dd": True,
        "drawdown_pct": 10.0,
        "use_atr": True,
        "atr_multiple": 2.5,
        "min_triggers": 1,
    },
}

# ---------- Indicator helpers ----------

def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = np.where(delta > 0, delta, 0.0)
    down = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(up, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    roll_down = pd.Series(down, index=series.index).ewm(alpha=1/period, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(0)

def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()

def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(window=period, min_periods=period).mean()

# ---------- Sell rules ----------

def rule_rsi_overbought_cross(price: pd.Series, threshold: int = 70) -> dict:
    r = rsi(price, 14)
    # True when RSI is currently < threshold but was recently above => crossed down
    crossed = (r.iloc[-2] >= threshold) and (r.iloc[-1] < threshold)
    return {"name": f"RSI cross below {threshold}", "triggered": bool(crossed), "value": round(float(r.iloc[-1]), 2)}

def rule_ma_bearish_cross(price: pd.Series, fast: int = 20, slow: int = 50) -> dict:
    f = ema(price, fast)
    s = sma(price, slow)
    if pd.isna(s.iloc[-1]) or pd.isna(s.iloc[-2]):
        return {"name": f"EMA{fast}↓SMA{slow}", "triggered": False, "value": None}
    # Bearish cross: fast was above and now below slow
    crossed = (f.iloc[-2] > s.iloc[-2]) and (f.iloc[-1] < s.iloc[-1])
    val = f"{round(float(f.iloc[-1]),2)} vs {round(float(s.iloc[-1]),2)}"
    return {"name": f"EMA{fast}↓SMA{slow} cross", "triggered": bool(crossed), "value": val}

def rule_drawdown_from_52w(price: pd.Series, max_dd_pct: float = 15.0) -> dict:
    # percent off the 252-trading-day high (~1y)
    window = min(252, len(price))
    if window < 50:
        return {"name": f">{max_dd_pct}% below 52w high", "triggered": False, "value": None}
    rolling_high = price.rolling(window=window).max()
    dd = (rolling_high.iloc[-1] - price.iloc[-1]) / rolling_high.iloc[-1] * 100.0
    trig = dd >= max_dd_pct
    return {"name": f">{max_dd_pct:.0f}% below 52w high", "triggered": bool(trig), "value": round(float(dd), 2)}

def rule_atr_trailing_stop(high: pd.Series, low: pd.Series, close: pd.Series, multiple: float = 3.0) -> dict:
    # Simple Chandelier-like exit: stop = highest close(22) - multiple*ATR(14); sell if close < stop
    period_high = 22
    a = atr(high, low, close, 14)
    highest_close = close.rolling(window=period_high).max()
    stop = highest_close - multiple * a
    if pd.isna(stop.iloc[-1]):
        return {"name": f"Close < (Highest{period_high} - {multiple}×ATR)", "triggered": False, "value": None}
    trig = close.iloc[-1] < stop.iloc[-1]
    val = f"Close {round(float(close.iloc[-1]),2)} vs Stop {round(float(stop.iloc[-1]),2)}"
    return {"name": f"ATR trailing stop ({multiple}×)", "triggered": bool(trig), "value": val}

# ---------- Data ----------

def fetch_history(ticker: str, period_days: int = DEFAULT_LOOKBACK_DAYS) -> pd.DataFrame:
    end = dt.date.today()
    start = end - dt.timedelta(days=period_days + 20)
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=end.isoformat(),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
        interval="1d"
    )
    if isinstance(df.columns, pd.MultiIndex):
        # if yfinance returns multiindex for multiple tickers, select first level
        df = df.xs(ticker, axis=1, level=0, drop_level=False).droplevel(0, axis=1)
    df = df.dropna()
    return df


def determine_market_condition() -> tuple[str, str]:
    """Classify the broad market as Bullish, Sideways, or Bearish."""

    df = fetch_history(MARKET_REFERENCE_TICKER, MARKET_LOOKBACK_DAYS)
    if df.empty or len(df) < 220:
        raise ValueError("Not enough data to determine market condition")

    close = df["Close"]
    sma50 = sma(close, 50)
    sma200 = sma(close, 200)

    latest_close = float(close.iloc[-1])
    latest_sma50 = float(sma50.iloc[-1])
    latest_sma200 = float(sma200.iloc[-1])

    if np.isnan(latest_sma50) or np.isnan(latest_sma200):
        raise ValueError("Not enough data for moving averages")

    ratio_to_200 = latest_close / latest_sma200
    sma50_20_ago = float(sma50.iloc[-20]) if not np.isnan(sma50.iloc[-20]) else latest_sma50
    slope = (latest_sma50 - sma50_20_ago) / sma50_20_ago if sma50_20_ago else 0.0

    if latest_close > latest_sma200 and latest_sma50 >= latest_sma200 and slope >= 0:
        condition = "Bullish"
    elif latest_close < latest_sma200 and latest_sma50 <= latest_sma200 and slope <= 0:
        condition = "Bearish"
    else:
        condition = "Sideways"

    pct_to_200 = (ratio_to_200 - 1.0) * 100.0
    slope_pct = slope * 100.0
    detail = f"Close vs SMA200: {pct_to_200:+.2f}% | SMA50 slope (20d): {slope_pct:+.2f}%"
    return condition, detail

# ---------- Screening logic ----------

def evaluate_ticker(ticker: str, config: dict) -> dict:
    try:
        df = fetch_history(ticker)
        if df.empty or len(df) < 50:
            return {"ticker": ticker, "status": "Insufficient data", "ready": False, "price": None, "rules": []}

        close = df["Close"]
        high = df["High"]
        low = df["Low"]

        results = []

        if config["use_rsi"]:
            results.append(rule_rsi_overbought_cross(close, config["rsi_threshold"]))

        if config["use_ma"]:
            results.append(rule_ma_bearish_cross(close, config["ma_fast"], config["ma_slow"]))

        if config["use_dd"]:
            results.append(rule_drawdown_from_52w(close, config["drawdown_pct"]))

        if config["use_atr"]:
            results.append(rule_atr_trailing_stop(high, low, close, config["atr_multiple"]))

        triggered = [r for r in results if r["triggered"]]
        ready = len(triggered) >= config["min_triggers"]

        last_price = float(close.iloc[-1])
        status = "Ready to SELL" if ready else "Hold/Review"

        return {
            "ticker": ticker.upper(),
            "status": status,
            "ready": ready,
            "price": round(last_price, 2),
            "rules": results
        }
    except Exception as e:
        return {"ticker": ticker, "status": f"Error: {e}", "ready": False, "price": None, "rules": []}

# ---------- GUI ----------

class ScreenerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("900x560")
        self.minsize(820, 520)

        self.watchlist: list[dict[str, object]] = []
        self.queue = queue.Queue()
        self.market_condition_var = tk.StringVar(value="Market: Checking…")
        self._sort_directions: dict[str, bool] = {}
        self._column_labels: dict[str, str] = {}

        self._build_controls()
        self._build_table()
        self._build_status()

        # defaults
        self.use_rsi_var.set(1)
        self.rsi_thr_var.set(70)
        self.use_ma_var.set(1)
        self.fast_var.set(20)
        self.slow_var.set(50)
        self.use_dd_var.set(0)
        self.dd_var.set(15)
        self.use_atr_var.set(1)
        self.atr_var.set(3.0)
        self.min_triggers_var.set(1)

        self.after(200, self._poll_queue)
        self.refresh_market_condition()

    def _build_controls(self):
        frm = ttk.LabelFrame(self, text="Watchlist & Rules", padding=10)
        frm.pack(side=tk.TOP, fill=tk.X, padx=10, pady=10)

        # Watchlist entry
        ttk.Label(frm, text="Ticker:").grid(row=0, column=0, sticky="w")
        self.ticker_entry = ttk.Entry(frm, width=12)
        self.ticker_entry.grid(row=0, column=1, padx=(4, 8))

        ttk.Label(frm, text="Cost:").grid(row=0, column=2, sticky="e")
        self.cost_var = tk.StringVar()
        self.cost_entry = ttk.Entry(frm, textvariable=self.cost_var, width=10)
        self.cost_entry.grid(row=0, column=3, padx=(4, 12))

        ttk.Label(frm, text="Qty (lots):").grid(row=0, column=4, sticky="e")
        self.qty_var = tk.StringVar(value="1")
        self.qty_entry = ttk.Entry(frm, textvariable=self.qty_var, width=8)
        self.qty_entry.grid(row=0, column=5, padx=(4, 12))

        ttk.Button(frm, text="Add/Update", command=self.add_ticker).grid(row=0, column=6)
        ttk.Button(frm, text="Remove Selected", command=self.remove_selected).grid(row=0, column=7, padx=(12, 0))
        ttk.Button(frm, text="Load CSV", command=self.load_csv).grid(row=0, column=8, padx=(12, 0))
        ttk.Button(frm, text="Save CSV", command=self.save_csv).grid(row=0, column=9, padx=(6, 0))
        ttk.Button(frm, text="Scan", command=self.scan).grid(row=0, column=10, padx=(24, 0))
        ttk.Button(frm, text="Help", command=self.show_help).grid(row=0, column=11, padx=(6, 0))
        ttk.Button(frm, text="Auto Adjust to Market", command=self.auto_adjust_to_market).grid(row=0, column=12, padx=(18, 0))

        # Listbox of tickers
        self.listbox = tk.Listbox(frm, height=6, selectmode=tk.EXTENDED)
        self.listbox.grid(row=1, column=0, columnspan=6, padx=(0, 12), pady=(8, 0), sticky="nsew")
        frm.grid_columnconfigure(1, weight=1)
        frm.grid_columnconfigure(5, weight=1)

        ttk.Label(frm, textvariable=self.market_condition_var).grid(row=2, column=0, columnspan=13, sticky="w", pady=(8, 0))

        # Rules panel
        rules = ttk.LabelFrame(frm, text="Sell Rules", padding=10)
        rules.grid(row=1, column=6, columnspan=7, sticky="nsew", pady=(8, 0))
        for c in range(5):
            rules.grid_columnconfigure(c, weight=1)

        # RSI
        self.use_rsi_var = tk.IntVar(value=1)
        ttk.Checkbutton(rules, text="RSI cross down", variable=self.use_rsi_var).grid(row=0, column=0, sticky="w")
        ttk.Label(rules, text="Threshold").grid(row=0, column=1, sticky="e")
        self.rsi_thr_var = tk.IntVar(value=70)
        ttk.Spinbox(rules, from_=50, to=90, textvariable=self.rsi_thr_var, width=5).grid(row=0, column=2, sticky="w")

        # MA cross
        self.use_ma_var = tk.IntVar(value=1)
        ttk.Checkbutton(rules, text="EMA fast crosses below SMA slow", variable=self.use_ma_var).grid(row=1, column=0, sticky="w", pady=(4,0))
        ttk.Label(rules, text="Fast").grid(row=1, column=1, sticky="e")
        self.fast_var = tk.IntVar(value=20)
        ttk.Spinbox(rules, from_=5, to=50, textvariable=self.fast_var, width=5).grid(row=1, column=2, sticky="w")
        ttk.Label(rules, text="Slow").grid(row=1, column=3, sticky="e")
        self.slow_var = tk.IntVar(value=50)
        ttk.Spinbox(rules, from_=20, to=250, textvariable=self.slow_var, width=5).grid(row=1, column=4, sticky="w")

        # Drawdown
        self.use_dd_var = tk.IntVar(value=0)
        ttk.Checkbutton(rules, text="Below 52w high by at least", variable=self.use_dd_var).grid(row=2, column=0, sticky="w", pady=(4,0))
        self.dd_var = tk.DoubleVar(value=15.0)
        ttk.Spinbox(rules, from_=5, to=60, increment=0.5, textvariable=self.dd_var, width=6).grid(row=2, column=1, sticky="w")
        ttk.Label(rules, text="%").grid(row=2, column=2, sticky="w")

        # ATR trailing stop
        self.use_atr_var = tk.IntVar(value=1)
        ttk.Checkbutton(rules, text="ATR trailing stop (close < highest(22) - N×ATR)", variable=self.use_atr_var).grid(row=3, column=0, sticky="w", pady=(4,0))
        self.atr_var = tk.DoubleVar(value=3.0)
        ttk.Label(rules, text="N=").grid(row=3, column=1, sticky="e")
        ttk.Spinbox(rules, from_=1.0, to=5.0, increment=0.5, textvariable=self.atr_var, width=6).grid(row=3, column=2, sticky="w")

        # Min triggers
        ttk.Label(rules, text="Min signals to flag as 'Ready to SELL'").grid(row=4, column=0, sticky="w", pady=(8, 0))
        self.min_triggers_var = tk.IntVar(value=1)
        ttk.Spinbox(rules, from_=1, to=4, textvariable=self.min_triggers_var, width=6).grid(row=4, column=1, sticky="w", pady=(8, 0))

    def _build_table(self):
        container = ttk.Frame(self)
        container.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0,10))

        columns = ("ticker", "price", "cost", "quantity", "gain_loss", "status", "signals")
        self.tree = ttk.Treeview(container, columns=columns, show="headings")

        self._column_labels = {
            "ticker": "Ticker",
            "price": "Last Price",
            "cost": "Cost Basis",
            "quantity": "Qty (lots)",
            "gain_loss": "Gain / Loss ($, %)",
            "status": "Result",
            "signals": "Signals / Values",
        }

        for col, label in self._column_labels.items():
            self._configure_heading(col, label)

        self.tree.column("ticker", width=100, anchor=tk.CENTER)
        self.tree.column("price", width=110, anchor=tk.E)
        self.tree.column("cost", width=110, anchor=tk.E)
        self.tree.column("quantity", width=110, anchor=tk.CENTER)
        self.tree.column("gain_loss", width=160, anchor=tk.E)
        self.tree.column("status", width=150, anchor=tk.CENTER)
        self.tree.column("signals", width=520, anchor=tk.W)

        vsb = ttk.Scrollbar(container, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=vsb.set)
        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        # color tags
        self.tree.tag_configure("ready", foreground="#b00020")     # red-ish
        self.tree.tag_configure("hold", foreground="#006400")       # green-ish
        self.tree.tag_configure("warn", foreground="#8B8000")       # dark golden

    def _build_status(self):
        self.status = tk.StringVar(value="Add tickers and click Scan.")
        bar = ttk.Label(self, textvariable=self.status, anchor="w")
        bar.pack(fill=tk.X, padx=10, pady=(0,10))

    # ---------- Actions ----------

    def refresh_market_condition(self):
        self.market_condition_var.set("Market: Checking…")
        threading.Thread(target=self._fetch_market_condition, args=(False,), daemon=True).start()

    def auto_adjust_to_market(self):
        self.status.set("Detecting market condition…")
        self.market_condition_var.set("Market: Checking…")
        threading.Thread(target=self._fetch_market_condition, args=(True,), daemon=True).start()

    def _fetch_market_condition(self, apply_preset: bool):
        try:
            condition, detail = determine_market_condition()
            self.queue.put(
                {
                    "type": "MARKET",
                    "condition": condition,
                    "detail": detail,
                    "apply": apply_preset,
                }
            )
        except Exception as exc:
            self.queue.put({"type": "MARKET_ERROR", "error": str(exc), "apply": apply_preset})

    def _refresh_watchlist(self):
        self.listbox.delete(0, tk.END)
        for item in self.watchlist:
            cost = item.get("cost")
            qty = item.get("quantity")
            cost_text = "—" if cost is None else f"{cost:.2f}"
            qty_text = "—" if qty in (None, "") else format_quantity(qty)
            display = f"{item['ticker']} | Cost: {cost_text} | Qty (lots): {qty_text}"
            self.listbox.insert(tk.END, display)

    def add_ticker(self):
        t = self.ticker_entry.get().strip().upper()
        if not t:
            return

        cost_str = self.cost_var.get().strip()
        qty_str = self.qty_var.get().strip()

        try:
            cost = float(cost_str)
            if cost < 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid cost", "Please enter a valid non-negative cost basis.")
            return

        try:
            quantity = float(qty_str)
            if quantity <= 0:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid quantity", "Quantity (lots) must be a positive number.")
            return

        updated = False
        for item in self.watchlist:
            if item["ticker"] == t:
                item["cost"] = cost
                item["quantity"] = quantity
                updated = True
                break
        if not updated:
            self.watchlist.append({"ticker": t, "cost": cost, "quantity": quantity})

        self._refresh_watchlist()
        self.ticker_entry.delete(0, tk.END)
        self.cost_var.set("")
        self.qty_var.set("1")
        self.status.set(
            f"{'Updated' if updated else 'Added'} {t} @ {cost:.2f} ({format_quantity(quantity)} lots)."
        )

    def remove_selected(self):
        sel = list(self.listbox.curselection())
        if not sel:
            return
        for idx in reversed(sel):
            if 0 <= idx < len(self.watchlist):
                self.watchlist.pop(idx)
        self._refresh_watchlist()
        if sel:
            removed = len(sel)
            plural = "s" if removed > 1 else ""
            self.status.set(f"Removed {removed} ticker{plural}.")

    def load_csv(self):
        path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            with open(path, newline="") as f:
                rows = list(csv.reader(f))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load CSV:\n{e}")
            return

        loaded: list[dict[str, object]] = []

        try:
            for row in rows:
                if not row:
                    continue
                ticker = row[0].strip().upper()
                if not ticker or ticker == "TICKER":
                    continue
                cost = float(row[1]) if len(row) > 1 and row[1].strip() else 0.0
                quantity = float(row[2]) if len(row) > 2 and row[2].strip() else 1.0
                loaded.append({"ticker": ticker, "cost": cost, "quantity": quantity})
        except ValueError:
            # Fallback: support legacy CSV that only listed tickers
            loaded = []
            for row in rows:
                for cell in row:
                    ticker = cell.strip().upper()
                    if ticker and ticker != "TICKER":
                        loaded.append({"ticker": ticker, "cost": 0.0, "quantity": 1.0})

        unique: dict[str, dict[str, object]] = {}
        for item in loaded:
            unique[item["ticker"]] = item

        self.watchlist = list(unique.values())
        self._refresh_watchlist()
        self.status.set(f"Loaded {len(self.watchlist)} tickers.")

    def save_csv(self):
        path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV", "*.csv")])
        if not path:
            return
        try:
            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Ticker", "Cost", "Quantity (lots)"])
                for item in self.watchlist:
                    writer.writerow([item["ticker"], item["cost"], item["quantity"]])
            self.status.set(f"Saved {len(self.watchlist)} tickers.")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save CSV:\n{e}")

    def get_config(self) -> dict:
        # Validate MA fast/slow
        fast = int(self.fast_var.get())
        slow = int(self.slow_var.get())
        if fast >= slow:
            messagebox.showwarning("Rule Warning", "EMA fast should be less than SMA slow. Adjusted automatically.")
            fast = min(fast, slow - 1)
            self.fast_var.set(fast)

        return {
            "use_rsi": bool(self.use_rsi_var.get()),
            "rsi_threshold": int(self.rsi_thr_var.get()),
            "use_ma": bool(self.use_ma_var.get()),
            "ma_fast": fast,
            "ma_slow": slow,
            "use_dd": bool(self.use_dd_var.get()),
            "drawdown_pct": float(self.dd_var.get()),
            "use_atr": bool(self.use_atr_var.get()),
            "atr_multiple": float(self.atr_var.get()),
            "min_triggers": int(self.min_triggers_var.get())
        }

    def scan(self):
        watchlist = [dict(item) for item in self.watchlist]
        if not watchlist:
            messagebox.showinfo("No tickers", "Add one or more tickers first.")
            return

        config = self.get_config()
        self.tree.delete(*self.tree.get_children())
        self.status.set("Scanning…")

        def worker():
            for item in watchlist:
                res = evaluate_ticker(item["ticker"], config)
                res.update({"cost": item.get("cost"), "quantity": item.get("quantity")})
                self.queue.put(res)
            self.queue.put("__DONE__")

        threading.Thread(target=worker, daemon=True).start()

    def _poll_queue(self):
        try:
            while True:
                item = self.queue.get_nowait()
                if item == "__DONE__":
                    self.status.set("Scan complete.")
                    break
                if isinstance(item, dict) and item.get("type") == "MARKET":
                    self._handle_market_message(item)
                    continue
                if isinstance(item, dict) and item.get("type") == "MARKET_ERROR":
                    self._handle_market_error(item)
                    continue
                self._add_result_row(item)
        except queue.Empty:
            pass
        finally:
            self.after(150, self._poll_queue)

    def _apply_market_preset(self, condition: str):
        preset = MARKET_PRESETS.get(condition)
        if not preset:
            return

        self.use_rsi_var.set(int(preset["use_rsi"]))
        self.rsi_thr_var.set(int(preset["rsi_threshold"]))
        self.use_ma_var.set(int(preset["use_ma"]))
        self.fast_var.set(int(preset["ma_fast"]))
        self.slow_var.set(int(preset["ma_slow"]))
        self.use_dd_var.set(int(preset["use_dd"]))
        self.dd_var.set(float(preset["drawdown_pct"]))
        self.use_atr_var.set(int(preset["use_atr"]))
        self.atr_var.set(float(preset["atr_multiple"]))
        self.min_triggers_var.set(int(preset["min_triggers"]))

    def _handle_market_message(self, item: dict):
        condition = item["condition"]
        detail = item["detail"]
        apply = item.get("apply", False)
        self.market_condition_var.set(f"Market: {condition} — {detail}")
        if apply:
            self._apply_market_preset(condition)
            self.status.set(f"Applied {condition} preset based on market condition.")
        else:
            self.status.set("Market condition updated.")

    def _handle_market_error(self, item: dict):
        self.market_condition_var.set("Market: Unable to determine (see status)")
        if item.get("apply"):
            self.status.set(f"Failed to auto-adjust: {item.get('error')}")
        else:
            self.status.set(f"Failed to refresh market info: {item.get('error')}")

    def _add_result_row(self, res: dict):
        sig_texts = []
        for r in res["rules"]:
            mark = "✅" if r["triggered"] else "—"
            val = "" if r["value"] is None else f" ({r['value']})"
            sig_texts.append(f"{mark} {r['name']}{val}")
        sig_col = " | ".join(sig_texts) if sig_texts else "No rules evaluated"

        tags = ()
        if res["status"].startswith("Ready"):
            tags = ("ready",)
        elif res["status"].startswith("Error"):
            tags = ("warn",)
        else:
            tags = ("hold",)

        price = res.get("price")
        cost = res.get("cost")
        quantity = res.get("quantity")

        price_text = "—" if price is None else f"{price:.2f}"
        cost_text = "—" if cost is None else f"{float(cost):.2f}"
        qty_text = "—" if quantity in (None, "") else format_quantity(quantity)

        gain_text = "—"
        if price is not None and cost not in (None, ""):
            try:
                cost_val = float(cost)
                qty_val = float(quantity) if quantity not in (None, "") else 1.0

                if qty_val <= 0:
                    qty_val = 1.0

                # Detect whether the provided cost looks like a per-unit or total
                # cost basis. When the absolute difference between the entered cost
                # and the current price is larger than the difference between the
                # inferred per-unit cost (cost / qty) and the current price, we
                # treat the input as a total cost basis. This allows users to enter
                # either total or per-unit costs without needing an explicit toggle.
                per_unit_assuming_total = cost_val / qty_val
                diff_per_share = abs(cost_val - price)
                diff_assuming_total = abs(per_unit_assuming_total - price)

                if qty_val > 0 and diff_assuming_total < diff_per_share:
                    total_cost = cost_val
                    cost_per_unit = per_unit_assuming_total
                else:
                    cost_per_unit = cost_val
                    total_cost = cost_val * qty_val

                diff = price * qty_val - total_cost
                pct = ((price - cost_per_unit) / cost_per_unit * 100.0) if cost_per_unit else None

                sign_diff = "+" if diff >= 0 else ""
                diff_text = f"{sign_diff}{diff:.2f}"

                if pct is None:
                    gain_text = diff_text
                else:
                    sign_pct = "+" if pct >= 0 else ""
                    gain_text = f"{diff_text} ({sign_pct}{pct:.2f}%)"
            except (TypeError, ValueError, ZeroDivisionError):
                pass

        self.tree.insert(
            "",
            tk.END,
            values=(
                res["ticker"],
                price_text,
                cost_text,
                qty_text,
                gain_text,
                res["status"],
                sig_col,
            ),
            tags=tags,
        )

    def show_help(self):
        msg = (
            "How it works:\n"
            "• RSI cross down: flags when RSI(14) drops back below your threshold (default 70).\n"
            "• EMA/SMA cross: flags when EMA(fast) crosses below SMA(slow) (default 20 vs 50).\n"
            "• 52-week drawdown: flags if price is ≥ your % below the 52-week high.\n"
            "• ATR trailing stop: flags when close falls below (highest 22-day close − N×ATR(14)).\n\n"
            "Set 'Min signals' to how many rules must trigger to mark 'Ready to SELL'.\n\n"
            "Tips:\n"
            "• Use CSV to save/load your watchlist.\n"
            "• Signals are heuristics, not advice. Always research before acting."
        )
        messagebox.showinfo("Help", msg)

    def _configure_heading(self, column: str, text: str) -> None:
        self.tree.heading(column, text=text, command=lambda c=column: self._sort_tree(c))

    def _sort_tree(self, column: str) -> None:
        reverse = self._sort_directions.get(column, False)
        rows = [
            (*self._sort_key(column, self.tree.set(child, column)), child)
            for child in self.tree.get_children("")
        ]

        sortable = [row for row in rows if not row[0]]
        missing = [row for row in rows if row[0]]

        sortable.sort(key=lambda item: item[1], reverse=reverse)
        ordered = sortable + missing

        for index, (_, _, item) in enumerate(ordered):
            self.tree.move(item, "", index)

        self._sort_directions[column] = not reverse
        for col, label in self._column_labels.items():
            if col == column:
                arrow = "▼" if reverse else "▲"
                self._configure_heading(col, f"{label} {arrow}")
            else:
                self._sort_directions[col] = False
                self._configure_heading(col, label)

    def _sort_key(self, column: str, value: str) -> tuple[bool, object]:
        if value in ("", "—", None):
            return (True, "")

        text = str(value).strip()

        if column in {"price", "cost", "quantity"}:
            try:
                return (False, float(text.replace(",", "")))
            except ValueError:
                pass

        if column == "gain_loss":
            match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
            if match:
                try:
                    return (False, float(match.group().replace(",", "")))
                except ValueError:
                    pass

        return (False, text.lower())

if __name__ == "__main__":
    app = ScreenerApp()
    app.mainloop()
