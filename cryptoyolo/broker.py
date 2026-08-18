"""Order execution: Binance REST client + a paper broker.

Safety ladder (see ExecutionConfig):

  1. mode="paper"                    simulated fills, nothing leaves the machine
  2. mode="binance", dry_run=True    POST /api/v3/order/test — Binance validates
                                     symbol, filters, notional and permissions,
                                     then discards the order. Nothing fills.
  3. mode="binance", dry_run=False   POST /api/v3/order — REAL ORDERS, REAL MONEY
                                     and additionally requires the environment
                                     variable CRYPTO_YOLO_ALLOW_LIVE=1

Step 3 needs two switches in two different places (a config field and an env
var) that must agree. That is deliberate: re-running a notebook cell should
never be able to place a live order by itself.

Host note: api.binance.com and testnet.binance.vision both return HTTP 451 from
US IP addresses. api.binance.us works and is the default venue. Switch with
CONFIG.execution.venue.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import time
import uuid
from datetime import datetime, timezone
from decimal import ROUND_DOWN, Decimal
from typing import Any
from urllib.parse import urlencode

import requests

from .config import CONFIG, Config, get_secret
from .store import Store, iso, utcnow


class BinanceError(RuntimeError):
    """Binance rejected the request. Carries the API's own code/message."""

    def __init__(self, status: int, code: Any, message: str):
        self.status, self.code, self.message = status, code, message
        super().__init__(f"HTTP {status} · code {code} · {message}")


class BinanceClient:
    """Thin signed-REST client. Only the endpoints this dashboard needs."""

    def __init__(self, cfg: Config = CONFIG):
        self.cfg = cfg
        self.base = cfg.execution.base_url
        self.key = get_secret("binance_key")
        self.secret = get_secret("binance_secret")
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": cfg.user_agent})
        if self.key:
            self.session.headers.update({"X-MBX-APIKEY": self.key})
        self._time_offset_ms = 0
        self._filters: dict[str, dict] = {}

    # -- plumbing -----------------------------------------------------------
    @property
    def configured(self) -> bool:
        return bool(self.key and self.secret)

    def sync_time(self) -> int:
        """Align our clock to Binance's.

        Binance rejects requests whose timestamp drifts outside recvWindow with
        error -1021. A laptop that slept for an hour will hit this immediately,
        so we measure the offset once and apply it to every signed call.
        """
        r = self.session.get(f"{self.base}/api/v3/time", timeout=self.cfg.http_timeout)
        r.raise_for_status()
        self._time_offset_ms = int(r.json()["serverTime"]) - int(time.time() * 1000)
        return self._time_offset_ms

    def _timestamp(self) -> int:
        return int(time.time() * 1000) + self._time_offset_ms

    def _sign(self, params: dict[str, Any]) -> str:
        query = urlencode(params, doseq=True)
        sig = hmac.new(self.secret.encode(), query.encode(), hashlib.sha256).hexdigest()
        return f"{query}&signature={sig}"

    def _request(self, method: str, path: str, params: dict | None = None,
                 signed: bool = False) -> Any:
        params = dict(params or {})
        url = f"{self.base}{path}"

        if signed:
            if not self.configured:
                raise BinanceError(0, "no-credentials",
                                   "BINANCE_API_KEY / BINANCE_API_SECRET are not set.")
            params.setdefault("recvWindow", self.cfg.execution.recv_window_ms)
            params["timestamp"] = self._timestamp()
            url = f"{url}?{self._sign(params)}"
            resp = self.session.request(method, url, timeout=self.cfg.http_timeout)
        else:
            resp = self.session.request(method, url, params=params,
                                        timeout=self.cfg.http_timeout)

        if resp.status_code == 451:
            raise BinanceError(451, 451,
                               f"{self.base} is geo-blocked from this IP. Set "
                               "CONFIG.execution.venue='binance-us' (or use a "
                               "supported jurisdiction).")
        if resp.status_code >= 400:
            try:
                body = resp.json()
                code, msg = body.get("code"), body.get("msg", resp.text[:300])
            except Exception:  # noqa: BLE001 - non-JSON error body
                code, msg = resp.status_code, resp.text[:300]
            # -1021 is clock drift; resync once and let the caller retry.
            if code == -1021:
                self.sync_time()
            raise BinanceError(resp.status_code, code, str(msg))
        return resp.json()

    # -- market data / filters ---------------------------------------------
    def exchange_filters(self, symbol_pair: str) -> dict[str, dict]:
        """Cache LOT_SIZE / PRICE_FILTER / NOTIONAL rules for a pair.

        Orders that ignore these are rejected outright, so every quantity and
        price this module sends is rounded to comply first.
        """
        if symbol_pair in self._filters:
            return self._filters[symbol_pair]
        info = self._request("GET", "/api/v3/exchangeInfo", {"symbol": symbol_pair})
        symbols = info.get("symbols") or []
        if not symbols:
            raise BinanceError(0, "no-symbol", f"{symbol_pair} not listed on {self.base}")
        filters = {f["filterType"]: f for f in symbols[0]["filters"]}
        filters["_status"] = symbols[0].get("status")
        filters["_baseAsset"] = symbols[0].get("baseAsset")
        filters["_quoteAsset"] = symbols[0].get("quoteAsset")
        self._filters[symbol_pair] = filters
        return filters

    @staticmethod
    def _round_step(value: float, step: str) -> str:
        """Floor `value` to a multiple of `step`, formatted without exponent."""
        d_step = Decimal(step)
        if d_step == 0:
            return format(Decimal(str(value)).normalize(), "f")
        quantized = (Decimal(str(value)) / d_step).to_integral_value(rounding=ROUND_DOWN) * d_step
        return format(quantized.normalize(), "f")

    def normalize_order(self, pair: str, qty: float, price: float | None = None
                        ) -> tuple[str, str | None, list[str]]:
        """Round qty/price to the pair's filters. Returns (qty, price, warnings)."""
        f = self.exchange_filters(pair)
        warnings: list[str] = []

        lot = f.get("LOT_SIZE", {})
        qty_s = self._round_step(qty, lot.get("stepSize", "0.00000001"))
        if float(qty_s) < float(lot.get("minQty", 0)):
            warnings.append(f"qty {qty_s} below minQty {lot.get('minQty')}")

        price_s = None
        if price is not None:
            pf = f.get("PRICE_FILTER", {})
            price_s = self._round_step(price, pf.get("tickSize", "0.00000001"))

        notional_f = f.get("NOTIONAL") or f.get("MIN_NOTIONAL") or {}
        min_notional = float(notional_f.get("minNotional", 0) or 0)
        est = float(qty_s) * float(price_s or price or 0)
        if min_notional and est and est < min_notional:
            warnings.append(f"notional ${est:.2f} below minNotional ${min_notional:.2f}")
        if f.get("_status") != "TRADING":
            warnings.append(f"symbol status is {f.get('_status')}")
        return qty_s, price_s, warnings

    # -- account ------------------------------------------------------------
    def account(self) -> dict:
        return self._request("GET", "/api/v3/account", signed=True)

    def balances(self) -> dict[str, float]:
        data = self.account()
        return {
            b["asset"]: float(b["free"]) + float(b["locked"])
            for b in data.get("balances", [])
            if float(b["free"]) + float(b["locked"]) > 0
        }

    def free_balances(self) -> dict[str, float]:
        """Spendable balances only.

        Distinct from `balances()`: funds locked in open orders still show in
        the total but cannot be spent, so sizing against the total would
        propose purchases the account can't actually fund.
        """
        data = self.account()
        return {
            b["asset"]: float(b["free"])
            for b in data.get("balances", [])
            if float(b["free"]) > 0
        }

    def price(self, pair: str) -> float:
        return float(self._request("GET", "/api/v3/ticker/price", {"symbol": pair})["price"])

    # -- orders -------------------------------------------------------------
    def place_order(self, pair: str, side: str, qty: str, order_type: str = "MARKET",
                    price: str | None = None, client_id: str | None = None,
                    test: bool = True) -> dict:
        params: dict[str, Any] = {
            "symbol": pair, "side": side.upper(), "type": order_type.upper(),
            "quantity": qty,
        }
        if client_id:
            params["newClientOrderId"] = client_id[:36]
        if order_type.upper() == "LIMIT":
            if price is None:
                raise ValueError("LIMIT orders require a price")
            params["price"] = price
            params["timeInForce"] = "GTC"

        path = "/api/v3/order/test" if test else "/api/v3/order"
        result = self._request("POST", path, params, signed=True)
        # order/test returns {} on success; make that explicit for the caller.
        return result if result else {"test": True, "validated": True, **params}

    # -- protective / resting orders ---------------------------------------
    def place_stop_limit(self, pair: str, qty: str, stop_price: str,
                         limit_price: str, side: str = "SELL",
                         client_id: str | None = None,
                         trailing_delta: int | None = None) -> dict:
        """STOP_LOSS_LIMIT.

        Binance.US does not offer market STOP_LOSS - the venue's orderTypes are
        LIMIT, LIMIT_MAKER, MARKET, STOP_LOSS_LIMIT and TAKE_PROFIT_LIMIT - so a
        protective stop is necessarily a stop-limit. That carries a real risk
        worth naming: in a gap or a fast flush the limit may not fill and the
        stop simply doesn't protect. `stop_limit_offset_bps` sets the limit
        below the trigger to make a fill likelier.
        """
        params: dict[str, Any] = {
            "symbol": pair, "side": side.upper(), "type": "STOP_LOSS_LIMIT",
            "quantity": qty, "stopPrice": stop_price, "price": limit_price,
            "timeInForce": "GTC",
        }
        if client_id:
            params["newClientOrderId"] = client_id[:36]
        if trailing_delta:
            params["trailingDelta"] = int(trailing_delta)
        return self._request("POST", "/api/v3/order", params, signed=True)

    def place_take_profit_limit(self, pair: str, qty: str, stop_price: str,
                                limit_price: str, side: str = "SELL",
                                client_id: str | None = None) -> dict:
        params: dict[str, Any] = {
            "symbol": pair, "side": side.upper(), "type": "TAKE_PROFIT_LIMIT",
            "quantity": qty, "stopPrice": stop_price, "price": limit_price,
            "timeInForce": "GTC",
        }
        if client_id:
            params["newClientOrderId"] = client_id[:36]
        return self._request("POST", "/api/v3/order", params, signed=True)

    def place_oco(self, pair: str, qty: str, take_profit_price: str,
                  stop_price: str, stop_limit_price: str, side: str = "SELL",
                  client_id: str | None = None,
                  trailing_delta: int | None = None) -> dict:
        """One-Cancels-Other: take-profit limit + stop-limit as a single order.

        This is the correct primitive when you want both a stop and a target.
        Two independent resting sells for the same quantity would let both fill
        (or the second be rejected for insufficient balance) - that is a
        double-sell hazard, not protection.

        Note there is no /api/v3/order/oco/test endpoint, so an OCO cannot be
        dry-run validated the way a plain order can. Callers must not send one
        while in dry-run mode.
        """
        params: dict[str, Any] = {
            "symbol": pair, "side": side.upper(), "quantity": qty,
            "price": take_profit_price,            # take-profit limit leg
            "stopPrice": stop_price,               # stop trigger
            "stopLimitPrice": stop_limit_price,    # stop's limit leg
            "stopLimitTimeInForce": "GTC",
        }
        if client_id:
            params["listClientOrderId"] = client_id[:36]
        if trailing_delta:
            params["trailingDelta"] = int(trailing_delta)
        # Binance.US exposes the legacy path; /api/v3/orderList/oco is 404 here.
        return self._request("POST", "/api/v3/order/oco", params, signed=True)

    def open_orders(self, pair: str | None = None) -> list[dict]:
        params = {"symbol": pair} if pair else {}
        return self._request("GET", "/api/v3/openOrders", params, signed=True)

    def cancel_open_orders(self, pair: str) -> Any:
        """Cancel every resting order on a pair, including OCO legs."""
        return self._request("DELETE", "/api/v3/openOrders", {"symbol": pair}, signed=True)


# --------------------------------------------------------------------------
# Brokers
# --------------------------------------------------------------------------
class PaperBroker:
    """Simulated execution against the SQLite book. The default."""

    mode = "paper"

    def __init__(self, store: Store, cfg: Config = CONFIG):
        self.store, self.cfg = store, cfg
        self.venue = "paper"

    def holdings(self) -> dict[str, float]:
        df = self.store.paper_positions()
        return dict(zip(df["symbol"], df["qty"])) if not df.empty else {}

    def available_cash(self) -> float | None:
        return self.store.paper_cash()

    def preflight(self, proposal: dict) -> list[str]:
        warn: list[str] = []
        if proposal["side"] == "SELL":
            held = self.holdings().get(proposal["symbol"], 0.0)
            if proposal["qty"] > held + 1e-12:
                warn.append(f"selling {proposal['qty']:.6f} but only hold {held:.6f}")
        if proposal["side"] == "BUY" and proposal["notional"] > self.store.paper_cash():
            warn.append(f"notional ${proposal['notional']:,.2f} exceeds paper cash "
                        f"${self.store.paper_cash():,.2f}")
        return warn

    def place_protection(self, proposal: dict, record: dict) -> dict | None:
        """Record simulated protection.

        Nothing rests anywhere in paper mode - there is no venue to hold the
        order. The row exists so the flow is testable and visible, but the
        protection it represents is only evaluated when you run the exits pass.
        Genuine between-run coverage requires a live venue.
        """
        ex = self.cfg.execution
        if not (ex.place_stop_orders or ex.place_limit_orders):
            return None
        row = {
            "symbol": proposal["symbol"], "kind": "simulated",
            "order_type": "SIMULATED", "exchange_ref": None, "order_list_id": None,
            "qty": float(record.get("qty") or proposal["qty"]),
            "stop_price": float(proposal["stop"]), "limit_price": None,
            "target_price": float(proposal["target"]), "trailing_delta": None,
            "status": "simulated", "mode": "paper", "venue": "paper",
            "placed_at": iso(), "response": json.dumps(
                {"note": "paper mode — no resting order exists at any exchange"}),
        }
        self.store.save_protective_order(row)
        print(f"    · protection simulated (paper): stop {proposal['stop']:,.6g} / "
              f"target {proposal['target']:,.6g} — evaluated only when you run the exits pass")
        return row

    def cancel_protection(self, symbol: str) -> int:
        return self.store.mark_protective_cancelled(symbol)

    def execute(self, proposal: dict, run_id: str) -> dict:
        fill = float(proposal["entry"])
        try:                                    # fill at live price when reachable
            from . import prices
            live = prices.latest_prices([proposal["symbol"]], self.cfg)
            fill = live.get(proposal["symbol"], fill)
        except Exception:  # noqa: BLE001 - fall back to the proposal's entry
            pass

        qty = float(proposal["qty"])
        cost = qty * fill

        # Approvals happen one at a time, so the slate-level cap isn't enough:
        # approving all three when only two are affordable must fail the third
        # here rather than drive the balance negative. The fill price is also
        # live, so it can differ from the price the cap was computed against.
        if proposal["side"] == "BUY":
            cash = self.store.paper_cash()
            if cost > cash + 1e-9:
                record = {
                    "order_id": f"paper-{uuid.uuid4().hex[:12]}",
                    "proposal_id": proposal["proposal_id"], "run_id": run_id,
                    "ts": iso(), "venue": "paper", "mode": "paper",
                    "symbol": proposal["symbol"], "side": proposal["side"],
                    "order_type": "MARKET", "qty": qty, "price": fill,
                    "status": "REJECTED_INSUFFICIENT_CASH", "exchange_ref": None,
                    "response": json.dumps({"required": round(cost, 2),
                                            "available": round(cash, 2)}),
                }
                self.store.save_order(record)
                print(f"  ✗ {proposal['symbol']}: needs ${cost:,.2f}, "
                      f"only ${cash:,.2f} available — not filled.")
                return record

        self.store.apply_paper_fill(proposal["symbol"], proposal["side"], qty, fill)
        record = {
            "order_id": f"paper-{uuid.uuid4().hex[:12]}",
            "proposal_id": proposal["proposal_id"], "run_id": run_id, "ts": iso(),
            "venue": "paper", "mode": "paper", "symbol": proposal["symbol"],
            "side": proposal["side"], "order_type": "MARKET", "qty": qty,
            "price": fill, "status": "FILLED", "exchange_ref": None,
            "response": json.dumps({"simulated": True, "fill_price": fill}),
        }
        self.store.save_order(record)
        return record


class BinanceBroker:
    """Live/validated execution through the Binance REST API."""

    mode = "binance"

    def __init__(self, store: Store, cfg: Config = CONFIG):
        self.store, self.cfg = store, cfg
        self.client = BinanceClient(cfg)
        self.venue = cfg.execution.venue
        self.live = cfg.execution.live_enabled
        if self.client.configured:
            try:
                self.client.sync_time()
            except Exception as exc:  # noqa: BLE001 - surfaced again at preflight
                print(f"  ! Binance time sync failed: {exc}")

    def pair(self, symbol: str) -> str:
        return f"{symbol}{self.cfg.execution.quote_asset}"

    def holdings(self) -> dict[str, float]:
        if not self.client.configured:
            return {}
        try:
            return self.client.balances()
        except Exception as exc:  # noqa: BLE001 - treat as unknown, not zero
            print(f"  ! Could not read Binance balances: {exc}")
            return {}

    def available_cash(self) -> float | None:
        """Free quote-asset balance, or None if it can't be read.

        None means "unknown", which is deliberately different from 0.0: the
        sizing cap is skipped rather than silently zeroing out every BUY, and
        preflight still warns per-trade.
        """
        if not self.client.configured:
            return None
        try:
            return float(self.client.free_balances().get(self.cfg.execution.quote_asset, 0.0))
        except Exception as exc:  # noqa: BLE001 - unknown, not zero
            print(f"  ! Could not read {self.cfg.execution.quote_asset} balance: {exc}")
            return None

    def preflight(self, proposal: dict) -> list[str]:
        """Validate against exchange filters before asking for approval."""
        warn: list[str] = []
        if not self.client.configured:
            return ["BINANCE_API_KEY / BINANCE_API_SECRET not set — cannot trade"]
        try:
            pair = self.pair(proposal["symbol"])
            _, _, w = self.client.normalize_order(pair, proposal["qty"], proposal["entry"])
            warn.extend(w)
        except BinanceError as exc:
            warn.append(str(exc))

        if proposal["side"] == "BUY":
            cash = self.available_cash()
            if cash is not None and float(proposal["notional"]) > cash:
                warn.append(f"notional ${float(proposal['notional']):,.2f} exceeds free "
                            f"{self.cfg.execution.quote_asset} balance ${cash:,.2f}")
        else:
            held = self.holdings().get(proposal["symbol"], 0.0)
            if float(proposal["qty"]) > held + 1e-12:
                warn.append(f"selling {float(proposal['qty']):.6f} but only hold {held:.6f}")
        return warn

    def _trailing_delta(self) -> int | None:
        """Binance's own trailing delta, in basis points (venue range 10-2000)."""
        ex = self.cfg.execution
        if not ex.use_trailing_delta:
            return None
        bps = ex.trailing_delta_bps or int(round(self.cfg.exits.trail_pct * 100))
        return max(10, min(2000, bps))

    def place_protection(self, proposal: dict, record: dict) -> dict | None:
        """Rest a stop and/or take-profit at Binance after an entry fills.

        This is the part that protects you between notebook runs: the exits pass
        only sees the market when you run it, whereas an order resting at
        Binance is watched continuously by Binance.
        """
        ex = self.cfg.execution
        if not (ex.place_stop_orders or ex.place_limit_orders):
            return None
        symbol = proposal["symbol"]
        pair = self.pair(symbol)
        qty_raw = float(record.get("qty") or proposal["qty"])
        stop = float(proposal["stop"])
        target = float(proposal["target"])
        # Limit sits below the trigger so a fast move still crosses it.
        stop_limit = stop * (1 - ex.stop_limit_offset_bps / 10_000.0)
        want_both = ex.place_stop_orders and ex.place_limit_orders

        if want_both and not ex.use_oco:
            print(f"    ⚠ {symbol}: both stop and target requested with use_oco=False. "
                  f"Two independent resting sells for the same quantity can double-sell; "
                  f"placing the STOP only.")

        try:
            qty_s, stop_s, warn = self.client.normalize_order(pair, qty_raw, stop)
            _, stop_limit_s, _ = self.client.normalize_order(pair, qty_raw, stop_limit)
            _, target_s, _ = self.client.normalize_order(pair, qty_raw, target)
        except BinanceError as exc:
            print(f"    ✗ {symbol}: cannot size protective order ({exc})")
            return None

        # OCO has no /test endpoint, so it can never be validated dry-run.
        if not self.live:
            row = {
                "symbol": symbol, "kind": "oco" if want_both and ex.use_oco else "stop",
                "order_type": "NOT_SENT", "exchange_ref": None, "order_list_id": None,
                "qty": float(qty_s), "stop_price": float(stop_s),
                "limit_price": float(stop_limit_s), "target_price": float(target_s),
                "trailing_delta": self._trailing_delta(), "status": "simulated",
                "mode": "binance-test", "venue": self.venue, "placed_at": iso(),
                "response": json.dumps({"note": "not sent — validate-only mode. Binance "
                                                "offers no test endpoint for OCO."}),
            }
            self.store.save_protective_order(row)
            print(f"    · protection NOT sent ({symbol}): validate-only mode. Would rest "
                  f"stop {stop_s} (limit {stop_limit_s}) / target {target_s}.")
            return row

        trailing = self._trailing_delta()
        cid = f"cyp{uuid.uuid4().hex[:16]}"
        try:
            if want_both and ex.use_oco:
                resp = self.client.place_oco(pair, qty_s, target_s, stop_s,
                                             stop_limit_s, client_id=cid,
                                             trailing_delta=trailing)
                kind, otype = "oco", "OCO"
                ref = str(resp.get("orderListId") or "")
            elif ex.place_stop_orders:
                resp = self.client.place_stop_limit(pair, qty_s, stop_s, stop_limit_s,
                                                    client_id=cid, trailing_delta=trailing)
                kind, otype = "stop", "STOP_LOSS_LIMIT"
                ref = str(resp.get("orderId") or "")
            else:
                resp = self.client.place_take_profit_limit(pair, qty_s, target_s,
                                                           target_s, client_id=cid)
                kind, otype = "target", "TAKE_PROFIT_LIMIT"
                ref = str(resp.get("orderId") or "")
            status = "resting"
            print(f"    ✓ protection resting on {self.venue}: {otype} {symbol} "
                  f"qty {qty_s}"
                  + (f", stop {stop_s} (limit {stop_limit_s})" if kind != "target" else "")
                  + (f", target {target_s}" if kind in ("oco", "target") else "")
                  + (f", trailingDelta {trailing}bps" if trailing else ""))
        except BinanceError as exc:
            resp, kind, otype, ref, status = {"error": str(exc)}, "stop", "FAILED", "", "failed"
            print(f"    ✗ {symbol}: protective order rejected — {exc}")

        row = {
            "symbol": symbol, "kind": kind, "order_type": otype,
            "exchange_ref": ref, "order_list_id": str(resp.get("orderListId") or "") or None,
            "qty": float(qty_s), "stop_price": float(stop_s),
            "limit_price": float(stop_limit_s), "target_price": float(target_s),
            "trailing_delta": trailing, "status": status, "mode": "binance-live",
            "venue": self.venue, "placed_at": iso(),
            "response": json.dumps({"warnings": warn, "response": resp}, default=str),
        }
        self.store.save_protective_order(row)
        return row

    def cancel_protection(self, symbol: str) -> int:
        """Cancel resting orders before selling.

        A resting sell LOCKS the base asset, so an exit that market-sells
        without cancelling first fails on insufficient free balance. This is the
        single most likely way the protective-order feature breaks the exit
        path, so it runs unconditionally before every SELL.
        """
        n = 0
        if self.live and self.client.configured:
            try:
                resting = self.store.resting_orders(symbol)
                if not resting.empty:
                    self.client.cancel_open_orders(self.pair(symbol))
                    print(f"    · cancelled {len(resting)} resting order(s) on {symbol} "
                          f"to free the balance for this sell")
            except BinanceError as exc:
                # -2011 just means there was nothing open; anything else matters.
                if getattr(exc, "code", None) != -2011:
                    print(f"    ⚠ {symbol}: could not cancel resting orders ({exc}). "
                          f"The sell may fail on locked balance.")
        n = self.store.mark_protective_cancelled(symbol)
        return n

    def execute(self, proposal: dict, run_id: str) -> dict:
        pair = self.pair(proposal["symbol"])
        use_limit = self.cfg.execution.order_type.upper() == "LIMIT"

        limit_price = None
        if use_limit:
            offset = self.cfg.execution.limit_offset_bps / 10_000.0
            ref = float(proposal["entry"])
            limit_price = ref * (1 + offset) if proposal["side"] == "BUY" else ref * (1 - offset)

        qty_s, price_s, warnings = self.client.normalize_order(pair, proposal["qty"], limit_price)
        test = not self.live
        client_id = f"cy{proposal['proposal_id'].replace('-', '')[:30]}"

        try:
            resp = self.client.place_order(
                pair, proposal["side"], qty_s,
                order_type=self.cfg.execution.order_type,
                price=price_s, client_id=client_id, test=test,
            )
            status = resp.get("status") or ("VALIDATED" if test else "SENT")
            ref = str(resp.get("orderId") or "")
            fill_price = float(resp.get("price") or 0) or float(proposal["entry"])
            if resp.get("fills"):
                fills = resp["fills"]
                notional = sum(float(f["price"]) * float(f["qty"]) for f in fills)
                filled_qty = sum(float(f["qty"]) for f in fills)
                fill_price = notional / filled_qty if filled_qty else fill_price
        except BinanceError as exc:
            resp, status, ref, fill_price = {"error": str(exc), "code": exc.code}, "REJECTED", "", 0.0
            print(f"  ✗ Binance rejected {proposal['symbol']}: {exc}")

        record = {
            "order_id": f"bnc-{uuid.uuid4().hex[:12]}",
            "proposal_id": proposal["proposal_id"], "run_id": run_id, "ts": iso(),
            "venue": self.venue, "mode": "binance-live" if self.live else "binance-test",
            "symbol": proposal["symbol"], "side": proposal["side"],
            "order_type": self.cfg.execution.order_type, "qty": float(qty_s),
            "price": fill_price, "status": status, "exchange_ref": ref,
            "response": json.dumps({"warnings": warnings, "response": resp}, default=str),
        }
        self.store.save_order(record)
        return record


def get_broker(store: Store, cfg: Config = CONFIG):
    """Pick the broker implied by config. Paper unless explicitly told otherwise."""
    if cfg.execution.mode == "binance":
        return BinanceBroker(store, cfg)
    return PaperBroker(store, cfg)


def execution_banner(cfg: Config = CONFIG) -> str:
    """One-line description of exactly what will happen on approval."""
    ex = cfg.execution
    if ex.mode == "paper":
        return "PAPER — simulated fills in the local SQLite book. Nothing is sent to any exchange."
    if ex.live_enabled:
        return (f"*** LIVE TRADING *** — real orders on {ex.venue} ({ex.base_url}). "
                f"Approved trades spend real money.")
    reason = ("dry_run=True" if ex.dry_run
              else "env CRYPTO_YOLO_ALLOW_LIVE is not set to 1")
    return (f"BINANCE VALIDATE-ONLY — orders go to /api/v3/order/test on {ex.venue} "
            f"and will NOT fill ({reason}).")
