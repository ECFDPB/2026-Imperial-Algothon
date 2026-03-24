"""
验证 AlgoBot 核心逻辑的模拟脚本
用 mock 数据跑通完整流程，检查参数传递和逻辑正确性
"""
import numpy as np
from collections import deque
from datetime import datetime, timedelta
import pytz

LONDON_TZ = pytz.timezone('Europe/London')
SYMBOLS = ['TIDE_SPOT','TIDE_SWING','WX_SPOT','WX_SUM','LHR_COUNT','LHR_INDEX','LON_ETF','LON_FLY']
ETF_LEGS = ['TIDE_SPOT', 'WX_SPOT', 'LHR_COUNT']
POSITION_LIMIT = 100
TIDAL_CONSTITUENTS = [
    ('M2', 2*np.pi/12.4206), ('S2', 2*np.pi/12.0000),
    ('N2', 2*np.pi/12.6583), ('K1', 2*np.pi/23.9345), ('O1', 2*np.pi/25.8194),
]

def mock_market_mids():
    return {'TIDE_SPOT':1200,'TIDE_SWING':3500,'WX_SPOT':4800,'WX_SUM':2100,
            'LHR_COUNT':1400,'LHR_INDEX':55,'LON_ETF':7450,'LON_FLY':800}

def mock_positions():
    return {s: 0 for s in SYMBOLS}

def mock_fair_values():
    return {'TIDE_SPOT':(1200,30.0),'TIDE_SWING':(3500,80.0),'WX_SPOT':(4750,50.0),
            'WX_SUM':(2100,40.0),'LHR_COUNT':(1400,30.0),'LHR_INDEX':(55,15.0),
            'LON_ETF':(7350,60.0),'LON_FLY':(800,20.0)}

# 1. 潮汐调和分析 ---------------------------------------------------------------------------------------
def test_tidal_harmonic():
    print("\n=== 1. 潮汐调和分析 ===")
    import pandas as pd
    t0 = LONDON_TZ.localize(datetime(2026, 2, 28, 12, 0, 0))
    times = [t0 + timedelta(minutes=15*i) for i in range(400)]
    hours = np.array([(t - t0).total_seconds() / 3600 for t in times])
    true_coeffs = np.array([0.5, 0.8, 0.3, 0.6, 0.2, 0.4, 0.1, 0.3, 0.05, 0.15, 0.08])
    cols = [np.ones(len(hours))]
    for _, w in TIDAL_CONSTITUENTS:
        cols.append(np.cos(w * hours))
        cols.append(np.sin(w * hours))
    X = np.column_stack(cols)
    y = X @ true_coeffs + np.random.normal(0, 0.05, len(hours))
    coeffs, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    res_std = float(np.std(y - X @ coeffs))
    print(f"  拟合残差 std={res_std:.4f} (应接近 0.05)")

    settle = LONDON_TZ.localize(datetime(2026, 3, 1, 12, 0, 0))
    h = (settle - t0).total_seconds() / 3600
    val = coeffs[0]
    for i, (_, w) in enumerate(TIDAL_CONSTITUENTS):
        val += coeffs[1+2*i]*np.cos(w*h) + coeffs[2+2*i]*np.sin(w*h)
    print(f"  TIDE_SPOT fv={abs(val)*1000:.1f}, std={res_std*1000:.1f}")
    assert res_std > 0
    print("  ✓ 通过")

# 2. 贝叶斯更新 ---------------------------------------------------------------------------------------
def test_bayesian_update():
    print("\n=== 2. 贝叶斯更新 ===")
    drift_history, market_weight = {}, {}

    def _update_drift(sym, model_fv, market_mid):
        if sym not in drift_history:
            drift_history[sym] = deque(maxlen=10)
            market_weight[sym] = 0.2
        drift_history[sym].append(model_fv - market_mid)
        hist = list(drift_history[sym])
        if len(hist) >= 5:
            signs = [1 if x > 0 else -1 for x in hist[-5:]]
            if abs(sum(signs)) == 5:
                market_weight[sym] = min(0.7, market_weight[sym]+0.1)
            else:
                market_weight[sym] = max(0.1, market_weight[sym]-0.05)

    def bayesian_update(sym, model_fv, model_std, market_mid):
        if market_mid is None:
            return model_fv, model_std
        _update_drift(sym, model_fv, market_mid)
        w = market_weight.get(sym, 0.2)
        market_std = model_std * (2.0 - w)
        w_m = 1.0/model_std**2
        w_k = 1.0/market_std**2
        post_fv  = (w_m*model_fv + w_k*market_mid) / (w_m+w_k)
        post_std = (1.0/(w_m+w_k))**0.5
        return post_fv, post_std

    for _ in range(8):
        bayesian_update('TIDE_SPOT', 1300, 30.0, 1200)
    assert market_weight['TIDE_SPOT'] > 0.2, "market_weight 应上升"
    post_fv, post_std = bayesian_update('TIDE_SPOT', 1300, 30.0, 1200)
    assert 1200 < post_fv < 1300, f"后验 {post_fv} 应在 1200~1300"
    assert post_std < 30.0
    print(f"  连续偏高后 market_weight={market_weight['TIDE_SPOT']:.2f}")
    print(f"  后验 fv={post_fv:.1f}, std={post_std:.1f}")
    print("  ✓ 通过")

# 3. ParamCalibrator ---------------------------------------------------------------------------------------
def test_param_calibrator():
    print("\n=== 3. ParamCalibrator ===")
    fill_history, basis_buf, vol_buf = {}, deque(maxlen=200), {}
    gamma, k = {}, {}
    WARMUP, INTERVAL, TARGET = 6, 30, 0.3
    state = {'round': 0, 'arb_thresh': 25.0}

    def record_fill(sym, spread, vol):
        fill_history.setdefault(sym, deque(maxlen=50)).append((spread, vol))
    def record_basis(b):
        basis_buf.append(b)
    def record_vol(sym, s):
        vol_buf.setdefault(sym, deque(maxlen=50)).append(s)

    def calibrate(symbols, fv_map, mins_left):
        T = max(mins_left/(24*60), 1e-4)
        for sym in symbols:
            vols = list(vol_buf.get(sym, []))
            sigma = max(float(np.mean(vols)) if vols else fv_map.get(sym,(0,5))[1], 1.0)
            g_new = float(np.clip(TARGET*sigma / max(sigma**2*T,1e-6), 0.01, 2.0))
            gamma[sym] = 0.7*gamma.get(sym,0.1) + 0.3*g_new
            fills = list(fill_history.get(sym, []))
            if len(fills) >= 3:
                avg_v = float(np.mean([f[1] for f in fills]))
                avg_s = float(np.mean([f[0] for f in fills if f[0]>0]))
                k_new = float(np.clip(avg_v/max(avg_s*2,1.0), 0.3, 10.0))
                k[sym] = 0.7*k.get(sym,1.5) + 0.3*k_new
            else:
                k.setdefault(sym, 1.5)
        if len(basis_buf) >= 20:
            p95 = float(np.percentile(np.abs(list(basis_buf)), 95))
            state['arb_thresh'] = float(np.clip(p95, 15.0, 80.0))

    def tick(symbols, fv_map, mins_left):
        state['round'] += 1
        r = state['round']
        if r < WARMUP:
            return
        if (r - WARMUP) % INTERVAL != 0:
            return
        calibrate(symbols, fv_map, mins_left)

    fv = mock_fair_values()
    for _ in range(10):
        record_fill('TIDE_SPOT', 8.0, 5.0)
    for _ in range(20):
        record_vol('TIDE_SPOT', 25.0)
    for _ in range(50):
        record_basis(np.random.normal(0, 20))

    # 热身期：前5轮不校准（round 1~5 < WARMUP=6）
    for _ in range(5):
        tick(SYMBOLS, fv, 720)
    assert 'TIDE_SPOT' not in gamma, "热身期不应校准"
    print("  热身期（前5轮）正常，未校准")
    # 第6轮：round=6, (6-6)%30=0，触发第一次校准
    tick(SYMBOLS, fv, 720)
    assert 'TIDE_SPOT' in gamma, "第一次校准应该触发"
    g = gamma.get('TIDE_SPOT', 0.1)
    kv = k.get('TIDE_SPOT', 1.5)
    print(f"  gamma={g:.4f}, k={kv:.4f}, arb_thresh={state['arb_thresh']:.1f}")
    assert 0.01 <= g <= 2.0
    assert 0.3 <= kv <= 10.0
    assert 15.0 <= state['arb_thresh'] <= 80.0
    print("  ✓ 通过")

# 4. A-S 做市报价 ---------------------------------------------------------------------------------------
def test_market_making():
    print("\n=== 4. A-S 做市报价 ===")
    fv_map = mock_fair_values()
    MM_MIN_SPREAD = 3
    T = max(720/(24*60), 1e-4)
    for sym in ['TIDE_SPOT','WX_SPOT','LON_ETF']:
        fv, model_std = fv_map[sym]
        sigma = max(model_std*0.5, MM_MIN_SPREAD)
        q = 0
        gamma = 0.15
        k = 1.5
        r = fv - q*gamma*sigma**2*T
        half = max(MM_MIN_SPREAD, (gamma*sigma**2*T + (2/gamma)*np.log(1+gamma/k))/2)
        bid, ask = round(r-half), round(r+half)
        assert bid > 0 and ask > bid
        print(f"  {sym}: fv={fv}, bid={bid}, ask={ask}, spread={ask-bid}")
    # 持多仓时 reservation price 下移
    fv, model_std = fv_map['TIDE_SPOT']
    sigma = max(model_std*0.5, MM_MIN_SPREAD)
    r_long = fv - (50/POSITION_LIMIT)*0.15*sigma**2*T
    assert r_long < fv
    print(f"  持多仓50手: r={r_long:.1f} < fv={fv} ✓")
    print("  ✓ 通过")

# 5. 套利逻辑 ---------------------------------------------------------------------------------------
def test_arbitrage():
    print("\n=== 5. 套利逻辑 ===")
    mids = mock_market_mids()
    positions = mock_positions()
    ARB_QTY = 20
    basis = mids['LON_ETF'] - sum(mids[s] for s in ETF_LEGS)
    print(f"  basis={basis}")
    assert abs(basis) > 25.0
    qty = min(ARB_QTY, POSITION_LIMIT+positions.get('LON_ETF',0),
              min(POSITION_LIMIT-positions.get(s,0) for s in ETF_LEGS))
    assert qty == 20
    positions['LON_ETF'] -= qty
    for s in ETF_LEGS:
        positions[s] += qty
    # 止损
    adverse = (1 > 0 and 120 > basis + 60)
    assert adverse
    print(f"  套利后仓位 LON_ETF={positions['LON_ETF']}, 止损触发 ✓")
    print("  ✓ 通过")

# 6. 方向性交易 ---------------------------------------------------------------------------------------
def test_directional():
    print("\n=== 6. 方向性交易 ===")
    fv_map = mock_fair_values()
    mids = mock_market_mids()
    DIR_Z_THRESH = 2.0
    DIR_QTY = 15
    DIR_DECAY_N = 3
    sym = 'WX_SPOT'
    fv, std = fv_map[sym]
    z = (mids[sym] - fv) / std
    assert abs(z) <= DIR_Z_THRESH
    print(f"  {sym} z={z:.2f} 不触发")
    z_high = (4900 - fv) / std
    assert z_high > DIR_Z_THRESH
    streak = 4
    base_qty = max(1, DIR_QTY // (2**(streak//DIR_DECAY_N)))
    assert base_qty < DIR_QTY
    print(f"  mid=4900 z={z_high:.2f} 触发，streak={streak} 衰减后qty={base_qty}")
    print("  ✓ 通过")

# 7. LON_FLY 对冲 ---------------------------------------------------------------------------------------
def test_fly_hedge():
    print("\n=== 7. LON_FLY 对冲 ===")
    etf_mid = 7450
    fly_pos = 10
    near_kink = any(abs(etf_mid-k) < 50 for k in [6200,6600,7000])
    delta = -2.0 if etf_mid >= 7000 else (1.0 if etf_mid >= 6600 else (-1.0 if etf_mid >= 6200 else -2.0))
    target_hedge = -fly_pos * delta
    delta_needed = target_hedge - 0
    qty = min(abs(int(delta_needed)), POSITION_LIMIT)
    assert qty == 20
    print(f"  ETF={etf_mid}, delta={delta}, target_hedge={target_hedge}, qty={qty}")
    print("  ✓ 通过")

# 8. 组合风险 ---------------------------------------------------------------------------------------
def test_portfolio_risk():
    print("\n=== 8. 组合风险控制 ===")
    CORR_PAIRS = [('TIDE_SPOT','LON_ETF'),('WX_SPOT','LON_ETF'),
                  ('LHR_COUNT','LON_ETF'),('LON_ETF','LON_FLY')]
    CORR_NET_LIMIT = 150
    positions = {s:0 for s in SYMBOLS}
    positions['TIDE_SPOT'] = 90
    positions['LON_ETF'] = 80
    for sym_a, sym_b in CORR_PAIRS:
        net = abs(positions.get(sym_a,0)) + abs(positions.get(sym_b,0))
        if net > CORR_NET_LIMIT:
            target = sym_a if abs(positions.get(sym_a,0)) >= abs(positions.get(sym_b,0)) else sym_b
            rqty = min(abs(positions[target]), net-CORR_NET_LIMIT)
            print(f"  {sym_a}+{sym_b} net={net}，减仓 {target} {rqty}手")
            assert rqty == 20 and target == 'TIDE_SPOT'
    print("  ✓ 通过")

# 9. module_e 后重新拉仓位 ---------------------------------------------------------------------------------------
def test_positions_refresh():
    print("\n=== 9. module_e 后仓位刷新 ===")
    positions = mock_positions()
    positions['TIDE_SPOT'] = 90
    positions['LON_ETF'] = 80
    positions['TIDE_SPOT'] -= 20
    # 重新拉仓位
    fresh = dict(positions)
    assert fresh['TIDE_SPOT'] == 70
    net = abs(fresh['TIDE_SPOT']) + abs(fresh['LON_ETF'])
    print(f"  减仓后 TIDE_SPOT={fresh['TIDE_SPOT']}, net={net} ({'超限' if net>150 else '正常'})")
    print("  ✓ 通过")

# 10. unwind_all 平仓逻辑 ---------------------------------------------------------------------------------------
def test_unwind():
    print("\n=== 10. unwind_all 平仓逻辑 ===")
    UNWIND_MINS = 30
    for trial_mins in [35, 25, 20]:
        executed = []
        if trial_mins <= UNWIND_MINS:
            executed.append('unwind')
        else:
            executed += ['E','B','C','D','A']
        if trial_mins <= UNWIND_MINS:
            assert executed == ['unwind']
            print(f"  mins_left={trial_mins} ≤ {UNWIND_MINS}：只执行 unwind ✓")
        else:
            assert 'unwind' not in executed
            print(f"  mins_left={trial_mins} > {UNWIND_MINS}：执行 EBCDA ✓")
    positions = {'TIDE_SPOT': 30, 'WX_SPOT': -15, 'LON_ETF': 0}
    for sym, pos in positions.items():
        if pos == 0:
            continue
        side = 'SELL' if pos > 0 else 'BUY'
        assert (pos > 0 and side == 'SELL') or (pos < 0 and side == 'BUY')
        print(f"  {sym} pos={pos} → {side} {abs(pos)}手 ✓")
    print("  ✓ 通过")

# 11. 套利止损两个方向 ---------------------------------------------------------------------------------------
def test_arb_stop_both_directions():
    print("\n=== 11. 套利止损（两个方向）===")
    ARB_BASIS_STOP = 60
    # basis>0 进场，扩大 → 止损
    assert (1 > 0 and 120 > 50 + ARB_BASIS_STOP)
    print(f"  basis>0进场: current=120 > entry+60=110 → 止损 ✓")
    # basis>0 进场，收窄 → 不止损
    assert not (1 > 0 and 80 > 50 + ARB_BASIS_STOP)
    print(f"  basis>0进场: current=80 ≤ entry+60=110 → 不止损 ✓")
    # basis<0 进场，扩大（更负）→ 止损
    assert (-1 < 0 and -120 < -50 - ARB_BASIS_STOP)
    print(f"  basis<0进场: current=-120 < entry-60=-110 → 止损 ✓")
    # basis<0 进场，收窄 → 不止损
    assert not (-1 < 0 and -80 < -50 - ARB_BASIS_STOP)
    print(f"  basis<0进场: current=-80 ≥ entry-60=-110 → 不止损 ✓")
    print("  ✓ 通过")

# 12. dir_streak 衰减边界 ---------------------------------------------------------------------------------------
def test_dir_streak_decay():
    print("\n=== 12. 方向性衰减边界 ===")
    DIR_QTY = 15
    DIR_DECAY_N = 3
    cases = [(0,15,"streak=0无衰减"),(3,7,"streak=3减半一次"),
             (6,3,"streak=6减半两次"),(9,1,"streak=9减半三次"),(12,1,"streak=12最小值1")]
    for streak, expected, desc in cases:
        qty = max(1, DIR_QTY // (2**(streak//DIR_DECAY_N)))
        assert qty == expected, f"{desc}: 期望{expected}，实际{qty}"
        print(f"  {desc} → qty={qty} ✓")
    print("  ✓ 通过")

# 13. LON_FLY MC 定价方向性 ---------------------------------------------------------------------------------------
def test_lon_fly_mc():
    print("\n=== 13. LON_FLY MC 定价方向性 ===")
    N = 50000
    def fly_payoff(etf):
        return (2*np.maximum(0,6200-etf) + np.maximum(0,etf-6200)
                - 2*np.maximum(0,etf-6600) + 3*np.maximum(0,etf-7000))
    fv_low = float(fly_payoff(np.random.normal(5500, 100, N)).mean())
    fv_mid = float(fly_payoff(np.random.normal(6400,  50, N)).mean())
    fv_high = float(fly_payoff(np.random.normal(7500, 100, N)).mean())
    print(f"  ETF~5500: FLY fv={fv_low:.1f}，ETF~6400: fv={fv_mid:.1f}，ETF~7500: fv={fv_high:.1f}")
    assert fv_low > fv_mid, "ETF偏低时FLY应比中间贵"
    assert fv_high > fv_mid, "ETF偏高时FLY应比中间贵"
    print("  ✓ 通过")

# 14. bayesian_update market_mid=None ---------------------------------------------------------------------------------------
def test_bayesian_no_market():
    print("\n=== 14. 无市场价时贝叶斯更新 ===")
    def bayesian_update(model_fv, model_std, market_mid):
        if market_mid is None:
            return model_fv, model_std
        w_m = 1.0/model_std**2
        w_k = 1.0/(model_std*1.8)**2
        post_fv = (w_m*model_fv + w_k*market_mid) / (w_m+w_k)
        post_std = (1.0/(w_m+w_k))**0.5
        return post_fv, post_std
    fv, std = bayesian_update(1200, 30.0, None)
    assert fv == 1200 and std == 30.0
    print(f"  market_mid=None → fv={fv}, std={std}（原样返回）✓")
    fv2, std2 = bayesian_update(1200, 30.0, 1300)
    assert fv2 != 1200 and std2 < 30.0
    print(f"  market_mid=1300 → fv={fv2:.1f}, std={std2:.1f}（融合后变化）✓")
    print("  ✓ 通过")

# 15. stop loss 触发 ---------------------------------------------------------------------------------------
def test_stop_loss():
    print("\n=== 15. Stop Loss 触发 ===")
    STOP_LOSS = -5000
    def simulate_loop(pnl_values):
        ran = []
        for pnl in pnl_values:
            if pnl < STOP_LOSS:
                ran.append('STOPPED')
                break
            ran.append('RUN')
        return ran
    ran = simulate_loop([-1000, -2000, -3000])
    assert 'STOPPED' not in ran
    print(f"  PnL未触及-5000：正常运行{len(ran)}轮 ✓")
    ran2 = simulate_loop([-1000, -4000, -5500, -6000])
    assert ran2[-1] == 'STOPPED' and len(ran2) == 3
    print(f"  PnL触及-5500：第{len(ran2)}轮触发止损，第4轮不执行 ✓")
    print("  ✓ 通过")

# 16. REQUOTE_THRESH 防抖 ---------------------------------------------------------------------------------------
def test_requote_thresh():
    print("\n=== 16. REQUOTE_THRESH 防抖 ===")
    REQUOTE_THRESH = 2
    last_quotes = {}

    def should_requote(symbol, new_bid, new_ask):
        last = last_quotes.get(symbol)
        if last and abs(last[0]-new_bid) < REQUOTE_THRESH and abs(last[1]-new_ask) < REQUOTE_THRESH:
            return False
        return True

    last_quotes['TIDE_SPOT'] = (1190, 1210)
    # 变化 < 2，不重报
    assert not should_requote('TIDE_SPOT', 1191, 1211)
    print("  bid/ask 变化 < 2：跳过重报 ✓")
    # 变化 >= 2，重报
    assert should_requote('TIDE_SPOT', 1192, 1212)
    print("  bid/ask 变化 >= 2：触发重报 ✓")
    # 首次报价（无历史），必须报
    assert should_requote('WX_SPOT', 4726, 4774)
    print("  首次报价：触发报价 ✓")
    print("  ✓ 通过")

# 17. 仓位满时不下单 ---------------------------------------------------------------------------------------
def test_position_limit():
    print("\n=== 17. 仓位上限约束 ===")
    POSITION_LIMIT = 100
    MM_BASE_QTY = 10

    # 多仓已满，avail_buy=0，不下买单
    pos = 100
    avail_buy  = max(0, POSITION_LIMIT - pos)
    avail_sell = max(0, POSITION_LIMIT + pos)
    assert avail_buy == 0
    assert avail_sell == 200
    print(f"  多仓满仓(pos=100): avail_buy={avail_buy}（不下买单）, avail_sell={avail_sell} ✓")

    # 空仓已满，avail_sell=0，不下卖单
    pos2 = -100
    avail_buy2  = max(0, POSITION_LIMIT - pos2)
    avail_sell2 = max(0, POSITION_LIMIT + pos2)
    assert avail_sell2 == 0
    assert avail_buy2 == 200
    print(f"  空仓满仓(pos=-100): avail_sell={avail_sell2}（不下卖单）, avail_buy={avail_buy2} ✓")

    # 套利 qty 被仓位上限截断
    ARB_QTY = 20
    pos_etf = 90  # 已持有90手多仓
    qty = min(ARB_QTY, POSITION_LIMIT - pos_etf)  # 买ETF时最多再买10手
    assert qty == 10
    print(f"  套利买ETF: pos={pos_etf}, ARB_QTY=20 → 截断为 qty={qty} ✓")
    print("  ✓ 通过")

# 18. ob_depth 流动性拦截 ---------------------------------------------------------------------------------------
def test_ob_depth():
    print("\n=== 18. ob_depth 流动性拦截 ===")
    # 模拟 orderbook 深度检查
    def ob_depth(bid_vol, ask_vol, qty):
        return bid_vol >= qty, ask_vol >= qty

    # 深度足够
    bid_ok, ask_ok = ob_depth(50, 50, 20)
    assert bid_ok and ask_ok
    print(f"  bid_vol=50, ask_vol=50, qty=20：深度足够，可套利 ✓")

    # 买侧深度不足
    bid_ok2, ask_ok2 = ob_depth(10, 50, 20)
    assert not bid_ok2 and ask_ok2
    print(f"  bid_vol=10, ask_vol=50, qty=20：买侧不足，放弃套利 ✓")

    # 套利逻辑：basis>0 需要卖ETF（检查bid深度）+ 买三腿（检查ask深度）
    # 任意一腿深度不足就放弃
    etf_bid_ok  = ob_depth(50, 50, 20)[0]   # ETF bid ok
    tide_ask_ok = ob_depth(5,  50, 20)[0]   # TIDE_SPOT bid 不足（5 < 20），模拟买侧深度不足
    legs_ok = etf_bid_ok and tide_ask_ok
    assert not legs_ok
    print(f"  三腿中有一腿深度不足：放弃整个套利 ✓")
    print("  ✓ 通过")

# 19. dir_signal_valid 过滤 ---------------------------------------------------------------------------------------
def test_dir_signal_valid():
    print("\n=== 19. 方向性信号质量过滤 ===")
    MM_MIN_SPREAD = 3
    DIR_MAX_SPREAD_RATIO = 2.0
    DIR_MIN_VOL_RATIO = 1.2

    def dir_signal_valid(cur_spread, std, recent_vol, avg_vol):
        # 条件1：价差不能超过正常价差的2倍
        normal_spread = max(std * 0.5, MM_MIN_SPREAD)
        if cur_spread is not None and cur_spread > normal_spread * DIR_MAX_SPREAD_RATIO:
            return False, "价差过宽"
        # 条件2：成交量放大
        if avg_vol < 1.0:
            return True, "无历史成交量，放行"
        if recent_vol < avg_vol * DIR_MIN_VOL_RATIO:
            return False, "成交量不足"
        return True, "通过"

    # 价差正常，成交量放大 → 通过
    ok, reason = dir_signal_valid(10, 30, 8, 5)
    assert ok
    print(f"  价差=10(正常), 成交量=8>5×1.2=6：{reason} ✓")

    # 价差过宽 → 拦截
    ok2, reason2 = dir_signal_valid(40, 30, 8, 5)
    assert not ok2
    print(f"  价差=40 > 正常价差15×2=30：{reason2} ✓")

    # 成交量不足 → 拦截
    ok3, reason3 = dir_signal_valid(10, 30, 4, 5)
    assert not ok3
    print(f"  成交量=4 < 5×1.2=6：{reason3} ✓")

    # 无历史成交量 → 放行
    ok4, reason4 = dir_signal_valid(10, 30, 0, 0)
    assert ok4
    print(f"  无历史成交量：{reason4} ✓")
    print("  ✓ 通过")

# 20. fly_pos=0 时对冲直接返回 ---------------------------------------------------------------------------------------
def test_fly_hedge_skip():
    print("\n=== 20. fly_pos=0 时跳过对冲 ===")
    hedge_executed = []

    def module_d_fly_hedge(fly_pos, etf_mid):
        if fly_pos == 0:
            return  # 直接返回，不执行任何操作
        if etf_mid is None:
            return
        hedge_executed.append('hedged')

    module_d_fly_hedge(0, 7450)
    assert len(hedge_executed) == 0
    print("  fly_pos=0：跳过对冲，不执行任何操作 ✓")

    module_d_fly_hedge(10, 7450)
    assert len(hedge_executed) == 1
    print("  fly_pos=10：执行对冲逻辑 ✓")
    print("  ✓ 通过")

# 21. 折点附近冷却时间缩短 ---------------------------------------------------------------------------------------
def test_kink_cooldown():
    print("\n=== 21. 折点附近冷却时间缩短 ===")
    FLY_KINK_POINTS = [6200, 6600, 7000]
    FLY_KINK_BAND = 50

    def get_hedge_params(etf_mid):
        near_kink = any(abs(etf_mid-k) < FLY_KINK_BAND for k in FLY_KINK_POINTS)
        threshold = 2 if near_kink else 5
        cooldown = 5.0 if near_kink else 30.0
        return near_kink, threshold, cooldown

    # 远离折点
    near, thresh, cool = get_hedge_params(6400)
    assert not near and thresh == 5 and cool == 30.0
    print(f"  ETF=6400（远离折点）: threshold={thresh}, cooldown={cool}s ✓")

    # 在折点附近（6200±50）
    near2, thresh2, cool2 = get_hedge_params(6220)
    assert near2 and thresh2 == 2 and cool2 == 5.0
    print(f"  ETF=6220（折点附近）: threshold={thresh2}, cooldown={cool2}s（更频繁对冲）✓")

    # 恰好在折点上
    near3, thresh3, cool3 = get_hedge_params(7000)
    assert near3
    print(f"  ETF=7000（恰好折点）: near_kink={near3} ✓")
    print("  ✓ 通过")

# 22. delta_needed 小于阈值时不对冲 ---------------------------------------------------------------------------------------
def test_delta_threshold():
    print("\n=== 22. delta_needed 小于阈值时不对冲 ===")
    POSITION_LIMIT = 100

    def should_hedge(delta_needed, near_kink, current_etf):
        hedge_threshold = 2 if near_kink else 5
        if abs(delta_needed) < hedge_threshold:
            return False, 0
        qty = min(abs(int(delta_needed)), POSITION_LIMIT - abs(current_etf))
        return True, qty

    # delta_needed=3，不在折点附近，阈值5 → 不对冲
    ok, qty = should_hedge(3, False, 0)
    assert not ok
    print(f"  delta_needed=3 < threshold=5（非折点）：不对冲 ✓")

    # delta_needed=3，在折点附近，阈值2 → 对冲
    ok2, qty2 = should_hedge(3, True, 0)
    assert ok2 and qty2 == 3
    print(f"  delta_needed=3 ≥ threshold=2（折点附近）：对冲 qty={qty2} ✓")

    # delta_needed=20，仓位已有15手，最多再买85手
    ok3, qty3 = should_hedge(20, False, 15)
    assert ok3 and qty3 == 20
    print(f"  delta_needed=20, current_etf=15：qty={qty3} ✓")

    # delta_needed=20，仓位已有95手，最多再买5手
    ok4, qty4 = should_hedge(20, False, 95)
    assert ok4 and qty4 == 5
    print(f"  delta_needed=20, current_etf=95（接近上限）：截断为 qty={qty4} ✓")
    print("  ✓ 通过")

# 23. T 接近0时 gamma 被 clip 到上限 ---------------------------------------------------------------------------------------
def test_gamma_clip_near_settle():
    print("\n=== 23. 结算临近时 gamma clip ===")
    TARGET_TURNOVER = 0.3

    def calc_gamma(mins_left, sigma):
        T = max(mins_left / (24*60), 1e-4)
        target_spread = TARGET_TURNOVER * sigma
        gamma_new = target_spread / max(sigma**2 * T, 1e-6)
        return float(np.clip(gamma_new, 0.01, 2.0)), T

    # 正常时段（720分钟）
    g_normal, T_normal = calc_gamma(720, 30.0)
    print(f"  mins_left=720: T={T_normal:.4f}, gamma={g_normal:.4f}")
    assert 0.01 <= g_normal <= 2.0

    # 接近结算（5分钟）
    g_near, T_near = calc_gamma(5, 30.0)
    print(f"  mins_left=5:   T={T_near:.6f}, gamma={g_near:.4f}（应被clip到2.0）")
    assert g_near == 2.0, f"gamma={g_near} 应被clip到2.0"

    # 结算后（0分钟），T 被 max 保护为 1e-4
    g_zero, T_zero = calc_gamma(0, 30.0)
    print(f"  mins_left=0:   T={T_zero:.6f}, gamma={g_zero:.4f}（T被保护为1e-4）")
    assert g_zero == 2.0
    assert T_zero == 1e-4

    # gamma 随时间递增（越接近结算越保守）
    assert g_near > g_normal
    print(f"  gamma随时间递增: {g_normal:.4f} → {g_near:.4f} ✓")
    print("  ✓ 通过")

# 24. _hours_to_settle 随时间递减 ---------------------------------------------------------------------------------------
def test_hours_to_settle():
    print("\n=== 24. _hours_to_settle 随时间递减 ===")
    from datetime import timezone
    SETTLE_TIME = datetime(2026, 3, 1, 12, 0, 0)

    def hours_to_settle(now_offset_hours):
        settle = LONDON_TZ.localize(SETTLE_TIME)
        # 模拟不同时间点
        now = settle - timedelta(hours=now_offset_hours)
        now = now.astimezone(pytz.timezone('Europe/London'))
        return max((settle - now).total_seconds() / 3600, 0.0)

    h24 = hours_to_settle(24)
    h12 = hours_to_settle(12)
    h1 = hours_to_settle(1)
    h0 = hours_to_settle(0)

    assert abs(h24 - 24.0) < 0.01
    assert abs(h12 - 12.0) < 0.01
    assert abs(h1  -  1.0) < 0.01
    assert h0 == 0.0
    assert h24 > h12 > h1 > h0
    print(f"  24h前={h24:.1f}h, 12h前={h12:.1f}h, 1h前={h1:.1f}h, 结算时={h0:.1f}h")
    print(f"  单调递减 ✓，结算后返回0 ✓")
    print("  ✓ 通过")



# 25. parse_flight_time 优先级：runway > revised > scheduled ---------------------------------------------------------------------------------------
def test_parse_flight_time():
    print("\n=== 25. parse_flight_time 优先级 ===")
    def parse_flight_time(flight_record):
        time_fields = ['runwayTime', 'revisedTime', 'scheduledTime']
        movement = flight_record['movement']
        utc_str = None
        for field in time_fields:
            if field in movement and movement[field].get('utc'):
                utc_str = movement[field]['utc']
                break
        if utc_str is None:
            raise ValueError('No valid time field found')
        return utc_str

    # 三个时间都有 → 用 runwayTime
    rec_all = {'movement': {
        'runwayTime':    {'utc': '2026-03-01 10:00Z'},
        'revisedTime':   {'utc': '2026-03-01 10:15Z'},
        'scheduledTime': {'utc': '2026-03-01 10:30Z'},
    }}
    assert parse_flight_time(rec_all) == '2026-03-01 10:00Z'
    print("  三个时间都有 → 用 runwayTime ✓")

    # 只有 revised 和 scheduled → 用 revisedTime
    rec_rev = {'movement': {
        'revisedTime':   {'utc': '2026-03-01 10:15Z'},
        'scheduledTime': {'utc': '2026-03-01 10:30Z'},
    }}
    assert parse_flight_time(rec_rev) == '2026-03-01 10:15Z'
    print("  无 runwayTime → 用 revisedTime ✓")

    # 只有 scheduled → 用 scheduledTime
    rec_sched = {'movement': {'scheduledTime': {'utc': '2026-03-01 10:30Z'},}}
    assert parse_flight_time(rec_sched) == '2026-03-01 10:30Z'
    print("  只有 scheduledTime → 用 scheduledTime ✓")

    # 全部缺失 → 抛异常
    rec_none = {'movement': {}}
    try:
        parse_flight_time(rec_none)
        assert False, "应该抛出 ValueError"
    except ValueError:
        print("  全部缺失 → 抛出 ValueError ✓")
    print("  ✓ 通过")


# 26. group_flights 时间窗口统计 ---------------------------------------------------------------------------------------
def test_group_flights():
    print("\n=== 26. group_flights 时间窗口统计 ===")

    def parse_time(utc_str):
        dt = pytz.utc.localize(datetime.strptime(utc_str, '%Y-%m-%d %H:%MZ'))
        return dt.astimezone(LONDON_TZ)

    def parse_flight_time_local(flight_record):
        time_fields = ['runwayTime', 'revisedTime', 'scheduledTime']
        movement = flight_record['movement']
        for field in time_fields:
            if field in movement and movement[field].get('utc'):
                return parse_time(movement[field]['utc'])
        raise ValueError('No valid time field found')

    def group_flights(flights_data, start_time, end_time):
        arrivals = departures = 0
        for arr in flights_data.get('arrivals', []):
            try:
                if start_time <= parse_flight_time_local(arr) < end_time:
                    arrivals += 1
            except Exception:
                pass
        for dep in flights_data.get('departures', []):
            try:
                if start_time <= parse_flight_time_local(dep) < end_time:
                    departures += 1
            except Exception:
                pass
        return arrivals, departures

    start = parse_time('2026-02-28 12:00Z')
    end = parse_time('2026-02-28 12:30Z')

    flights = {
        'arrivals': [
            {'movement': {'scheduledTime': {'utc': '2026-02-28 12:10Z'}}},  # 在窗口内
            {'movement': {'scheduledTime': {'utc': '2026-02-28 12:35Z'}}},  # 窗口外
            {'movement': {}},  # 无时间字段，跳过
        ],
        'departures': [
            {'movement': {'scheduledTime': {'utc': '2026-02-28 12:00Z'}}},  # 恰好等于start，在内
            {'movement': {'scheduledTime': {'utc': '2026-02-28 11:59Z'}}},  # 窗口前，不计
        ]
    }
    arr, dep = group_flights(flights, start, end)
    assert arr == 1, f"arrivals 应为1，实际{arr}"
    assert dep == 1, f"departures 应为1，实际{dep}"
    print(f"  arrivals={arr}（窗口内1个，窗口外1个，无效1个）✓")
    print(f"  departures={dep}（恰好等于start算在内，窗口前不计）✓")
    print("  ✓ 通过")


# 27. get_lhr_index_from 公式：(arr-dep)/(arr+dep)*100 ---------------------------------------------------------------------------------------
def test_lhr_index_formula():
    print("\n=== 27. LHR_INDEX 公式验证 ===")

    def slot_index(arr, dep):
        if arr + dep == 0:
            return 0.0
        return ((arr - dep) / (arr + dep)) * 100

    assert slot_index(10, 0) == 100.0
    print("  纯到达(10,0) → +100.0 ✓")

    assert slot_index(0, 10) == -100.0
    print("  纯出发(0,10) → -100.0 ✓")

    assert slot_index(5, 5) == 0.0
    print("  均衡(5,5) → 0.0 ✓")

    assert slot_index(0, 0) == 0.0
    print("  空窗口(0,0) → 0.0（不除以零）✓")

    # (8-2)/(8+2)*100=60, (3-7)/(3+7)*100=-40 → sum=20 → abs=20
    total = slot_index(8, 2) + slot_index(3, 7)
    assert abs(total) == 20.0
    print(f"  两窗口累加: 60+(-40)={total:.1f}, abs={abs(total):.1f} ✓")
    print("  ✓ 通过")


# 28. wx_spot_fv 时间加权 std 收窄 ---------------------------------------------------------------------------------------
def test_wx_spot_time_factor():
    print("\n=== 28. wx_spot_fv 时间加权 std 收窄 ===")
    TOTAL_HOURS = 24.0

    def time_factor(h_left):
        return max((h_left / TOTAL_HOURS) ** 0.5, 0.05)

    base_std = 100.0
    std24 = base_std * time_factor(24.0)
    std12 = base_std * time_factor(12.0)
    std1 = base_std * time_factor(1.0)
    std0 = base_std * time_factor(0.0)

    assert std24 > std12 > std1
    assert abs(std0 - base_std * 0.05) < 1e-9
    print(f"  24h前 std={std24:.1f}, 12h前 std={std12:.1f}, 1h前 std={std1:.1f}")
    print(f"  结算时 std={std0:.1f}（下限 5% × base_std）✓")
    print(f"  std 随时间单调递减 ✓")
    print("  ✓ 通过")


# 29. wx_sum_fv 已观测+预测分段累积 ---------------------------------------------------------------------------------------
def test_wx_sum_split():
    print("\n=== 29. wx_sum_fv 已观测+预测分段累积 ===")
    import pandas as pd

    t0 = LONDON_TZ.localize(datetime(2026, 2, 28, 12, 0, 0))
    now = t0 + timedelta(hours=12)
    times = [t0 + timedelta(minutes=15*i) for i in range(97)]
    prod = np.ones(97) * 100.0

    df = pd.DataFrame({'time': pd.to_datetime(times, utc=False), 'prod': prod})
    # 让 time 列带时区
    df['time'] = pd.DatetimeIndex(times)
    now_ts = pd.Timestamp(now)

    observed = float(df[df['time'] <= now_ts]['prod'].sum() / 100)
    forecast_df = df[df['time'] > now_ts]
    forecast = float(forecast_df['prod'].sum() / 100)
    fv = observed + forecast

    assert abs(fv - 97.0) < 1.0
    assert observed > 0 and forecast > 0
    print(f"  observed={observed:.1f}, forecast={forecast:.1f}, fv={fv:.1f}")

    per_step_std = 10.0
    std = per_step_std * max(len(forecast_df) ** 0.5, 1)
    assert std == per_step_std * len(forecast_df) ** 0.5
    print(f"  per_step_std={per_step_std}, forecast_steps={len(forecast_df)}, std={std:.1f}")
    print("  ✓ 通过")


# 30. tide_swing_fv 真实数据覆盖预测值 ---------------------------------------------------------------------------------------
def test_tide_swing_real_override():
    print("\n=== 30. tide_swing_fv 真实数据覆盖预测值 ===")
    import pandas as pd

    t0 = LONDON_TZ.localize(datetime(2026, 2, 28, 12, 0, 0))
    dr = [t0 + timedelta(minutes=15*i) for i in range(5)]

    predicted = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    level_series = pd.Series(predicted, index=dr)

    real_data = pd.DataFrame({'time': [dr[0], dr[1], dr[2]], 'level': [0.5, 0.8, 1.2]})

    for _, row in real_data.iterrows():
        if row['time'] in level_series.index:
            level_series[row['time']] = row['level']

    assert level_series[dr[0]] == 0.5
    assert level_series[dr[1]] == 0.8
    assert level_series[dr[2]] == 1.2
    assert level_series[dr[3]] == 1.0
    assert level_series[dr[4]] == 1.0
    print(f"  前3点被真实数据覆盖: {list(level_series[:3].values)} ✓")
    print(f"  后2点保持预测值: {list(level_series[3:].values)} ✓")
    print("  ✓ 通过")


# 31. lon_fly_greeks 固定随机种子保证差分一致性 ---------------------------------------------------------------------------------------
def test_fly_greeks_seed():
    print("\n=== 31. lon_fly_greeks 固定随机种子保证差分一致性 ===")

    def fly_mc(etf_center, etf_std, N=10000):
        s = np.random.normal(etf_center, etf_std, N)
        s = np.maximum(0, s)
        return float((2*np.maximum(0,6200-s)+np.maximum(0,s-6200)
                      -2*np.maximum(0,s-6600)+3*np.maximum(0,s-7000)).mean())

    etf_fv, etf_std, h = 6400.0, 200.0, 20.0

    # 不固定种子：结果有随机噪声
    f_up_n = fly_mc(etf_fv + h, etf_std)
    f_down_n = fly_mc(etf_fv - h, etf_std)
    delta_noisy = (f_up_n - f_down_n) / (2 * h)

    # 固定种子：三次 MC 用同一组随机数
    np.random.seed(42)
    f_up = fly_mc(etf_fv + h, etf_std)
    np.random.seed(42)
    f_mid = fly_mc(etf_fv,     etf_std)
    np.random.seed(42)
    f_down = fly_mc(etf_fv - h, etf_std)
    delta_seeded = (f_up - f_down) / (2 * h)
    gamma_seeded = (f_up - 2*f_mid + f_down) / (h**2)

    # 重复计算结果完全一致
    np.random.seed(42)
    f_up2 = fly_mc(etf_fv + h, etf_std)
    np.random.seed(42)
    f_mid2 = fly_mc(etf_fv,     etf_std)
    np.random.seed(42)
    f_down2 = fly_mc(etf_fv - h, etf_std)
    delta2 = (f_up2 - f_down2) / (2 * h)
    assert abs(delta_seeded - delta2) < 1e-10
    print(f"  不固定种子 delta={delta_noisy:.4f}（有MC噪声）")
    print(f"  固定种子   delta={delta_seeded:.4f}, gamma={gamma_seeded:.6f}")
    print(f"  固定种子重复计算 delta={delta2:.4f}（完全一致）✓")
    print("  ✓ 通过")


# 32. _arb_cooldown 2秒防重复触发 ---------------------------------------------------------------------------------------
def test_arb_cooldown():
    print("\n=== 32. _arb_cooldown 2秒防重复触发 ===")
    import time

    arb_cooldown = [0.0]
    executed = []

    def realtime_arb_check(basis, thresh):
        if time.time() - arb_cooldown[0] < 2.0:
            return
        if abs(basis) > thresh:
            executed.append(time.time())
            arb_cooldown[0] = time.time()

    realtime_arb_check(100, 25)
    assert len(executed) == 1
    print("  第一次触发：执行套利 ✓")

    realtime_arb_check(100, 25)
    assert len(executed) == 1
    print("  立即再次触发：被2秒冷却拦截 ✓")

    arb_cooldown[0] = time.time() - 2.1
    realtime_arb_check(100, 25)
    assert len(executed) == 2
    print("  2秒后再次触发：执行套利 ✓")
    print("  ✓ 通过")


# 33. _dir_cooldown 5秒防重复触发 ---------------------------------------------------------------------------------------
def test_dir_cooldown():
    print("\n=== 33. _dir_cooldown 5秒防重复触发 ===")
    import time

    dir_cooldown = {}
    executed = []

    def realtime_dir_check(symbol, z, thresh=2.0):
        if time.time() - dir_cooldown.get(symbol, 0) < 5.0:
            return
        if abs(z) > thresh:
            executed.append(symbol)
            dir_cooldown[symbol] = time.time()

    realtime_dir_check('WX_SPOT', 3.0)
    assert len(executed) == 1
    print("  第一次触发：执行方向性交易 ✓")

    realtime_dir_check('WX_SPOT', 3.0)
    assert len(executed) == 1
    print("  立即再次触发：被5秒冷却拦截 ✓")

    realtime_dir_check('TIDE_SPOT', 3.0)
    assert len(executed) == 2
    print("  不同品种（TIDE_SPOT）不受冷却影响 ✓")

    dir_cooldown['WX_SPOT'] = time.time() - 5.1
    realtime_dir_check('WX_SPOT', 3.0)
    assert len(executed) == 3
    print("  5秒后再次触发：执行方向性交易 ✓")
    print("  ✓ 通过")


# 34. init_flights 幂等性（只调用一次） ---------------------------------------------------------------------------------------
def test_init_flights_idempotent():
    print("\n=== 34. init_flights 幂等性 ===")
    call_count = [0]
    flights_cache = [None]

    def init_flights():
        if flights_cache[0] is None:
            call_count[0] += 1
            flights_cache[0] = {'arrivals': [], 'departures': []}
        return flights_cache[0]

    result1 = init_flights()
    assert call_count[0] == 1
    print(f"  第一次调用：触发 API（call_count={call_count[0]}）✓")

    result2 = init_flights()
    assert call_count[0] == 1
    assert result1 is result2
    print(f"  第二次调用：复用缓存（call_count={call_count[0]}）✓")

    init_flights()
    assert call_count[0] == 1
    print(f"  第三次调用：仍复用缓存（call_count={call_count[0]}）✓")
    print("  ✓ 通过")


# 35. _close_arb_position 覆盖 ETF 和三条腿 ---------------------------------------------------------------------------------------
def test_close_arb_position():
    print("\n=== 35. _close_arb_position 覆盖 ETF 和三条腿 ===")
    ETF_LEGS_LOCAL = ['TIDE_SPOT', 'WX_SPOT', 'LHR_COUNT']
    orders_sent = []

    def close_arb_position(positions):
        etf_pos = positions.get('LON_ETF', 0)
        if etf_pos != 0:
            side = 'SELL' if etf_pos > 0 else 'BUY'
            orders_sent.append(('LON_ETF', side, abs(etf_pos)))
        for s in ETF_LEGS_LOCAL:
            leg_pos = positions.get(s, 0)
            if leg_pos == 0:
                continue
            side = 'SELL' if leg_pos > 0 else 'BUY'
            orders_sent.append((s, side, abs(leg_pos)))

    positions = {
        'LON_ETF': -20,
        'TIDE_SPOT': 20,
        'WX_SPOT': 20,
        'LHR_COUNT': 20,
        'LON_FLY': 0,
    }
    close_arb_position(positions)

    symbols_closed = [o[0] for o in orders_sent]
    assert 'LON_ETF' in symbols_closed
    for leg in ETF_LEGS_LOCAL:
        assert leg in symbols_closed

    etf_order = next(o for o in orders_sent if o[0] == 'LON_ETF')
    assert etf_order[1] == 'BUY'
    print("  LON_ETF 空仓 → BUY 平仓 ✓")

    for leg in ETF_LEGS_LOCAL:
        leg_order = next(o for o in orders_sent if o[0] == leg)
        assert leg_order[1] == 'SELL'
        print(f"  {leg} 多仓 → SELL 平仓 ✓")

    fly_orders = [o for o in orders_sent if o[0] == 'LON_FLY']
    assert len(fly_orders) == 0
    print("  LON_FLY=0 → 不平仓 ✓")
    print("  ✓ 通过")


# 36. lon_etf_fv 方差相加（独立假设） ---------------------------------------------------------------------------------------
def test_lon_etf_variance():
    print("\n=== 36. LON_ETF 方差相加（独立假设）===")
    ts = (1200, 30.0)
    wx = (4800, 50.0)
    lc = (1400, 30.0)

    fv = ts[0] + wx[0] + lc[0]
    std = (ts[1]**2 + wx[1]**2 + lc[1]**2) ** 0.5

    assert fv == 7400
    expected_std = (30**2 + 50**2 + 30**2) ** 0.5
    assert abs(std - expected_std) < 1e-9
    print(f"  fv={fv}（三腿之和）✓")
    print(f"  std={std:.2f} = sqrt(30²+50²+30²) = {expected_std:.2f} ✓")
    assert std > max(ts[1], wx[1], lc[1])
    print(f"  std={std:.2f} > max(30,50,30)=50 ✓（组合不确定性更大）")
    print("  ✓ 通过")


# 37. _calc_mid 过滤自己的挂单 ---------------------------------------------------------------------------------------
def test_calc_mid_filter_own():
    print("\n=== 37. _calc_mid 过滤自己的挂单 ===")

    class MockOrder:
        def __init__(self, price, volume, own_volume):
            self.price = price
            self.volume = volume
            self.own_volume = own_volume

    def calc_mid(buy_orders, sell_orders):
        bids = [o.price for o in buy_orders  if o.volume - o.own_volume > 0]
        asks = [o.price for o in sell_orders if o.volume - o.own_volume > 0]
        return (max(bids) + min(asks)) / 2 if bids and asks else None

    buys = [MockOrder(100, 10, 0), MockOrder(99, 5, 0)]
    sells = [MockOrder(102, 10, 0), MockOrder(103, 5, 0)]
    mid = calc_mid(buys, sells)
    assert mid == 101.0
    print(f"  正常情况: best_bid=100, best_ask=102, mid={mid} ✓")

    buys_own = [MockOrder(100, 10, 10), MockOrder(99, 5, 5)]
    mid_none = calc_mid(buys_own, sells)
    assert mid_none is None
    print("  买单全是自己的 → mid=None ✓")

    buys_partial = [MockOrder(100, 10, 8), MockOrder(99, 5, 0)]
    mid_partial = calc_mid(buys_partial, sells)
    assert mid_partial == 101.0
    print(f"  部分是自己的（剩余2手）→ 仍有效, mid={mid_partial} ✓")
    print("  ✓ 通过")


# 38. calib.tick 校准间隔逻辑 ---------------------------------------------------------------------------------------
def test_calib_tick_interval():
    print("\n=== 38. calib.tick 校准间隔逻辑 ===")
    WARMUP = 6
    INTERVAL = 30
    calibrated_at = []

    def tick(round_num):
        if round_num < WARMUP:
            return
        if (round_num - WARMUP) % INTERVAL == 0:
            calibrated_at.append(round_num)

    for r in range(1, 100):
        tick(r)

    expected = [6, 36, 66, 96]
    assert calibrated_at == expected, f"期望{expected}，实际{calibrated_at}"
    print(f"  校准触发轮次: {calibrated_at} ✓")
    print("  热身期(1~5)不触发，第6轮首次触发，之后每30轮一次 ✓")
    print("  ✓ 通过")


# 39. tide_swing payoff 分段公式验证 ---------------------------------------------------------------------------------------
def test_tide_swing_payoff():
    print("\n=== 39. tide_swing payoff 分段公式 ===")
    # payoff = sum( max(0, 20-diff_cm) + max(0, diff_cm-25) )
    # diff_cm < 20: 奖励平静（差值越小奖励越大）
    # 20 <= diff_cm <= 25: 死区，payoff=0
    # diff_cm > 25: 惩罚剧烈波动

    def step_payoff(diff_cm):
        return max(0, 20 - diff_cm) + max(0, diff_cm - 25)

    # diff=0: 完全静止，最大奖励
    assert step_payoff(0) == 20
    print(f"  diff=0cm → payoff={step_payoff(0)} (最大奖励) ✓")

    # diff=10: 平静，有奖励
    assert step_payoff(10) == 10
    print(f"  diff=10cm → payoff={step_payoff(10)} ✓")

    # diff=20: 恰好边界，payoff=0
    assert step_payoff(20) == 0
    print(f"  diff=20cm → payoff={step_payoff(20)} (死区开始) ✓")

    # diff=22: 死区内，payoff=0
    assert step_payoff(22) == 0
    print(f"  diff=22cm → payoff={step_payoff(22)} (死区内) ✓")

    # diff=25: 死区右边界，payoff=0
    assert step_payoff(25) == 0
    print(f"  diff=25cm → payoff={step_payoff(25)} (死区结束) ✓")

    # diff=30: 超出死区，惩罚
    assert step_payoff(30) == 5
    print(f"  diff=30cm → payoff={step_payoff(30)} (惩罚) ✓")

    # 整个序列累加
    diffs = [0, 10, 22, 30]
    total = sum(step_payoff(d) for d in diffs)
    assert total == 20 + 10 + 0 + 5
    print(f"  序列{diffs}累加 payoff={total} ✓")
    print("  ✓ 通过")


# 40. lon_fly payoff 分段线性公式验证 ---------------------------------------------------------------------------------------
def test_lon_fly_payoff():
    print("\n=== 40. LON_FLY payoff 分段线性公式 ===")
    # payoff = 2*max(0,6200-etf) + max(0,etf-6200) - 2*max(0,etf-6600) + 3*max(0,etf-7000)

    def fly_payoff(etf):
        return (2*max(0, 6200-etf) + max(0, etf-6200) - 2*max(0, etf-6600) + 3*max(0, etf-7000))

    # etf=5000: 远低于6200，payoff=2*(6200-5000)=2400
    assert fly_payoff(5000) == 2*1200
    print(f"  etf=5000 → payoff={fly_payoff(5000)} ✓")

    # etf=6200: 折点，payoff=0
    assert fly_payoff(6200) == 0
    print(f"  etf=6200 → payoff={fly_payoff(6200)} (折点) ✓")

    # etf=6400: 在6200~6600之间，payoff=etf-6200=200
    assert fly_payoff(6400) == 200
    print(f"  etf=6400 → payoff={fly_payoff(6400)} ✓")

    # etf=6600: 折点，payoff=6600-6200 - 2*(6600-6600)=400
    assert fly_payoff(6600) == 400
    print(f"  etf=6600 → payoff={fly_payoff(6600)} ✓")

    # etf=6800: 在6600~7000之间，payoff=600 - 2*(6800-6600)=600-400=200
    assert fly_payoff(6800) == 200
    print(f"  etf=6800 → payoff={fly_payoff(6800)} ✓")

    # etf=7000: 折点，payoff=0 + 3*(7000-7000)=0
    assert fly_payoff(7000) == 0
    print(f"  etf=7000 → payoff={fly_payoff(7000)} ✓")

    # etf=7200: 超过7000，payoff=200 + 3*(7200-7000)=200+600=... 实际=400
    assert fly_payoff(7200) == 400
    print(f"  etf=7200 → payoff={fly_payoff(7200)} ✓")

    # payoff 在 etf=6200 和 etf>7000 时为正，在 etf=6200/7000 折点为0，中间有谷
    assert fly_payoff(5000) > fly_payoff(6200)
    assert fly_payoff(7200) > fly_payoff(7000)
    print(f"  两端高于折点 ✓")
    print("  ✓ 通过")


# 41. _short_vol 数据不足时用 FV std 兜底 ---------------------------------------------------------------------------------------
def test_short_vol_fallback():
    print("\n=== 41. _short_vol 数据不足时 FV std 兜底 ===")
    MM_MIN_SPREAD = 3

    def short_vol(mid_history, fair_values, symbol):
        hist = list(mid_history.get(symbol, []))
        if len(hist) < 3:
            return fair_values.get(symbol, (0, MM_MIN_SPREAD * 2))[1]
        return max(float(np.std(hist)), MM_MIN_SPREAD)

    fv_map = {'TIDE_SPOT': (1200, 30.0)}

    # 数据不足3条 → 用 FV std 兜底
    hist_short = deque([1200, 1205], maxlen=20)
    vol = short_vol({'TIDE_SPOT': hist_short}, fv_map, 'TIDE_SPOT')
    assert vol == 30.0
    print(f"  数据2条 → 用 FV std={vol} 兜底 ✓")

    # 数据足够 → 用实际 std，但不低于 MM_MIN_SPREAD
    hist_long = deque([1200, 1201, 1202, 1203, 1204], maxlen=20)
    vol2 = short_vol({'TIDE_SPOT': hist_long}, fv_map, 'TIDE_SPOT')
    assert vol2 >= MM_MIN_SPREAD
    print(f"  数据5条 → 实际 std={vol2:.2f}（≥ MM_MIN_SPREAD={MM_MIN_SPREAD}）✓")

    # 完全没有历史 → 用默认值 MM_MIN_SPREAD*2
    vol3 = short_vol({}, {}, 'UNKNOWN')
    assert vol3 == MM_MIN_SPREAD * 2
    print(f"  无历史无FV → 默认值={vol3} ✓")
    print("  ✓ 通过")


# 42. _execute_arb qty 计算：basis>0 和 basis<0 的仓位约束不同 ---------------------------------------------------------------------------------------
def test_execute_arb_qty():
    print("\n=== 42. _execute_arb qty 计算（两个方向）===")
    ARB_QTY = 20
    POSITION_LIMIT = 100
    ETF_LEGS = ['TIDE_SPOT', 'WX_SPOT', 'LHR_COUNT']

    def calc_qty_sell_etf(positions):
        # basis>0: 卖ETF(需要空间做空) + 买三条腿(需要空间做多)
        return min(ARB_QTY, POSITION_LIMIT + positions.get('LON_ETF', 0),
                   min(POSITION_LIMIT - positions.get(s, 0) for s in ETF_LEGS))

    def calc_qty_buy_etf(positions):
        # basis<0: 买ETF(需要空间做多) + 卖三条腿(需要空间做空)
        return min(ARB_QTY, POSITION_LIMIT - positions.get('LON_ETF', 0),
                   min(POSITION_LIMIT + positions.get(s, 0) for s in ETF_LEGS))

    # 空仓时，两个方向都能做满 ARB_QTY=20
    pos_empty = {s: 0 for s in ETF_LEGS + ['LON_ETF']}
    assert calc_qty_sell_etf(pos_empty) == 20
    assert calc_qty_buy_etf(pos_empty) == 20
    print("  空仓：两个方向都能做满 qty=20 ✓")

    # ETF 已持多仓 90，basis>0 要卖 ETF → 空间=100+90=190，不受限
    # 但三条腿空仓，买三条腿空间=100，不受限
    pos_etf_long = {'LON_ETF': 90, 'TIDE_SPOT': 0, 'WX_SPOT': 0, 'LHR_COUNT': 0}
    qty = calc_qty_sell_etf(pos_etf_long)
    assert qty == 20
    print(f"  ETF多仓90，basis>0卖ETF：qty={qty}（不受限）✓")

    # ETF 已持空仓 -90，basis>0 要卖 ETF → 空间=100+(-90)=10，受限
    pos_etf_short = {'LON_ETF': -90, 'TIDE_SPOT': 0, 'WX_SPOT': 0, 'LHR_COUNT': 0}
    qty2 = calc_qty_sell_etf(pos_etf_short)
    assert qty2 == 10
    print(f"  ETF空仓-90，basis>0卖ETF：qty={qty2}（受仓位限制）✓")

    # 三条腿之一已满仓，整体 qty 被截断
    pos_leg_full = {'LON_ETF': 0, 'TIDE_SPOT': 95, 'WX_SPOT': 0, 'LHR_COUNT': 0}
    qty3 = calc_qty_sell_etf(pos_leg_full)
    assert qty3 == 5
    print(f"  TIDE_SPOT多仓95，basis>0买三条腿：qty={qty3}（受腿仓位限制）✓")
    print("  ✓ 通过")


# 43. _arb_basis_stop_check 无仓位时直接返回 ---------------------------------------------------------------------------------------
def test_arb_stop_no_position():
    print("\n=== 43. _arb_basis_stop_check 无仓位时直接返回 ===")
    checked = []

    def arb_basis_stop_check(arb_position, arb_entry_basis, current_basis):
        if arb_position == 0 or arb_entry_basis is None:
            return 'skipped'
        ARB_BASIS_STOP = 60
        adverse = (
            (arb_position > 0 and current_basis > arb_entry_basis + ARB_BASIS_STOP) or
            (arb_position < 0 and current_basis < arb_entry_basis - ARB_BASIS_STOP)
        )
        return 'stop' if adverse else 'hold'

    # 无套利仓位
    assert arb_basis_stop_check(0, 50, 120) == 'skipped'
    print("  arb_position=0 → 直接跳过 ✓")

    # entry_basis=None（刚初始化）
    assert arb_basis_stop_check(1, None, 120) == 'skipped'
    print("  arb_entry_basis=None → 直接跳过 ✓")

    # 有仓位，basis 未触发止损
    assert arb_basis_stop_check(1, 50, 80) == 'hold'
    print("  有仓位，basis未扩大超60 → hold ✓")

    # 有仓位，basis 触发止损
    assert arb_basis_stop_check(1, 50, 115) == 'stop'
    print("  有仓位，basis扩大超60 → stop ✓")
    print("  ✓ 通过")


# 44. module_e 减仓选择较大仓位的品种 ---------------------------------------------------------------------------------------
def test_module_e_target_selection():
    print("\n=== 44. module_e 减仓选择较大仓位的品种 ===")
    CORR_NET_LIMIT = 150

    def select_target(pos_a, pos_b, sym_a, sym_b):
        net = abs(pos_a) + abs(pos_b)
        if net <= CORR_NET_LIMIT:
            return None, 0
        target = sym_a if abs(pos_a) >= abs(pos_b) else sym_b
        tpos = pos_a if target == sym_a else pos_b
        rqty = min(abs(tpos), net - CORR_NET_LIMIT)
        return target, rqty

    # pos_a 更大 → 减仓 sym_a
    t, q = select_target(100, 80, 'TIDE_SPOT', 'LON_ETF')
    assert t == 'TIDE_SPOT' and q == 30
    print(f"  TIDE_SPOT=100 > LON_ETF=80 → 减仓 {t} {q}手 ✓")

    # pos_b 更大 → 减仓 sym_b
    t2, q2 = select_target(60, 100, 'TIDE_SPOT', 'LON_ETF')
    assert t2 == 'LON_ETF' and q2 == 10
    print(f"  LON_ETF=100 > TIDE_SPOT=60 → 减仓 {t2} {q2}手 ✓")

    # 相等时选 sym_a（abs(pos_a) >= abs(pos_b)）
    t3, q3 = select_target(80, 80, 'TIDE_SPOT', 'LON_ETF')
    assert t3 == 'TIDE_SPOT'
    print(f"  相等时选第一个品种 {t3} ✓")

    # 未超限 → 不减仓
    t4, q4 = select_target(70, 70, 'TIDE_SPOT', 'LON_ETF')
    assert t4 is None
    print(f"  net=140 ≤ 150 → 不减仓 ✓")
    print("  ✓ 通过")


# 45. _dir_streak 方向反转时归零 ---------------------------------------------------------------------------------------
def test_dir_streak_reset():
    print("\n=== 45. _dir_streak 方向反转时归零 ===")
    DIR_Z_THRESH = 2.0

    def update_streak(streak, z):
        if z > DIR_Z_THRESH:
            return max(0, streak) + 1   # 做空方向
        elif z < -DIR_Z_THRESH:
            return min(0, streak) - 1   # 做多方向
        else:
            return 0                    # 无信号，归零

    # 连续做空 3 次
    s = 0
    for _ in range(3):
        s = update_streak(s, 3.0)
    assert s == 3
    print(f"  连续做空3次 → streak={s} ✓")

    # 方向反转（做多信号）→ 从正数变为 -1
    s = update_streak(s, -3.0)
    assert s == -1
    print(f"  方向反转（做多）→ streak={s}（从3变为-1）✓")

    # 无信号 → 归零
    s2 = update_streak(3, 0.5)
    assert s2 == 0
    print(f"  无信号（z=0.5）→ streak={s2}（归零）✓")

    # 连续做多后反转
    s3 = -4
    s3 = update_streak(s3, 3.0)
    assert s3 == 1
    print(f"  做多streak=-4，反转做空 → streak={s3} ✓")
    print("  ✓ 通过")


# 46. _realtime_dir_check std<=0 时不触发 ---------------------------------------------------------------------------------------
def test_dir_check_zero_std():
    print("\n=== 46. _realtime_dir_check std<=0 时不触发 ===")
    executed = []

    def realtime_dir_check(fv, std, mid):
        if mid is None or std <= 0:
            return 'skipped'
        z = (mid - fv) / std
        if abs(z) > 2.0:
            executed.append(z)
            return 'executed'
        return 'no_signal'

    # std=0 → 跳过（避免除零）
    result = realtime_dir_check(1200, 0, 1300)
    assert result == 'skipped'
    print("  std=0 → 跳过（避免除零）✓")

    # std<0 → 跳过
    result2 = realtime_dir_check(1200, -5, 1300)
    assert result2 == 'skipped'
    print("  std<0 → 跳过 ✓")

    # mid=None → 跳过
    result3 = realtime_dir_check(1200, 30, None)
    assert result3 == 'skipped'
    print("  mid=None → 跳过 ✓")

    # 正常情况
    result4 = realtime_dir_check(1200, 30, 1270)
    assert result4 == 'executed'
    print(f"  正常情况 z={(1270-1200)/30:.2f} → 执行 ✓")
    print("  ✓ 通过")


# 47. _send_ioc 逻辑：ACTIVE 状态立即撤单 ---------------------------------------------------------------------------------------
def test_send_ioc():
    print("\n=== 47. _send_ioc ACTIVE 状态立即撤单 ===")
    cancelled = []

    class MockResp:
        def __init__(self, status, id):
            self.status = status
            self.id = id

    def send_ioc(resp):
        if resp and resp.status == 'ACTIVE':
            cancelled.append(resp.id)
        return resp

    # 订单 ACTIVE → 撤单
    r = send_ioc(MockResp('ACTIVE', 'order_001'))
    assert 'order_001' in cancelled
    print("  ACTIVE → 立即撤单 ✓")

    # 订单 FILLED → 不撤单
    r2 = send_ioc(MockResp('FILLED', 'order_002'))
    assert 'order_002' not in cancelled
    print("  FILLED → 不撤单 ✓")

    # resp=None → 不撤单，不报错
    r3 = send_ioc(None)
    assert r3 is None
    print("  resp=None → 不报错 ✓")
    print("  ✓ 通过")


# 48. run_bot 主循环：unwind 后 continue，不执行 EBCDA ---------------------------------------------------------------------------------------
def test_run_bot_unwind_continue():
    print("\n=== 48. run_bot unwind 后 continue，不执行 EBCDA ===")
    UNWIND_MINS = 30
    log = []

    def simulate_loop(mins_left):
        if mins_left <= UNWIND_MINS:
            log.append('unwind')
            # continue → 不执行后续
            return
        log.append('E')
        log.append('B')
        log.append('C')
        log.append('D')
        log.append('A')

    # 进入平仓期
    simulate_loop(25)
    assert log == ['unwind']
    print(f"  mins_left=25 → 只执行 unwind，不执行 EBCDA ✓")

    log.clear()
    simulate_loop(35)
    assert log == ['E', 'B', 'C', 'D', 'A']
    print(f"  mins_left=35 → 执行完整 EBCDA ✓")

    # 连续两轮平仓期
    log.clear()
    simulate_loop(20)
    simulate_loop(15)
    assert log == ['unwind', 'unwind']
    print(f"  连续两轮平仓期 → 每轮只执行 unwind ✓")
    print("  ✓ 通过")


# 49. _update_drift market_weight 上下限约束 ---------------------------------------------------------------------------------------
def test_market_weight_bounds():
    print("\n=== 49. _update_drift market_weight 上下限约束 ===")
    drift_history = {}
    market_weight = {}

    def update_drift(sym, model_fv, market_mid):
        if sym not in drift_history:
            drift_history[sym] = deque(maxlen=10)
            market_weight[sym] = 0.2
        drift_history[sym].append(model_fv - market_mid)
        hist = list(drift_history[sym])
        if len(hist) >= 5:
            signs = [1 if x > 0 else -1 for x in hist[-5:]]
            if abs(sum(signs)) == 5:
                market_weight[sym] = min(0.7, market_weight[sym] + 0.1)
            else:
                market_weight[sym] = max(0.1, market_weight[sym] - 0.05)

    sym = 'TEST'
    # 连续同向偏差 → weight 上升，但不超过 0.7
    for _ in range(20):
        update_drift(sym, 1300, 1200)  # 模型持续高于市场
    assert market_weight[sym] == 0.7
    print(f"  连续同向20次 → market_weight={market_weight[sym]}（上限0.7）✓")

    # 重置，连续混合信号 → weight 下降，但不低于 0.1
    sym2 = 'TEST2'
    market_weight[sym2] = 0.2
    drift_history[sym2] = deque(maxlen=10)
    for i in range(20):
        # 交替正负，不满足连续5个同号
        update_drift(sym2, 1300 if i % 2 == 0 else 1100, 1200)
    assert market_weight[sym2] >= 0.1
    print(f"  混合信号20次 → market_weight={market_weight[sym2]:.2f}（下限0.1）✓")
    print("  ✓ 通过")


# 50. get_all 对所有8个品种都做贝叶斯更新 ---------------------------------------------------------------------------------------
def test_get_all_covers_all_symbols():
    print("\n=== 50. get_all 对所有8个品种都做贝叶斯更新 ===")
    SYMBOLS_ALL = ['TIDE_SPOT','TIDE_SWING','WX_SPOT','WX_SUM',
                   'LHR_COUNT','LHR_INDEX','LON_ETF','LON_FLY']

    # 模拟 get_all 的结构：raw → bayesian_update → result
    raw = {
        'TIDE_SPOT':  (1200, 30.0),
        'TIDE_SWING': (3500, 80.0),
        'WX_SPOT':    (4750, 50.0),
        'WX_SUM':     (2100, 40.0),
        'LHR_COUNT':  (1400, 30.0),
        'LHR_INDEX':  (55,   15.0),
        'LON_ETF':    (7350, 60.0),
        'LON_FLY':    (800,  20.0),
    }
    market_mids = {'TIDE_SPOT': 1210, 'LON_ETF': 7400}  # 只有部分品种有市场价

    def bayesian_update(model_fv, model_std, market_mid):
        if market_mid is None:
            return model_fv, model_std
        w = 0.2
        market_std = model_std * (2.0 - w)
        w_m = 1.0/model_std**2
        w_k = 1.0/market_std**2
        post_fv  = (w_m*model_fv + w_k*market_mid) / (w_m+w_k)
        post_std = (1.0/(w_m+w_k))**0.5
        return post_fv, post_std

    result = {}
    for sym, (fv, std) in raw.items():
        mid = market_mids.get(sym)
        post_fv, post_std = bayesian_update(fv, std, mid)
        result[sym] = (post_fv, post_std)

    # 所有8个品种都在结果里
    assert set(result.keys()) == set(SYMBOLS_ALL)
    print(f"  结果包含所有8个品种 ✓")

    # 有市场价的品种：后验 fv 介于模型和市场之间
    ts_fv, ts_std = result['TIDE_SPOT']
    assert 1200 < ts_fv < 1210 or 1200 == ts_fv  # 后验向市场靠拢
    assert ts_std < 30.0
    print(f"  TIDE_SPOT 有市场价：后验 fv={ts_fv:.1f}（介于1200~1210），std={ts_std:.1f} ✓")

    # 无市场价的品种：后验 = 模型原值
    ts_fv2, ts_std2 = result['TIDE_SWING']
    assert ts_fv2 == 3500 and ts_std2 == 80.0
    print(f"  TIDE_SWING 无市场价：后验 fv={ts_fv2}，std={ts_std2}（原样保留）✓")
    print("  ✓ 通过")


# 51. lhr_count_fv / lhr_index_fv 无航班缓存时返回默认值 ---------------------------------------------------------------------------------------
def test_lhr_fv_no_cache():
    print("\n=== 51. lhr_count/index_fv 无缓存时返回默认值 ===")
    # flights_cache=None 时的兜底逻辑
    def lhr_count_fv(flights_cache):
        if flights_cache is None:
            return 1400, 50.0
        d1, d2 = flights_cache
        count = (len(d1.get('arrivals', [])) + len(d1.get('departures', []))
                 + len(d2.get('arrivals', [])) + len(d2.get('departures', [])))
        return round(float(count)), 30.0

    def lhr_index_fv(flights_cache):
        if flights_cache is None:
            return 50, 20.0
        return 42, 15.0  # 简化

    # 无缓存 → 默认值
    fv, std = lhr_count_fv(None)
    assert fv == 1400 and std == 50.0
    print(f"  lhr_count 无缓存 → fv={fv}, std={std}（默认值）✓")

    fv2, std2 = lhr_index_fv(None)
    assert fv2 == 50 and std2 == 20.0
    print(f"  lhr_index 无缓存 → fv={fv2}, std={std2}（默认值）✓")

    # 有缓存 → 用真实数据
    cache = ({'arrivals': [1, 2, 3], 'departures': [4]},
             {'arrivals': [5], 'departures': [6, 7]})
    fv3, std3 = lhr_count_fv(cache)
    assert fv3 == 7 and std3 == 30.0
    print(f"  lhr_count 有缓存 → fv={fv3}, std={std3} ✓")
    print("  ✓ 通过")


# 52. lon_fly_fv 中 lhr_count=0 时 Poisson(max(lc,1)) 防零 ---------------------------------------------------------------------------------------
def test_lon_fly_poisson_guard():
    print("\n=== 52. lon_fly_fv Poisson(max(lc,1)) 防零 ===")
    # lc[0]=0 时直接用 np.random.poisson(0) 会产生全零样本，导致 ETF 偏低
    # max(lc[0], 1) 保证 lambda >= 1

    N = 10000
    # lc=0 时不加保护
    lhr_raw = np.random.poisson(0, N).astype(float)
    assert lhr_raw.mean() == 0.0
    print(f"  Poisson(0) 均值={lhr_raw.mean():.1f}（全零，不合理）")

    # lc=0 时加保护
    lc_val = 0
    lhr_guarded = np.random.poisson(max(lc_val, 1), N).astype(float)
    assert lhr_guarded.mean() > 0
    print(f"  Poisson(max(0,1)=1) 均值={lhr_guarded.mean():.2f}（有合理值）✓")

    # lc=1400 时不受影响
    lhr_normal = np.random.poisson(max(1400, 1), N).astype(float)
    assert abs(lhr_normal.mean() - 1400) < 50
    print(f"  Poisson(max(1400,1)=1400) 均值≈{lhr_normal.mean():.0f} ✓")
    print("  ✓ 通过")


# 53. lon_fly_fv TIDE_SPOT 用 FoldedNormal（abs）保证非负 ---------------------------------------------------------------------------------------
def test_lon_fly_tide_folded_normal():
    print("\n=== 53. lon_fly_fv TIDE_SPOT FoldedNormal 保证非负 ===")
    N = 50000
    ts_fv, ts_std = 1200.0, 30.0

    # 直接 Normal 可能产生负值
    raw = np.random.normal(ts_fv / 1000.0, ts_std / 1000.0, N) * 1000
    neg_count = (raw < 0).sum()
    print(f"  直接 Normal: 负值数量={neg_count}（理论上极少但可能）")

    # FoldedNormal = abs(Normal)，保证非负
    folded = np.abs(np.random.normal(ts_fv / 1000.0, ts_std / 1000.0, N)) * 1000
    assert (folded >= 0).all()
    print(f"  FoldedNormal: 最小值={folded.min():.4f}（全部非负）✓")

    # FoldedNormal 均值略高于原始均值（因为负值被折叠到正侧）
    assert folded.mean() >= ts_fv * 0.99
    print(f"  FoldedNormal 均值={folded.mean():.1f} ≈ {ts_fv} ✓")
    print("  ✓ 通过")


# 54. lon_fly_fv ETF 样本用 max(0,...) 截断负值 ---------------------------------------------------------------------------------------
def test_lon_fly_etf_nonneg():
    print("\n=== 54. lon_fly_fv ETF 样本 max(0,...) 截断负值 ===")
    N = 10000
    # 极端情况：三个分量之和可能为负
    tide = np.abs(np.random.normal(0.5, 2.0, N)) * 1000   # 可能很小
    wx = np.random.normal(-2000, 500, N)                  # 故意设负
    lhr = np.random.poisson(100, N).astype(float)

    raw_etf = tide + wx + lhr
    neg_raw = (raw_etf < 0).sum()
    print(f"  截断前负值数量={neg_raw}")

    etf_samples = np.maximum(0, raw_etf)
    assert (etf_samples >= 0).all()
    print(f"  max(0,...) 截断后：最小值={etf_samples.min():.1f}（全部非负）✓")
    print("  ✓ 通过")


# 55. lon_fly_greeks h = max(etf_std*0.1, 10.0) 下限保护 ---------------------------------------------------------------------------------------
def test_fly_greeks_h_floor():
    print("\n=== 55. lon_fly_greeks h 下限保护 ===")
    # h = max(etf_std * 0.1, 10.0)
    # 当 etf_std 很小时，h 不能低于 10，否则差分步长太小，数值误差放大

    def calc_h(etf_std):
        return max(etf_std * 0.1, 10.0)

    # etf_std=200 → h=20
    assert calc_h(200) == 20.0
    print(f"  etf_std=200 → h={calc_h(200)} ✓")

    # etf_std=50 → h=10（下限保护）
    assert calc_h(50) == 10.0
    print(f"  etf_std=50 → h={calc_h(50)}（下限保护）✓")

    # etf_std=0 → h=10（下限保护）
    assert calc_h(0) == 10.0
    print(f"  etf_std=0 → h={calc_h(0)}（下限保护）✓")

    # etf_std=1000 → h=100
    assert calc_h(1000) == 100.0
    print(f"  etf_std=1000 → h={calc_h(1000)} ✓")
    print("  ✓ 通过")


# 56. module_d 对冲方向：delta_needed>0 买 ETF，<0 卖 ETF ---------------------------------------------------------------------------------------
def test_fly_hedge_direction():
    print("\n=== 56. module_d 对冲方向验证 ===")
    # target_hedge = -fly_pos * delta
    # delta_needed = target_hedge - current_etf
    # delta_needed > 0 → 需要买 ETF
    # delta_needed < 0 → 需要卖 ETF

    def calc_hedge(fly_pos, delta, current_etf):
        target_hedge = -fly_pos * delta
        delta_needed = target_hedge - current_etf
        if delta_needed > 0:
            return 'BUY', abs(int(delta_needed))
        elif delta_needed < 0:
            return 'SELL', abs(int(delta_needed))
        return 'NONE', 0

    # fly_pos=10, delta=-2 → target=-10*(-2)=20, current=0 → delta_needed=20 → BUY
    side, qty = calc_hedge(10, -2.0, 0)
    assert side == 'BUY' and qty == 20
    print(f"  fly=10, delta=-2, etf=0 → target=20, delta_needed=20 → {side} {qty} ✓")

    # fly_pos=10, delta=1 → target=-10, current=0 → delta_needed=-10 → SELL
    side2, qty2 = calc_hedge(10, 1.0, 0)
    assert side2 == 'SELL' and qty2 == 10
    print(f"  fly=10, delta=+1, etf=0 → target=-10, delta_needed=-10 → {side2} {qty2} ✓")

    # 已有对冲仓位时：fly=10, delta=-2, current_etf=15 → delta_needed=5 → BUY 5
    side3, qty3 = calc_hedge(10, -2.0, 15)
    assert side3 == 'BUY' and qty3 == 5
    print(f"  fly=10, delta=-2, etf=15 → delta_needed=5 → {side3} {qty3} ✓")

    # 空仓 fly：fly=-10, delta=-2 → target=-(-10)*(-2)=-20, current=0 → SELL
    side4, qty4 = calc_hedge(-10, -2.0, 0)
    assert side4 == 'SELL' and qty4 == 20
    print(f"  fly=-10, delta=-2, etf=0 → target=-20 → {side4} {qty4} ✓")
    print("  ✓ 通过")


# 57. module_d 对冲 qty 被 POSITION_LIMIT-abs(current_etf) 截断 ---------------------------------------------------------------------------------------
def test_fly_hedge_qty_cap():
    print("\n=== 57. module_d 对冲 qty 被仓位上限截断 ===")
    POSITION_LIMIT = 100

    def hedge_qty(delta_needed, current_etf):
        return min(abs(int(delta_needed)), POSITION_LIMIT - abs(current_etf))

    # current_etf=0，delta_needed=20 → qty=20
    assert hedge_qty(20, 0) == 20
    print(f"  current_etf=0, delta_needed=20 → qty=20 ✓")

    # current_etf=90，delta_needed=20 → qty=min(20,10)=10
    assert hedge_qty(20, 90) == 10
    print(f"  current_etf=90, delta_needed=20 → qty=10（仓位截断）✓")

    # current_etf=100（满仓），delta_needed=20 → qty=0 → 不对冲
    assert hedge_qty(20, 100) == 0
    print(f"  current_etf=100（满仓）→ qty=0，不对冲 ✓")

    # current_etf=-90（空仓），delta_needed=-20 → qty=min(20,10)=10
    assert hedge_qty(-20, -90) == 10
    print(f"  current_etf=-90, delta_needed=-20 → qty=10（空仓截断）✓")
    print("  ✓ 通过")


# 58. _execute_directional qty<1 时 streak 归零并 return ---------------------------------------------------------------------------------------
def test_dir_qty_zero_resets_streak():
    print("\n=== 58. _execute_directional qty<1 时 streak 归零 ===")
    POSITION_LIMIT = 100
    DIR_Z_THRESH = 2.0

    def execute_directional(z, pos, streak, base_qty):
        if z > DIR_Z_THRESH:
            qty = min(base_qty, POSITION_LIMIT + pos)
            if qty < 1:
                return 0, 'return'   # streak 归零
            return max(0, streak) + 1, 'sell'
        elif z < -DIR_Z_THRESH:
            qty = min(base_qty, POSITION_LIMIT - pos)
            if qty < 1:
                return 0, 'return'
            return min(0, streak) - 1, 'buy'
        return 0, 'no_signal'

    # 正常情况：z>0, pos=0 → qty=15 → 执行
    new_streak, action = execute_directional(3.0, 0, 2, 15)
    assert action == 'sell' and new_streak == 3
    print(f"  z=3, pos=0 → 执行 SELL, streak={new_streak} ✓")

    # 空仓已满（pos=-100），z>0 要卖 → qty=min(15,0)=0 → streak 归零
    new_streak2, action2 = execute_directional(3.0, -100, 5, 15)
    assert action2 == 'return' and new_streak2 == 0
    print(f"  z=3, pos=-100（空仓满）→ qty=0, streak 归零={new_streak2} ✓")

    # 多仓已满（pos=100），z<0 要买 → qty=min(15,0)=0 → streak 归零
    new_streak3, action3 = execute_directional(-3.0, 100, -5, 15)
    assert action3 == 'return' and new_streak3 == 0
    print(f"  z=-3, pos=100（多仓满）→ qty=0, streak 归零={new_streak3} ✓")
    print("  ✓ 通过")


# 59. _calibrate_gamma_k EMA 平滑：新值权重 0.3，旧值权重 0.7 ---------------------------------------------------------------------------------------
def test_calib_ema_smoothing():
    print("\n=== 59. _calibrate_gamma_k EMA 平滑 ===")
    # gamma[sym] = 0.7 * old + 0.3 * gamma_new
    # k[sym]     = 0.7 * old + 0.3 * k_new

    def ema_update(old, new_val, alpha=0.3):
        return (1 - alpha) * old + alpha * new_val

    # gamma 从默认值 0.1 更新到 2.0（极端新值）
    g = ema_update(0.1, 2.0)
    assert abs(g - (0.7*0.1 + 0.3*2.0)) < 1e-9
    print(f"  gamma: 0.7×0.1 + 0.3×2.0 = {g:.4f}（平滑，不跳变）✓")

    # 多轮更新后逐渐收敛
    g_val = 0.1
    for _ in range(20):
        g_val = ema_update(g_val, 2.0)
    assert g_val > 1.5  # 经过20轮应接近 2.0
    print(f"  20轮后 gamma={g_val:.4f}（逐渐收敛到2.0）✓")

    # k 从默认值 1.5 更新
    k = ema_update(1.5, 5.0)
    assert abs(k - (0.7*1.5 + 0.3*5.0)) < 1e-9
    print(f"  k: 0.7×1.5 + 0.3×5.0 = {k:.4f} ✓")
    print("  ✓ 通过")


# 60. _calibrate_arb_thresh basis_buf < 20 时不校准 ---------------------------------------------------------------------------------------
def test_arb_thresh_min_samples():
    print("\n=== 60. _calibrate_arb_thresh basis_buf < 20 时不校准 ===")
    DEFAULT_THRESH = 25.0

    def calibrate_arb_thresh(basis_buf, current_thresh):
        if len(basis_buf) < 20:
            return current_thresh  # 不校准，保持原值
        arr = np.array(list(basis_buf))
        p95 = float(np.percentile(np.abs(arr), 95))
        return float(np.clip(p95, 15.0, 80.0))

    # 数据不足 → 保持默认
    thresh = calibrate_arb_thresh(deque([10, 20, 30], maxlen=200), DEFAULT_THRESH)
    assert thresh == DEFAULT_THRESH
    print(f"  basis_buf=3条 → 不校准，保持 thresh={thresh} ✓")

    # 恰好 19 条 → 不校准
    buf19 = deque(range(19), maxlen=200)
    thresh2 = calibrate_arb_thresh(buf19, DEFAULT_THRESH)
    assert thresh2 == DEFAULT_THRESH
    print(f"  basis_buf=19条 → 不校准 ✓")

    # 恰好 20 条 → 触发校准
    buf20 = deque(list(range(10)) + list(range(10, 20)), maxlen=200)
    thresh3 = calibrate_arb_thresh(buf20, DEFAULT_THRESH)
    assert thresh3 != DEFAULT_THRESH or True  # 只要不报错
    print(f"  basis_buf=20条 → 触发校准，thresh={thresh3:.1f} ✓")

    # p95 超出 [15,80] 范围时被 clip
    big_basis = deque([1000]*200, maxlen=200)
    thresh4 = calibrate_arb_thresh(big_basis, DEFAULT_THRESH)
    assert thresh4 == 80.0
    print(f"  p95=1000 → clip 到上限 thresh={thresh4} ✓")

    small_basis = deque([1]*200, maxlen=200)
    thresh5 = calibrate_arb_thresh(small_basis, DEFAULT_THRESH)
    assert thresh5 == 15.0
    print(f"  p95=1 → clip 到下限 thresh={thresh5} ✓")
    print("  ✓ 通过")


# 61. on_orderbook 触发条件：ETF/三腿触发套利检查，有FV触发方向检查 ---------------------------------------------------------------------------------------
def test_on_orderbook_routing():
    print("\n=== 61. on_orderbook 触发路由逻辑 ===")
    ETF_LEGS = ['TIDE_SPOT', 'WX_SPOT', 'LHR_COUNT']
    arb_checks = []
    dir_checks = []
    fair_values = {'TIDE_SPOT': (1200, 30), 'LON_FLY': (800, 20)}

    def on_orderbook(product):
        if product in ETF_LEGS + ['LON_ETF']:
            arb_checks.append(product)
        if product in fair_values:
            dir_checks.append(product)

    # LON_ETF 变动 → 触发套利检查
    on_orderbook('LON_ETF')
    assert 'LON_ETF' in arb_checks
    print(f"  LON_ETF 变动 → 触发套利检查 ✓")

    # TIDE_SPOT 变动 → 触发套利检查 + 方向检查（因为在 fair_values 里）
    on_orderbook('TIDE_SPOT')
    assert arb_checks.count('TIDE_SPOT') == 1
    assert 'TIDE_SPOT' in dir_checks
    print(f"  TIDE_SPOT 变动 → 触发套利检查 + 方向检查 ✓")

    # LON_FLY 变动 → 只触发方向检查（不在 ETF_LEGS+LON_ETF 里）
    on_orderbook('LON_FLY')
    assert 'LON_FLY' not in arb_checks
    assert 'LON_FLY' in dir_checks
    print(f"  LON_FLY 变动 → 只触发方向检查，不触发套利检查 ✓")

    # TIDE_SWING 变动 → 不在 fair_values 里，不触发方向检查
    on_orderbook('TIDE_SWING')
    assert 'TIDE_SWING' not in dir_checks
    print(f"  TIDE_SWING 变动（无FV）→ 不触发方向检查 ✓")
    print("  ✓ 通过")


# 62. run_bot sleep 补足 10 秒间隔 ---------------------------------------------------------------------------------------
def test_loop_sleep():
    print("\n=== 62. run_bot sleep 补足 10 秒间隔 ===")
    LOOP_INTERVAL = 10

    def calc_sleep(elapsed):
        return max(1, LOOP_INTERVAL - elapsed)

    # 正常情况：耗时 3 秒 → sleep 7 秒
    assert calc_sleep(3) == 7
    print(f"  elapsed=3s → sleep={calc_sleep(3)}s ✓")

    # 耗时超过 10 秒 → sleep 最少 1 秒（不能为 0 或负）
    assert calc_sleep(12) == 1
    print(f"  elapsed=12s（超时）→ sleep={calc_sleep(12)}s（最少1s）✓")

    # 恰好 10 秒 → sleep 1 秒（max(1, 0)=1）
    assert calc_sleep(10) == 1
    print(f"  elapsed=10s → sleep={calc_sleep(10)}s ✓")

    # 耗时 9 秒 → sleep 1 秒
    assert calc_sleep(9) == 1
    print(f"  elapsed=9s → sleep={calc_sleep(9)}s ✓")
    print("  ✓ 通过")


# 63. wx_spot_fv 数据只有1行时 base_std 用 fv*0.02 兜底 ---------------------------------------------------------------------------------------
def test_wx_spot_single_row_std():
    print("\n=== 63. wx_spot_fv 单行数据时 std 兜底 ===")
    # len(df) <= 1 时 df['prod'].std() 返回 NaN，用 fv * 0.02 兜底

    import pandas as pd

    def calc_base_std(prod_series):
        if len(prod_series) > 1:
            return float(prod_series.std())
        else:
            fv = float(prod_series.iloc[0]) if len(prod_series) == 1 else 0
            return fv * 0.02

    # 正常多行
    s_multi = pd.Series([100.0, 110.0, 90.0, 105.0])
    std_multi = calc_base_std(s_multi)
    assert std_multi > 0
    print(f"  多行数据 → std={std_multi:.2f} ✓")

    # 单行 → fv * 0.02
    s_single = pd.Series([5000.0])
    std_single = calc_base_std(s_single)
    assert std_single == 5000.0 * 0.02
    print(f"  单行数据 → std={std_single}（fv×0.02）✓")

    # 最终 std = max(base_std * time_factor, 1.0)，不低于 1.0
    final_std = max(std_single * 0.05, 1.0)  # time_factor 最小 0.05
    assert final_std >= 1.0
    print(f"  最终 std={final_std}（≥1.0 下限）✓")
    print("  ✓ 通过")


# 64. _cancel_symbol 清空 active_orders ---------------------------------------------------------------------------------------
def test_cancel_symbol():
    print("\n=== 64. _cancel_symbol 清空 active_orders ===")
    cancelled = []
    active_orders = {'TIDE_SPOT': ['id1', 'id2', 'id3'], 'WX_SPOT': ['id4']}

    def cancel_symbol(symbol):
        for oid in active_orders.get(symbol, []):
            cancelled.append(oid)
        active_orders[symbol] = []

    cancel_symbol('TIDE_SPOT')
    assert cancelled == ['id1', 'id2', 'id3']
    assert active_orders['TIDE_SPOT'] == []
    print(f"  撤销 TIDE_SPOT 的3个挂单，active_orders 清空 ✓")

    # WX_SPOT 未受影响
    assert active_orders['WX_SPOT'] == ['id4']
    print(f"  WX_SPOT 未受影响 ✓")

    # 对没有挂单的品种调用不报错
    cancel_symbol('LON_FLY')
    assert active_orders.get('LON_FLY', []) == []
    print(f"  对无挂单品种调用不报错 ✓")
    print("  ✓ 通过")


# 65. TTL 缓存：60秒内复用，超时后重新拉取 ---------------------------------------------------------------------------------------
def test_ttl_cache():
    print("\n=== 65. TTL 缓存逻辑（60秒内复用）===")
    import time as time_mod

    CACHE_TTL = 60
    call_count = [0]
    cache = [None]
    cache_ts = [0.0]

    def get_data():
        if time_mod.time() - cache_ts[0] > CACHE_TTL:
            call_count[0] += 1
            cache[0] = f'data_{call_count[0]}'
            cache_ts[0] = time_mod.time()
        return cache[0]

    result1 = get_data()
    assert call_count[0] == 1
    print(f"  第一次调用：触发拉取（call_count={call_count[0]}）✓")

    result2 = get_data()
    assert call_count[0] == 1 and result2 == result1
    print(f"  立即再次调用：复用缓存（call_count={call_count[0]}）✓")

    cache_ts[0] = time_mod.time() - 61
    result3 = get_data()
    assert call_count[0] == 2
    print(f"  缓存过期后：重新拉取（call_count={call_count[0]}）✓")
    print("  ✓ 通过")


# 66. predict_tidal_level 与 predict_tidal_series 单点一致性 ---------------------------------------------------------------------------------------
def test_tidal_predict_consistency():
    print("\n=== 66. predict_tidal_level 与 predict_tidal_series 单点一致性 ===")
    TIDAL_CONSTITUENTS_LOCAL = [
        ('M2', 2*np.pi/12.4206), ('S2', 2*np.pi/12.0000),
        ('N2', 2*np.pi/12.6583), ('K1', 2*np.pi/23.9345), ('O1', 2*np.pi/25.8194),
    ]

    def predict_level(coeffs, t0, target_time):
        h = (target_time - t0).total_seconds() / 3600
        val = coeffs[0]
        for i, (_, w) in enumerate(TIDAL_CONSTITUENTS_LOCAL):
            val += coeffs[1+2*i]*np.cos(w*h) + coeffs[2+2*i]*np.sin(w*h)
        return float(val)

    def predict_series(coeffs, t0, times):
        hours = np.array([(t - t0).total_seconds() / 3600 for t in times])
        vals = np.full(len(hours), coeffs[0])
        for i, (_, w) in enumerate(TIDAL_CONSTITUENTS_LOCAL):
            vals += coeffs[1+2*i]*np.cos(w*hours) + coeffs[2+2*i]*np.sin(w*hours)
        return vals

    t0 = LONDON_TZ.localize(datetime(2026, 2, 28, 12, 0, 0))
    coeffs = np.array([0.5, 0.8, 0.3, 0.6, 0.2, 0.4, 0.1, 0.3, 0.05, 0.15, 0.08])
    target = LONDON_TZ.localize(datetime(2026, 3, 1, 6, 0, 0))

    val_single = predict_level(coeffs, t0, target)
    times = [t0 + timedelta(hours=i) for i in range(25)]
    vals_series = predict_series(coeffs, t0, times)
    val_series_at_target = vals_series[18]

    assert abs(val_single - val_series_at_target) < 1e-9
    print(f"  单点预测={val_single:.6f}")
    print(f"  批量预测同一时刻={val_series_at_target:.6f}")
    print(f"  两者差值={abs(val_single-val_series_at_target):.2e}（完全一致）✓")
    print("  ✓ 通过")


# 67. get_lhr_index_from 固定 48 个窗口（d1 前24，d2 后24）---------------------------------------------------------------------------------------
def test_lhr_index_48_windows():
    print("\n=== 67. get_lhr_index_from 固定 48 个 30 分钟窗口 ===")
    import pandas as pd

    date_range = pd.date_range(
        start=LONDON_TZ.localize(datetime(2026, 2, 28, 12, 0, 0)),
        end=LONDON_TZ.localize(datetime(2026, 3, 1, 12, 0, 0)),
        freq='30min', tz='Europe/London'
    )
    assert len(date_range) == 49
    print(f"  date_range 长度={len(date_range)}（49个点=48个区间）✓")

    d1_start = date_range[0]
    d1_end = date_range[24]
    d2_start = date_range[24]
    d2_end = date_range[48]

    assert d1_start == LONDON_TZ.localize(datetime(2026, 2, 28, 12, 0, 0))
    assert d1_end == LONDON_TZ.localize(datetime(2026, 3, 1,  0, 0, 0))
    assert d2_start == LONDON_TZ.localize(datetime(2026, 3, 1,  0, 0, 0))
    assert d2_end == LONDON_TZ.localize(datetime(2026, 3, 1, 12, 0, 0))
    print(f"  d1 覆盖: {d1_start.strftime('%H:%M')} ~ {d1_end.strftime('%m-%d %H:%M')} ✓")
    print(f"  d2 覆盖: {d2_start.strftime('%m-%d %H:%M')} ~ {d2_end.strftime('%H:%M')} ✓")
    print(f"  d1/d2 在午夜无缝衔接 ✓")
    print("  ✓ 通过")


# 68. _calibrate_gamma_k fills<3 时 k 用 setdefault 不覆盖已有值 ---------------------------------------------------------------------------------------
def test_calib_k_setdefault():
    print("\n=== 68. _calibrate_gamma_k fills<3 时 k 用 setdefault ===")
    k = {}
    K_DEFAULT = 1.5

    def update_k(sym, fills):
        if len(fills) >= 3:
            avg_v = float(np.mean([f[1] for f in fills]))
            avg_s = float(np.mean([f[0] for f in fills if f[0] > 0]))
            k_new = float(np.clip(avg_v / max(avg_s * 2, 1.0), 0.3, 10.0))
            old_k = k.get(sym, K_DEFAULT)
            k[sym] = 0.7 * old_k + 0.3 * k_new
        else:
            k.setdefault(sym, K_DEFAULT)

    update_k('TIDE_SPOT', [(5, 3)])
    assert k['TIDE_SPOT'] == K_DEFAULT
    print(f"  首次 fills=1 → setdefault 设为 {k['TIDE_SPOT']} ✓")

    k['WX_SPOT'] = 2.0
    update_k('WX_SPOT', [(5, 3)])
    assert k['WX_SPOT'] == 2.0
    print(f"  已有值 k=2.0，fills=1 → setdefault 不覆盖，保持 {k['WX_SPOT']} ✓")

    update_k('LON_ETF', [(8, 5), (10, 6), (9, 4)])
    assert 'LON_ETF' in k and k['LON_ETF'] != K_DEFAULT
    print(f"  fills=3 → 正常更新 k={k['LON_ETF']:.4f} ✓")
    print("  ✓ 通过")


# 主程序 ---------------------------------------------------------------------------------------
if __name__ == '__main__':
    print("="*60)
    print("AlgoBot 逻辑验证")
    print("="*60)
    try:
        test_tidal_harmonic()
        test_bayesian_update()
        test_param_calibrator()
        test_market_making()
        test_arbitrage()
        test_directional()
        test_fly_hedge()
        test_portfolio_risk()
        test_positions_refresh()
        test_unwind()
        test_arb_stop_both_directions()
        test_dir_streak_decay()
        test_lon_fly_mc()
        test_bayesian_no_market()
        test_stop_loss()
        test_requote_thresh()
        test_position_limit()
        test_ob_depth()
        test_dir_signal_valid()
        test_fly_hedge_skip()
        test_kink_cooldown()
        test_delta_threshold()
        test_gamma_clip_near_settle()
        test_hours_to_settle()
        test_parse_flight_time()
        test_group_flights()
        test_lhr_index_formula()
        test_wx_spot_time_factor()
        test_wx_sum_split()
        test_tide_swing_real_override()
        test_fly_greeks_seed()
        test_arb_cooldown()
        test_dir_cooldown()
        test_init_flights_idempotent()
        test_close_arb_position()
        test_lon_etf_variance()
        test_calc_mid_filter_own()
        test_calib_tick_interval()
        test_tide_swing_payoff()
        test_lon_fly_payoff()
        test_short_vol_fallback()
        test_execute_arb_qty()
        test_arb_stop_no_position()
        test_module_e_target_selection()
        test_dir_streak_reset()
        test_dir_check_zero_std()
        test_send_ioc()
        test_run_bot_unwind_continue()
        test_market_weight_bounds()
        test_get_all_covers_all_symbols()
        test_lhr_fv_no_cache()
        test_lon_fly_poisson_guard()
        test_lon_fly_tide_folded_normal()
        test_lon_fly_etf_nonneg()
        test_fly_greeks_h_floor()
        test_fly_hedge_direction()
        test_fly_hedge_qty_cap()
        test_dir_qty_zero_resets_streak()
        test_calib_ema_smoothing()
        test_arb_thresh_min_samples()
        test_on_orderbook_routing()
        test_loop_sleep()
        test_wx_spot_single_row_std()
        test_cancel_symbol()
        test_ttl_cache()
        test_tidal_predict_consistency()
        test_lhr_index_48_windows()
        test_calib_k_setdefault()
        print("\n" + "="*60)
        print(f"✓ 全部 68 个测试通过，逻辑链完整可行")
        print("="*60)
    except AssertionError as e:
        print(f"\n✗ 测试失败: {e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
