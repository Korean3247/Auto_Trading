Auto-Trading RL System (BTC Perpetuals)
=======================================

This repo is a minimal but end-to-end scaffold for BTC perpetual futures auto-trading: offline training → live/paper loop → online fine-tuning. It supports GPU training on a server (e.g., Lambda) and live/paper operation on a laptop, with modules split by function.

What’s here
-----------
- `offline_training/`: data load, feature generation, supervised/RL warm-up training, backtest stub.
- `online_trading/`: live 1m feed (or simulator), feature updates, policy inference, paper orders, replay buffer, online fine-tuning loop.
- `models/`: policy definitions and checkpoint load/save; `models/checkpoint_manager.py` for listing metadata.
- `envs/`: Gym-style trading environment with costs, excess-return rewards, and risk penalties.
- `rl_training/`: PPO trainer, evaluator (greedy or sampled), Sharpe/reward-based checkpointing.
- `strategies/`: common `Strategy` interface with RL/rule-based examples.
- `online_trading/execution_engine.py`: paper/testnet/live execution interface.
- `risk/`: `RiskManager` for daily loss/leverage/cooldown controls.
- `backtesting/`: shared backtester.
- `data/`: raw loaders (ccxt), feature store utilities.
- `reports/`: log aggregation → LLM report pipeline.
- `llm/`: news fetch stub, OpenAI summarizer, report generator.
- `config/config.yaml`: paths, hyperparams, cost model, data source, risk, action mapping.
- `requirements.txt`: dependencies.

Quick start (local / paper)
---------------------------
1. Create venv and install deps  
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Env vars (if needed)  
   - `BINANCE_API_KEY`, `BINANCE_API_SECRET` (for exchange access)  
   - `OPENAI_API_KEY` (for LLM summaries/reports)
3. Offline example training (synthetic or your parquet/CSV)  
   ```bash
   python offline_training/train_offline.py --config config/config.yaml
   ```
4. Paper trading loop (uses live feed or simulator)  
   ```bash
   python online_trading/paper_trader.py --config config/config.yaml
   ```

Design notes
------------
- Latency target: split I/O (WS/REST) and compute; keep inference/feature updates lightweight.
- Online fine-tune: sample from replay buffer with tiny LR and weight anchoring to avoid catastrophic forgetting.
- Checkpoints versioned under `models/checkpoints/`; `best_rl_policy.pt` updated when the chosen metric improves.
- Action mapping is defined in config; default is long-only scaling `[0.0, 0.5, 1.0]` with cost/risk-aware rewards.

Known limitations
-----------------
- Real exchange data/trading requires API keys and network; code defaults to simulator/testnet mode.
- LLM modules require an OpenAI API key; otherwise they no-op gracefully.

Recommended workflow
--------------------
1) Offline train on GPU, produce `models/checkpoints/best_policy.pt`.  
2) PPO (optional): `python -m rl_training.train_ppo --config config/config.yaml` → `best_rl_policy.pt` (metric-based checkpointing).  
3) Paper trader on laptop: `python online_trading/paper_trader.py --config config/config.yaml` to accumulate replay and monitor signals (strategy type / execution mode / risk via config).  
4) Online fine-tune: `python online_trading/online_train.py` to version checkpoints.  
5) Reporting: `python -m reports.report_pipeline` (uses `logs/live/trades.jsonl`) for summaries/LLM reports.

Data prep
---------
- Define exchange/symbol/timeframe in `config/config.yaml` under `data_source`. With `auto_download_on_missing=true`, ccxt will fetch to `paths.data` if missing; `allow_synthetic_fallback=false` stops training when real data is absent. Use `fallback_exchanges` or change `exchange_id` if location-blocked.

RL evaluation
-------------
- Evaluate trained policy:  
  ```bash
  python -m rl_training.evaluate --config config/config.yaml --checkpoint models/checkpoints/best_rl_policy.pt
  ```  
  Outputs to log and `logs/rl_eval_summary.json`.
- Force a constant action for sanity checks: `--force_action {short,flat,long}` (mapped via `action_mapping`).
- Sample instead of argmax: `--sample --temperature 1.0`.
- PPO restart ignoring old checkpoint: `python -m rl_training.train_ppo --config config/config.yaml --reset_policy`.
- `train_ppo` updates `best_rl_policy.pt` when the selected metric improves (`best_metric` in config: `reward` or `sharpe`).
