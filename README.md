BTC 선물 자동매매 AI 시스템 코드베이스
======================================

이 리포지토리는 BTC Perpetual Futures 자동매매를 위한 오프라인 학습 → 실시간 모의매매 → 온라인 미세학습 파이프라인을 모두 포함하는 최소 실행 골격(minimal working skeleton)입니다. Lambda GPU에서의 초기 학습과 맥북 로컬 실시간 파이프라인을 모두 지원하도록 모듈을 분리했습니다.

구성 개요
---------
- `offline_training/`: 과거 데이터 로딩, 피처 생성, 지도/간단 RL 학습, 백테스트 스텁.
- `online_trading/`: 실시간 1분봉 수신, 피처 업데이트, 정책 추론, 모의주문, 리플레이 버퍼, 온라인 미세학습 루프.
- `models/`: 정책 모델 정의 및 체크포인트 로딩/저장.
- `models/checkpoint_manager.py`: 체크포인트 리스트/메타 관리.
- `envs/`: Gym 스타일 트레이딩 환경.
- `rl_training/`: PPO 등 RL 트레이너.
- `strategies/`: RL/룰 기반 공통 인터페이스(`Strategy`)와 구현.
- `online_trading/execution_engine.py`: paper/testnet/live 실행 인터페이스.
- `risk/`: `RiskManager`로 일손실/레버리지/쿨다운 제어.
- `backtesting/`: 공통 백테스터.
- `data/`: 원시 로더/피처 스토어.
- `reports/`: 로그 집계 → LLM 리포트 파이프라인.
- `data_source` 설정: exchange/symbol/timeframe/auto-download/fallback 제어.
- `llm/`: 뉴스 수집 스텁, OpenAI 요약, 리포트 생성.
- `config/config.yaml`: 경로, 하이퍼파라미터, 거래 수수료, API 키 참조.
- `requirements.txt`: 필수 파이썬 패키지 목록.

빠른 시작 (로컬 모의 환경)
-------------------------
1. 가상환경 생성 및 패키지 설치  
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. 환경 변수 설정(필요시)  
   - `BINANCE_API_KEY`, `BINANCE_API_SECRET` (실거래/테스트넷 사용 시)  
   - `OPENAI_API_KEY` (LLM 요약/리포트 사용 시)
3. 오프라인 예제 학습(랜덤/샘플 데이터)  
   ```bash
   python offline_training/train_offline.py --config config/config.yaml
   ```
4. 실시간 모의 매매 루프(웹소켓 불가 시 로컬 시뮬레이터 사용)  
   ```bash
   python online_trading/paper_trader.py --config config/config.yaml
   ```

설계 포인트
-----------
- 지연시간 500ms 목표: 피처 업데이트와 모델 추론을 분리하고, I/O(웹소켓, REST)와 연산을 비동기로 구성.
- 온라인 미세학습: 리플레이 버퍼에서 미니배치 샘플링 후 아주 작은 학습률(기본 1e-5)로 업데이트, weight anchoring을 통해 catastrophic forgetting 방지.
- 체크포인트 버전 관리: `models/checkpoints/policy_v*.pt` 자동 저장.
- 확장: 실계좌 전환 시 `online_trading/paper_trader.py`의 `PaperTrader`를 거래소 API 래퍼로 교체하면 동일한 액션 파이프라인 유지.

알려진 제한
-----------
- 실제 Binance 실시간 데이터/주문은 API 키와 네트워크가 필요하며, 제공 코드는 테스트넷/시뮬레이터 우선입니다.
- LLM 모듈은 OpenAI API 키가 필요하며, 없는 경우 graceful degrade 됩니다.

권장 워크플로우
---------------
1) Lambda 등 GPU 환경에서 오프라인 학습 후 `models/checkpoints/best_policy.pt` 확보  
2) RL 학습(선택): `python rl_training/train_ppo.py --config config/config.yaml` → `best_rl_policy.pt` 생성  
3) 맥북 로컬에서 `paper_trader.py` 실행 → 실시간/시뮬 모의매매 및 리플레이 축적 (RL 정책이 있으면 우선 사용). `strategy.type`/`execution.mode`/`risk.*`로 제어  
4) `online_train.py`로 주기적 미세학습 및 체크포인트 버전업  
5) `reports/report_pipeline.py`로 로그 집계 → LLM 리포트 생성, `logs/live/trades.jsonl` 활용  

데이터 준비
-----------
- `config/config.yaml`의 `data_source`로 거래소/심볼/기간을 정의. `auto_download_on_missing=true`이면 `paths.data`가 없을 때 ccxt로 다운로드를 시도하고, `allow_synthetic_fallback=false`이면 실데이터 없을 때 학습을 중단한다.
검색 제한(451) 등으로 실패 시 `fallback_exchanges`나 `exchange_id`를 다른 거래소로 교체 후 다시 실행.

RL 평가
-------
- 학습된 RL 정책 백테스트:  
  ```bash
  python -m rl_training.evaluate --config config/config.yaml --checkpoint models/checkpoints/best_rl_policy.pt
  ```
  결과는 로그로 출력되고 `logs/rl_eval_summary.json`에 저장된다.
- 액션 매핑은 일관되게 `0=short, 1=flat, 2=long`. 환경/평가/실거래가 동일하며, `--force_action {short,flat,long}`로 환경 보상 sanity-check가 가능하다.
- PPO 재시작 시 기존 체크포인트를 무시하고 새로 시작하려면 `python -m rl_training.train_ppo --config config/config.yaml --reset_policy`를 사용. 롤아웃 로그에 평균 엔트로피/액션 확률이 찍히며, 단일 액션으로 붕괴 시 경고를 출력한다.
- Collapse 대응 옵션: `rl.entropy_coef`, `rl.min_action_prob`(샘플링 최소 확률), `rl.flat_penalty`(flat 시 스텝 패널티)로 탐색 부족 시 튜닝 가능.
- 평가 시 argmax 대신 샘플링하려면 `--sample --temperature 1.0` 옵션 사용(액션 매핑 그대로 적용).
- `train_ppo`는 롤아웃 보상(`total_reward`)이 개선될 때마다 체크포인트를 갱신하므로, 가장 보상 좋은 정책을 자동으로 `best_rl_policy.pt`에 저장합니다.
