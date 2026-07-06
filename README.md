# dxchallenge_planning_eval

DX 챌린지(motion planning) 평가 프로그램. 참가자 제출물(planner)을 rideflux
데이터셋 위에서 시뮬레이션하고 **rideflux score**를 채점한다.
pure Waymax 기반이며 V-Max에 의존하지 않는다 (필요한 metric은
`challenge_metrics/`에 V-Max에서 복사).

## 실행

```bash
uv run evaluate.py \
    --path_dataset /path/to/rideflux_validation.tfrecord@495 \
    --submission submission_example \
    --batch_size 16 \          # 기본값 16
    --max_scenarios 100        # 생략하면 전체
```

결과: stdout 요약 + `<output_dir>/evaluation_episodes.csv`(시나리오별) +
`evaluation_results.txt`(평균). `--output_dir` 기본값은 `results/<제출물 이름>`.

평가는 항상 **lockstep batch**로 돈다: actor의 `init`/`select_action`을
batch에 vmap하므로 호출 하나하나는 여전히 unbatched state를 받는다
(per-scenario 의미는 동일). 따라서 **제출물은 JAX-traceable해야 한다**
(아래 제약 섹션 참조; batch 안에서 예외가 나면 그 batch 전체가 0점 처리됨).
시나리오 순회 순서는 shard 파일순×레코드순으로 고정되어 batch 크기와
무관하게 동일하다 (batch=1 vs batch=16 결과가 200/200 시나리오 완전 일치 확인).

## 시뮬레이션 구성

- `PlanningAgentEnvironment`: ego(SDC)만 참가자 planner가 제어, 나머지 객체는
  logged trajectory 재생.
- ego dynamics: `InvertibleBicycleModel(normalize_actions=True)`
  (V-Max 베이스라인과 동일).
- episode: 9초(91 step) 중 warmup 11 step 후 80 step 시뮬레이션.
  **충돌(overlap) 또는 도로이탈(offroad_in_box) 발생 시 조기 종료.**
- sdc_paths는 만들지 않는다 (roadgraph-free metric만 사용).

## 참가자에게 주어지는 입력 (미래 정보 차단)

매 step, planner는 `waymax.datatypes.SimulatorState`를 받는다. 단:

- `log_trajectory` / `log_traffic_light`의 **현재 timestep 이후는 모두
  invalid 처리 + 값 0으로 소거**된다 (valid flag를 무시해도 미래를 읽을 수 없음).
- 유일한 예외: **ego의 마지막 logged state(goal)는 남겨진다.**
  추출 예시는 `submission_example/actor.py`의 `get_goal_xy` 참조.
- observation 정의 / feature extraction은 전적으로 참가자 코드 몫이다.

## 제출물 형식

디렉토리 하나. 필수 파일은 `actor.py`:

```python
def create_actor(submission_dir: str) -> waymax.agents.actor_core.WaymaxActorCore:
    ...
```

- `WaymaxActorCore`를 상속(또는 `actor_core_factory` 사용)한 planner를 반환.
- **`init`/`select_action`은 JAX-traceable해야 한다** (아래 제약 섹션).
- weight 파일 등은 같은 디렉토리에 두고 `submission_dir` 기준으로 로드.
- `select_action(params=None, state, actor_state, rng)`이 반환하는
  `WaymaxActorOutput.action`:
  - `data`: float32 `(2,)` = (가속도, 조향), 각각 [-1, 1]
    (내부적으로 ±6.0 m/s², ±0.3 curvature로 스케일)
  - `valid`: bool `(1,)`
- 예시: `submission_example/` (등속 주행 planner).

## 채점

시나리오(에피소드)마다 — 조기 종료 시 실행된 step까지만 집계:

| metric           | per-episode 집계          |
| ---------------- | ------------------------- |
| `progress_ratio` | 마지막 step 값 (logged 경로 대비 진행률) |
| `comfort`        | step 평균 (nuPlan 6개 임계값 만족 비율) |
| `overlap`        | max (충돌 여부 0/1)       |
| `offroad_in_box` | max (도로이탈 여부 0/1)   |

```
rideflux_score = (7·clip(progress_ratio,0,1) + 3·comfort)/10 × (1−overlap) × (1−offroad_in_box)
```

최종 점수 = 전체 시나리오의 rideflux_score 평균. 참가자 코드가 특정
시나리오에서 예외를 던지면 해당 시나리오는 0점 처리(`error` 컬럼 표시).

## 제약: JAX-traceable 코드

평가기는 actor를 `jit(vmap(...))` 안에서 호출한다. 즉 `init`/`select_action`은
JAX가 tracer(값 없이 shape/dtype만 있는 추상 입력)로 녹화할 수 있는 코드여야
한다. 안 되는 것들:

- state 값에 의존하는 python 제어 흐름: `if state.timestep > 15:` →
  `jnp.where`/`lax.cond` 사용
- 구체 값 추출: `int(x)`, `float(x)`, `.item()`, `bool(x)`
- traced 배열을 JAX 밖으로 반출: `np.asarray(...)`, torch 연산
- 값에 따라 shape가 변하는 연산: boolean-mask 인덱싱 등 (마스킹은
  `jnp.where`로 고정 shape 유지)

제출물 안에서 직접 `jax.jit`를 걸 필요는 없다 — 평가기가 `select_action`을
`jit(vmap(...))`으로 감싸므로 내부 jit은 있어도 인라인되어 의미가 없다.
traceable 여부가 궁금하면 로컬에서 `jax.jit(actor.select_action)(None, state,
None, rng)`가 돌아가는지 확인해보면 된다.

```python
class MyPlanner(actor_core.WaymaxActorCore):
    def __init__(self, weights):
        self._weights = weights          # jnp 배열로 보관 → traced 연산에 사용 가능
    def select_action(self, params, state, actor_state, rng):
        ...                              # jnp/lax 연산만 사용 (jit 불필요)
```

`--disable_future_masking`은 주최측 디버그 전용(expert replay 검증 등)이며
채점에 사용 금지.
