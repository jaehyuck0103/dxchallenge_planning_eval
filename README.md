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
    --max_scenarios 100        # 생략하면 전체
```

결과: stdout 요약 + `<output_dir>/evaluation_episodes.csv`(시나리오별) +
`evaluation_results.txt`(평균). `--output_dir` 기본값은 `results/<제출물 이름>`.

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

## 성능 팁: jit은 제출물 안에서

평가기는 참가자 코드를 jit하지 않는다 (JAX가 아닐 수도 있으므로). JAX로 짠
planner라면 제출물 안에서 직접 `jax.jit`를 적용하면 된다 — 입력 shape가
시나리오 간 고정이라 컴파일은 최초 1회뿐이다 (expert 기준 시나리오당 약 6배 단축).

```python
# 순수 함수형이면 그대로 감싸기 (submission_expert/actor.py 참조):
return actor_core.actor_core_factory(
    init=base.init, select_action=jax.jit(base.select_action), name=...)

# 클래스 기반이면 __init__에서 bound method를 감싸기 (weights는 상수로 컴파일됨):
class MyPlanner(actor_core.WaymaxActorCore):
    def __init__(self, weights):
        self._weights = weights
        self._jit_select = jax.jit(self._select_impl)
    def select_action(self, params, state, actor_state, rng):
        return self._jit_select(state, actor_state, rng)
```

주의: jit 내부에서는 traced 값(`state.timestep` 등)에 대한 python `if`/`int()`가
불가하다 (`jnp.where`/`lax.cond` 사용).

`--disable_future_masking`은 주최측 디버그 전용(expert replay 검증 등)이며
채점에 사용 금지.
