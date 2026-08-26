# VRAM / GPU 최적화 문제 파악 피드백

작성: 2026-08-26 · Branch: `feat-add-algorithm` · HW: RTX PRO 2000 Blackwell Laptop 8GB / Ultra 7 265H / WSL2 47GB

## 1. 관측된 증상 타임라인

| 시점 | 실행 | 결과 | VRAM/전력 관측 |
|---|---|---|---|
| Run #1 | hallway 미니 전 `compare.sh hallway --standard` 초기 smoke (500f) | PC 강제 종료 + 재부팅 (uptime 4분) | GPU 7.8GB/8GB(97%) 100% + CPU 572% 동시 |
| Run #2 | 미니 500f 재실행 (격리 전) | Open3D 세그폴트(core dump), GPU 7.8GB, CPU 재시도 중 사망 | `ExtractTriangleMeshCUDA: Unable to allocate assistance mesh ... 51493 blocks` |
| Run #3 | hallway 5625f `compare.sh hallway --standard` | 배리어에 의해 worker 2회 kill, 파이프라인은 생존(399s, ok=False) | 4788MB → kill, 7826MB → kill (barrier 3909MB) |
| Run #4 | hallway 5625f 재실행(보수계수 적용 후) | 배리어 3회 kill, 생존 종료 | coarse(82f)도 4788MB로 kill |

미니(400~500f, 짧은 버스트)는 항상 생존, hallway(5625f, 수 분 지속 풀부하)만 사망/배리어 발동.

## 2. 근본 원인 분석

### 2.1 Open3D VBG 메모리 모델 — 선할제 + MC 2배 (P0)

`src/auto_mobility/reconstruction/fusion/open3d_vbg.py:estimate_block_count`

- VBG는 생성 시 `block_count` 전체를 선할제한다. 100000 블록 × 81920B = 8.19GB.
- Marching Cubes 추출 시 동일 크기의 assistance 구조를 추가 할당 → 총 2배.
- 기존 `cap_blocks=100000`은 8GB 카드에서 OOM을 보장하는 설정이었다.

### 2.2 `_BYTES_PER_BLOCK` 16배 과소평가 버그

`src/auto_mobility/reconstruction/fusion/open3d_vbg.py:10`

```python
_BYTES_PER_BLOCK = 16 * 16 * (1+1+3) * 4  # 5120B (오류)
# 정정: 16**3 * 20 = 81920B
```

이 버그로 VRAM 캡이 16배 관대해져 실질적으로 무력화됐다. hallway coarse(82f)조차 4788MB로 배리어를 넘은 직접 원인.

### 2.3 Occupancy 계수 불일치

`src/auto_mobility/reconstruction/fusion/open3d_vbg.py:_OCCUPANCY_SAFETY_FACTOR`

- 초기 계수 `0.22*1.25=0.275`는 복도 실측(51,493 active) 대비 4.4배 과대추정 → 불필요한 voxel degrade.
- 보정 후 0.08은 미니 대비 1.28배로 양호하나, 루프 복도(hallway 64m loop)는 공간을 더 빽빽히 채워 2배 이상 차이 발생.
- bbox 대각선 기반 추정은 장면 형태(단일 복도 vs 루프)를 구분하지 못한다. → hallway에서 20mm로 degrade해도 실제 7.8GB 필요.

### 2.4 CUDA Context 오버헤드 미계상

캘리브레이션 시도에서 `o3c.cuda.synchronize_device` API 부재로 실측 실패. 그러나 관측으로 추정:
- coarse 82f + buffer 1.34GB + MC 1.34GB = 2.7GB인데 실측 4.8GB → context/임시버퍼로 ~2GB 추가.
- `MachineProfile`은 `context_overhead_mb`를 측정·차감하지 않는다. `src/auto_mobility/reconstruction/runtime/budget.py:compute_resource_budgets`가 free 전체를 예산으로 취급.

### 2.5 HashMap 성장 — 캡 무력화

Open3D Tensor HashMap은 `block_count`를 초과하면 rehash로 2배씩 성장한다. `estimate_block_count`가 반환한 초기 캡(26.5k = 2172MB)이어도, 실제 active가 50k면 최종 65k→131k로 성장하며 VRAM을 초과한다. 배리어는 성장 *후*에야 kill하므로 이미 7.8GB까지 치솟은 뒤다.

### 2.6 지속 전력(유력한 강제종료 원인)

- 발열은 46~57°C로 정상. RAM <10GB.
- 공통점은 수 분간 GPU 100% + CPU 동시 부하. 노트북의 지속 전력/PSU 트립이 가장 유력. 배터리가 아닌 AC에서도 USB-C PD 65/100W 환경이면 150W급 지속 부하는 차단된다.
- L3 duty-cycle(`frames_per_chunk`/`chunk_pause_s`) 없이 4000프레임을 끊김 없이 통합한 것이 원인.

### 2.7 CPU Fallback 폭풍

`src/auto_mobility/reconstruction/fusion/open3d_vbg.py:integrate_frames` — CUDA 실패 시 600프레임 이하만 CPU 재시도하도록 제한하기 전에는, 대량 프레임 CPU 재통합이 수 분간 CPU 100%를 추가로 유발했다.

### 2.8 부모/워커 동시 부하

`src/auto_mobility/reconstruction/pipeline/standard.py:submit_fusion` — 부모가 품질 패스/raycasting을 하는 동안 워커가 GPU를 점유하면 합산 부하가 된다. 현재는 순차이나, 품질 패스는 5625장 decode로 77s간 CPU를 점유한다.

## 3. 적용된 수정 (코드 레벨)

| 파일 | 수정 |
|---|---|
| `fusion/open3d_vbg.py:10` | `_BYTES_PER_BLOCK` 81920B 정정 |
| `fusion/open3d_vbg.py:required_vram_mb` | `_OCCUPANCY_FACTOR=0.08`, `_OCCUPANCY_SAFETY_FACTOR=2.0` 도입, `max_fitting_voxel_mm` degrade 루프 |
| `fusion/open3d_vbg.py:_run` | 청크드 퓨전(`CHUNK=800`, chunk-local bbox) — 씬 크기와 무관하게 VRAM 상한 유지 |
| `fusion/open3d_vbg.py:integrate_frames` | duty-cycle pause, 대량 프레임 CPU 재시도 금지(>600f) |
| `fusion/isolated.py` | 경로 절대화, `gpu_limits`/`frames_per_chunk`/`chunk_pause_s` 전달 |
| `runtime/process.py:_gpu_sample` | nvidia-smi 폴링(2s 주기), `gpu_limits` breach 시 세션 kill |
| `runtime/thermal.py` | `power_source()` L0 배리어, `MAX 82°C/RESUME 74°C`, `wait_for_thermal_headroom()` |
| `config.py:ResourcePolicyConfig` | `vram_free_fraction 0.65→0.55`, `vram_reserve 1.25→1.5GB` → budget 4344MB |
| `pipeline/standard.py` | `fit_voxel_to_vram` DEGRADE 결정, `submit_fusion`에 L0/L2/L4 배리어 연결 |

## 4. 배리어 5계층 설계 (현재 상태)

```
L0 전원 프리플라이트: power_source() == battery → GPU 거부
L1 스케줄러 토큰: gpu_slots=1, vram/ram admission (CapacityError)
L2 워치독: run_monitored_process 폴링 중 vram>예산90%/87°C 지속 시 kill
L3 듀티사이클: 400프레임마다 8s 휴식 (워커 내부)
L4 CPU 폭풍 차단: 대량 프레임 CUDA 실패 시 CPU 재시도 금지
+ 청크드 퓨전: 900프레임 초과 시 800프레임 단위 VBG 재생성·병합
```

Run #3/#4에서 L2가 7.8GB 도달 전에 kill하여 **호스트 강제종료를 방지하고 파이프라인을 생존**시켰다(399s 우아한 실패). 미니(400f)는 배리어 미발동으로 정상 완료(146.5s, ok=True).

## 5. Sweet Spot 재검토 — 보수적이지 않은가?

- 현재 budget 4344MB(55%−1.5GB). Run #4에서 coarse(82f)도 4788MB로 kill된 것은 context 오버헤드 미계상이 원인. 예산 자체보다 **초기 캡 계산의 context 누락 + HashMap 성장**이 문제.
- 미니 15m diag에서 0.08×2배 안전계수는 19.5mm로 degrade → 실측 15.6mm로도 충분했던 점을 감안하면 **한 단계 보수적**. 루프 복도에서는 오히려 부족.
- GPU 연산은 compute-bound이므로 버퍼를 줄여도 util은 유지된다. Sweet spot은 “GPU가 메인 경로를 유지하되, 예산을 초과하면 voxel을 한 단계만 올린다”가 맞으며, 현재 0.55/1.5GB는 합리적이다. 단 **context 실측 반영 + 청크드 퓨전**이 병행돼야 과보호가 아니다.

## 6. 잔존 리스크

1. HashMap 성장 예측 불가 — 초기 캡만으로 VRAM을 보장할 수 없다. 청크드 퓨전으로 완화했으나, 단일 청크(800f) 내에서도 성장으로 배리어에 걸릴 수 있다.
2. Context 오버헤드 미계상 — budget 산출에 반영 필요.
3. 텍스처 베이커 S/C 행렬(T×V) — hallway 최종 메시 1M 삼각형 × 80뷰 = float32로도 0.4+1.2GB. 청크드 메시 병합 후 T가 커지면 재발 가능. 현재 MAX_VIEWS 80으로 완화.
4. 전체 데이터셋 hallway는 아직 `ok=True`를 달성하지 못함(배리어 kill로 인한 실패). 청크드 퓨전으로 재도전 필요.

## 7. 권고

1. `MachineProfile`에 CUDA context 실측(`memory.used` delta) 추가로 `gpu.context_overhead_mb`를 예산에서 차감.
2. 청크드 퓨전 기본 활성화 유지, `CHUNK=800`은 hallway 5625f 기준 7청크 → 각 청크 <8GB로 충분히 안전.
3. 전원 프리플라이트는 WSL에서 `/sys/class/power_supply`가 없을 수 있으므로 `unknown`은 경고만 하고 진행(현재 구현).
4. 배리어 한도는 예산 대비 90%로 유지, 로그에 `decisions`로 기록하여 재현성 확보.
5. 다음 실행은 `tmp_smoke/`가 아닌 `output/`으로 수행하되, 프레임 수는 전체 5625 그대로 진행하되 청크드+듀티사이클로 전력 평균을 낮춘다.
