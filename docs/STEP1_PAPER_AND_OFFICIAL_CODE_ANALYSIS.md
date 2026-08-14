# Step 1 — Original MSQuDDPM 논문 및 공식 구현 분석

> 상태: **완료**
> 범위: Original MSQuDDPM reproduction 사전 조사만 수행. Few-Step student/distillation은 포함하지 않는다.

## 1. 기준 자료와 provenance

| 자료 | 위치 | 판정 |
|---|---|---|
| 논문 | [arXiv:2411.17608v2](https://arxiv.org/abs/2411.17608v2) ([PDF](https://arxiv.org/pdf/2411.17608v2)) | 주 기준, 2025-03-05 PDF |
| 출판본 | [Phys. Rev. A 111, 032610](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.111.032610) | 논문 identity 확인 |
| 공식 구현 | [gkwun/msquddpm](https://github.com/gkwun/msquddpm) | README가 동일 논문의 official Python implementation이라고 명시 |
| 분석 기준 commit | [`158df6e9474aca6a9ab00d01b60fe4d65cc093ba`](https://github.com/gkwun/msquddpm/commit/158df6e9474aca6a9ab00d01b60fe4d65cc093ba) | 코드 비교 기준 |

공식 저장소는 저자 소유 및 README를 통해 공식성이 확인된다. 단, 논문 본문에서 저장소로 연결되는 reciprocal link는 발견되지 않았다.

## 2. Paper architecture

MSQuDDPM은 forward depolarization과 step별 measurement-assisted PQC로 구성된다.

\[
\{\rho_0\}\xrightarrow{\Phi_1,\ldots,\Phi_T}\{\rho_T\},
\qquad
\{\tilde\rho_T\}\xrightarrow{\tilde U_T}\cdots\xrightarrow{\tilde U_1}\{\tilde\rho_0\}.
\]

Reverse 학습은 end-to-end가 아니라 greedy/sequential 방식이다. Stage \(t=m+1\)에서 forward target \(\{\rho_m\}\)과, maximally mixed input을 이미 학습된 \(\tilde U_T,\ldots,\tilde U_{m+2}\) 및 현재 block \(\tilde U_{m+1}\)에 통과시킨 ensemble \(\{\tilde\rho_m\}\)을 비교한다. 현재 \(\theta_{m+1}\)만 학습한 뒤 고정한다.

근거: paper Sec. II.B, Fig. 1.

## 3. Forward process

논문 Eq. (1):

\[
\rho_{t+1}^{(i)}=(1-q_{t+1}^{(i)})\rho_t^{(i)}+q_{t+1}^{(i)}\frac{I}{d},
\qquad d=2^n.
\]

Cosine-exponent schedule, Eqs. (2)–(3):

\[
q_t=\left(1-\frac{\bar\alpha_t}{\bar\alpha_{t-1}}\right)^k,
\]

\[
\bar\alpha_t=\frac{f(t)}{f(0)},\qquad
f(t)=\cos^2\left(\frac{t/T+\epsilon}{1+\epsilon}\frac{\pi}{2}\right).
\]

- \(k=1\): cosine
- \(k=2\): cosine square
- 논문은 \(\epsilon\)을 small offset이라고만 하고 수치를 제시하지 않는다.
- 공식 코드는 `s=0.001`, beta clipping `[1e-4, 1]`을 사용한다.
- Appendix A의 p-SWAP은 물리적 실현 예이며, 공식 numerical implementation은 density matrix affine update를 직접 수행한다.

## 4. Reverse process

각 block은 system state와 ancilla pure state에 작용한다.

\[
\tilde\rho_t\otimes|\tilde\phi_t\rangle\langle\tilde\phi_t|.
\]

### 4.1 Circuit

각 \(L\) layer:

1. 모든 \(n+n_a\) qubit에 trainable \(R_X\)
2. 모든 qubit에 trainable \(R_Y\)
3. neighboring qubit 사이의 \(CZ\)

Step당 parameter 수:

\[
2(n+n_a)L.
\]

공식 코드는 각 layer에서 even pair loop와 odd pair loop를 모두 실행하여 open chain의 모든 neighboring CZ를 적용한다. 이를 layer별 alternating으로 해석하면 일부 edge가 누락된다.

### 4.2 Ancilla

\[
|0\rangle^{\otimes n_a}
\quad\text{또는}\quad
|\phi_{\mathrm{Haar}}\rangle\otimes|0\rangle^{\otimes(n_a-1)}.
\]

Haar mode는 step별로 \(n_{\rm train}\)개의 Haar state를 생성하여 해당 step 학습 중 고정하고, sampling 시 새로 생성한다.

### 4.3 Measurement

모든 ancilla를 Z/computational basis에서 측정한다. 결과 label은 버리지만 nonselective partial trace를 수행하는 것은 아니다. 공식 코드는 outcome을 확률적으로 sampling하고 해당 conditional system block을 정규화한다.

논문 회로와 공식 코드가 암시하는 instrument는 다음과 같다. 이 식 자체는 논문에 직접 제시되지 않았으므로 **DERIVED**이다.

\[
K_{t,z}=\langle z|_A\tilde U_t(\theta_t)|\tilde\phi_t\rangle_A,
\]

\[
p_z=\operatorname{Tr}(K_{t,z}\rho K_{t,z}^\dagger),\qquad
\rho'_{t-1,z}=\frac{K_{t,z}\rho K_{t,z}^\dagger}{p_z}.
\]

## 5. Loss

Superfidelity, Eq. (4):

\[
G(\rho,\sigma)=\operatorname{Tr}(\rho\sigma)+
\sqrt{[1-\operatorname{Tr}(\rho^2)][1-\operatorname{Tr}(\sigma^2)]}.
\]

Ensemble mean, Eq. (5):

\[
\bar G(A,B)=\mathbb E_{\rho\sim A,\sigma\sim B}[G(\rho,\sigma)].
\]

Squared MMD, Eq. (6):

\[
D_{\rm MMD}(A,B)=\bar G(A,A)+\bar G(B,B)-2\bar G(A,B).
\]

Wasserstein cost, Eq. (7):

\[
C_{ij}=1-G(\rho_i,\sigma_j),\qquad
\operatorname{Wass}(A,B)=\min_P\langle P,C\rangle
\]

with uniform ensemble marginals and \(P\ge0\). 공식 코드는 POT `ot.emd2`를 사용한다.

## 6. 1-qubit datasets

### 6.1 Clustered mixed states

\[
\rho_0=(1-q_0)|\psi_0\rangle\langle\psi_0|+q_0I/2,
\qquad q_0\in[0,0.01),
\]

\[
|\psi_0\rangle\propto|0\rangle+0.08c_0|1\rangle,
\qquad \operatorname{Re}c_0,\operatorname{Im}c_0\sim\mathcal N(0,1).
\]

논문은 \(q_0\)의 interval은 제시하지만 distribution notation으로 uniform을 명시하지 않는다. 공식 코드는 uniform sampling을 사용한다.

### 6.2 Circular mixed states

\[
\rho_0=(1-q_0)|\psi_0\rangle\langle\psi_0|+q_0I/2,
\qquad q_0\in[0,0.04),
\]

\[
|\psi_0\rangle=R_Y(\theta_0)|0\rangle,
\qquad \theta_0\sim U[0,2\pi).
\]

분포는 Bloch sphere X–Z plane의 depolarized circle이다. 공식 코드는 `[cos(phi), sin(phi)]`, `phi ~ U(0,2π)`를 사용한다. 이는 sample parameterization은 다르지만 uniform circle ensemble은 동일하다.

## 7. Paper 1-qubit configurations and results

Table I의 값이다.

| Task | \(n\) | \(n_a\) | \(n_{train}\) | \(T\) | \(L\) | Loss | Schedule | Ancilla | Reported result |
|---|---:|---:|---:|---:|---:|---|---|---|---|
| Cluster Fig. 1/6(a2) | 1 | 2 | 100 | 6 | 4 | Wasserstein | cosine | Haar + `|0>` | \(F_{data,0}=0.9853\pm0.0001\), \(F_{gen,0}=0.9873\pm10^{-5}\) |
| Cluster Fig. 6(a3) | 1 | 2 | 100 | 6 | 4 | Wasserstein | cosine | `|00>` | \(F_{data,0}=0.9852\pm0.0001\), \(F_{gen,0}=0.9888\pm10^{-5}\) |
| Circle Fig. 2/6(b3) | 1 | 2 | 200 | 6 | 8 | Wasserstein | cosine square | `|00>` | \(Wass_{data}=0.0063\), \(Wass_{gen}=0.0151\) |
| Circle Fig. 6(b2) | 1 | 2 | 200 | 4 | 8 | Wasserstein | cosine square | Haar + `|0>` | \(Wass_{data}=0.0068\), \(Wass_{gen}=0.0163\) |

Cluster metric은 generic state-to-state fidelity가 아니라:

\[
\bar F_0(E)=\frac{1}{|E|}\sum_{\rho\in E}\langle0|\rho|0\rangle.
\]

## 8. Official repository structure

```text
gkwun/msquddpm/
├── README.md
├── requirements.txt
├── MSQuDDPM_Visualizations.ipynb
├── src/
│   ├── arguments.py
│   ├── train.py
│   ├── generation.py
│   ├── loss_functions.py
│   ├── utils.py
│   └── models/
│       ├── forward_diffusion_model.py
│       └── backward_denoising_model.py
└── data/
    ├── cluster/
    ├── circular/
    └── many_body_phase/
```

Stack: Python 3.11.7, PyTorch 2.2.0, TensorCircuit 0.11.0, NumPy 1.26.4, SciPy 1.11.1, POT 0.9.3, QuTiP 4.7.5. `data/`는 raw dataset이 아니라 trained parameter/loss `.npy` artifact이며 dataset은 runtime에 합성된다.

## 9. Paper–code discrepancy register

| Severity | Item | Paper | Official code | Reproduction decision |
|---|---|---|---|---|
| High | Multi-qubit forward | Global depolarizing map 1회/step | 같은 global map을 qubit 수만큼 반복 | 1-qubit에는 영향 없음. 이후 paper-correct 구현 우선 |
| High | Default train return | 해당 없음 | `global_loss=false`이면 empty `torch.stack` crash | 수정 필요 |
| High | Generation | trained blocks를 순차 실행 | `opt_params=[]`, 수동 편집 없이는 실패 | checkpoint CLI 필요 |
| Medium | Cluster width | `epsilon_0=0.08` | `0.04` | paper-v2 baseline은 0.08 |
| Medium | README CLI | — | 존재하지 않는 `--p_limit` 사용 | 사용하지 않음 |
| Medium | Cluster example | \(n_a=2\), Wasserstein | README는 \(n_a=1\), MMD | reproduction config로 간주하지 않음 |
| Medium | Seeds | 미기재 | 고정되지 않음 | 명시적 seed 추가 |
| Low | Forward p-SWAP | feasible hardware circuit | direct matrix update | numerical reproduction에는 direct update 사용 |

## 10. Confirmed vs assumptions

### Confirmed

- Forward Eq. (1), schedules Eqs. (2)–(3)
- RX/RY + neighboring CZ, \(L\) layers
- Z-basis stochastic ancilla measurement
- Zero/Haar ancilla options
- Greedy reverse-step training
- Superfidelity MMD/Wasserstein
- Table I의 1-qubit \(n,n_a,n_{train},T,L\), loss, schedule, result

### ASSUMPTION registry for implementation

| ID | Assumption | 근거 |
|---|---|---|
| A-001 | cosine offset `epsilon=0.001` | 논문 미기재, official code 값 |
| A-002 | beta clipping `[1e-4,1]` | official code |
| A-003 | dataset \(q_0\)는 uniform sampling | official code 및 interval 문맥 |
| A-004 | general entanglement는 매 layer마다 official open-chain의 모든 neighboring CZ 적용 | 논문 general topology 불완전; official code의 even/odd loop 확인 |
| A-005 | reverse는 official code와 같은 sampled conditional branch | paper branching 설명 + official code |
| A-006 | paper에 없는 LR/epochs/decay는 config에 assumption으로 기록; `gamma=1`은 no effective decay | 임의의 paper fact 주장 방지 |
| A-007 | `n_test` 및 error-bar 반복 수는 reproduction config가 명시 | paper 미기재 |

## 11. Reproduction blockers and risks

논문만으로 exact numerical reproduction이 불가능한 항목:

- schedule offset 수치
- per-task learning rate 및 decay cadence/factor
- epoch/iteration 수와 stopping criterion
- \(n_{test}\)
- random seed 및 error-bar protocol
- 일반 register에서의 정확한 CZ topology
- reverse instrument의 명시적 수식

공식 코드가 일부 빈칸을 채우지만 deterministic 실행, checkpoint workflow, tests, artifact provenance가 부족하다. 따라서 reproduction에서는 확인된 paper 설정과 official-code assumption을 분리 기록해야 한다.

## 12. Step 1 completion criteria

- [x] 논문 architecture 분석
- [x] forward 수식 및 schedule 분석
- [x] reverse circuit/measurement/training 분석
- [x] loss 분석
- [x] clustered/circular dataset 분석
- [x] 1-qubit hyperparameter 및 reported result 추출
- [x] 공식 repository 확인 및 구조 분석
- [x] paper–code 차이 기록
- [x] 불명확한 항목과 ASSUMPTION registry 기록
- [x] 이후 구현 기준 결정

**Step 1은 완료되었다. 다음 작업은 Step 2 — 1-qubit dataset 및 forward trajectory 구현·실행 검증이다.**
