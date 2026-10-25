# Exact upstream RLSD method retained here

Source: public ZJU-REAL/SDAR, pinned in configs/upstream.json at
29dcb12a55d8fc17ba6ae03fd0a69048f43051f3.
Launcher: examples/rlsd_trainer/run_webshop_3b.sh.
This release reproduces that public RLSD implementation; it does not substitute
SDAR's gated auxiliary loss or reinterpret a different paper's algorithm.

## Observation -> rollout -> Teacher scoring -> update

1. The Student Actor sees only native WebShop instructions and recent
   observation/action history. Its original freely generated think/action
   responses are executed by the unchanged parser. No skill, local hint,
   Analyzer, RETHINK, forced prefix or replacement action enters rollout.
2. Native outcome reward and group-normalized GRPO sequence advantages are
   computed from the collected episodes. Invalid-action penalty0.1 is retained.
3. The unchanged SkillProvider loads general_skills.md and the mapped static
   WebShop markdown. Native text prompt keyword matching uses ANY keyword;
   multiple matching categories concatenate their skills, general text once.
   Native gamefile/data_source fallbacks are retained. Eight skill files and
   their mapping are individually hash-locked; no skill-generation API is needed.
4. The original build_teacher_batch in rlsd_ray_trainer.py decodes each Student
   prompt, prepends literal text "[Privileged Skill Information]", selected
   skill and blank separators, then tokenizes the augmented Teacher-only input.
   Identical original response token IDs, EOS and response padding mask are
   appended. This is a text prefix, not a replacement chat-message schema.
5. The same actor_rollout_wg scores both Student and Teacher inputs with the
   current Actor before each update. The Teacher follows subsequent Actor
   updates; it is not a frozen base, EMA or independently generated correction.
   The fixed RefPolicy is a different object, used only for reference KL.
6. The original RLSD loop replaces batch advantages with detached token
   advantages. The unchanged Actor then applies ordinary clipped policy loss,
   original response mask, token-mean reduction, entropy, reference KL,
   optimizer, scheduler, global minibatching and gradient accumulation.

## Exact native token advantages

For sequence advantage A_i and already sampled response token y_it, define

\[
\Delta_{it} =
\log\pi_\theta(y_{it}\mid r,s_{it})
-\log\pi_{\mathrm{old}}(y_{it}\mid s_{it}),
\]

\[
w_{it}=\operatorname{clip}\!\left(
\exp[\operatorname{sign}(A_i)\Delta_{it}],\,0.8,\,1.2\right),
\]

\[
\widehat A_{it}=
\operatorname{sg}\!\left[
A_i\{(1-\lambda_t)+\lambda_t w_{it}\}\,m_{it}
\right],
\qquad
\lambda_t=0.5\max(0,1-t/100).
\]

At the scoring point the current Actor and rollout-old policy have the same
pre-update weights. The code extracts A_i at the first valid response token;
padded tokens are zeroed. Positive A increases weight for Teacher-preferred
tokens; negative A uses the opposite signed ratio. A=0 remains zero, so RLSD
does not supply a new independent policy gradient for zero-advantage groups.
Weights and advantages are computed under torch.no_grad: neither Teacher nor
the old Student scoring pass receives an optimization gradient.

For current policy ratio rho_it relative to saved old Student probabilities,

\[
\mathcal L_{\mathrm{PG}}=
-\frac{\sum_{it}m_{it}\min\!\left(
\rho_{it}\widehat A_{it},
\operatorname{clip}(\rho_{it},1-\epsilon,1+\epsilon)\widehat A_{it}
\right)}{\sum_{it}m_{it}}.
\]

The total native objective keeps reference KL and entropy:

\[
\mathcal L=
\mathcal L_{\mathrm{PG}}+
0.01\mathcal L_{\mathrm{reference\text{-}KL}}
-0.001\mathcal H.
\]

No extra SDAR sigmoid gate, SDAR coefficient or direct Teacher CE/KL term is
added. At update1 lambda=.495, update50=.25, update100 onward=0; the last50
updates are ordinary GRPO with native regularizers. Original fit still computes
Teacher forward passes after warmdown; this release does not optimize them away.

## Preserve the original Teacher truncation

Student processing retains truncation=error and prompt4096/response512.
The original Teacher builder always left-truncates its augmented prompt to4096,
even when passed truncation=error. It can discard skill or older context.
This upstream behavior is documented, not silently repaired. CPU tests check
overflow, exact response/EOS/padding alignment and non-mutation of Student input.

## Fixed140 evaluation and minimal portability changes

The1.5B/3B RLSD, GRPO and SDAR releases share byte-identical task_split.json and
fixed_validation140.json. Five native categories each contribute28 tasks,
with140 distinct target ASINs. Before Actor interaction, each worker excludes
held-out content IDs and equivalent random-price variants from its training
catalog, including automatic resets. The remaining6765 native entries keep
their original order/weights. This is a custom synthetic objective-held-out
split, not canonical human500 or unseen-product zero-shot testing.

Each validation runs all140 tasks once with only the Student, at update0,
every5 and final150; no skill is provided. Original max15 steps/history2,
prompt4096/response512 and validation temperature.4 sampling are retained.

Original RLSD requests2 GPUs; this release binds four FSDP ranks/two TP2
replicas without changing16 groups x8 episodes/update, LR1e-6, globalmini64 or
micro8/GPU. Rank-local mini becomes16 rather than32; native accumulation is2
rather than4. Save20/final150, latest two SHA-verified complete checkpoints.

Only native WebShop worker factory/index range and main_rlsd's Trainer import
are patched. The wrapper calls original fit/_validate and changes only fixed
task selection, telemetry and transactional storage. RLSD/Teacher/loss source,
skills, environment manager, prompt, parser, reward and search remain hash-locked.

## Verification boundary

CPU CI executes original token-advantage clipping/sign/mask/detach,100-update
schedule, actual native fit advantage replacement, PPO backward/optimizer step,
current-Actor Teacher scoring, static skill matching and response alignment.
It also checks fixed task exclusion/identity, safe checkpoint failures/resume,
clean public installation and real WebShop search/buy/reward. This is not proof
of CUDA fit, NCCL, vLLM/FSDP startup, hardware save/load or150-update scores.
No benchmark numbers are prefilled and setup/tests do not launch GPU training.
