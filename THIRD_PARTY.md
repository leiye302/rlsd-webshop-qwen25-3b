# Sources, licenses and modifications

- **SDAR / veRL / verl-agent**: https://github.com/ZJU-REAL/SDAR .
  Public source pinned at `29dcb12a55d8fc17ba6ae03fd0a69048f43051f3`.
  Use official RLSD `examples/rlsd_trainer/run_webshop_3b.sh` and
  `verl.trainer.main_rlsd.run_rlsd`, not another distillation variant.
  Framework and new wrapper: Apache-2.0.
  Upstream notices remain in the downloaded checkout.
  Two native files are patched: the Trainer import alias in
  `verl/trainer/main_rlsd.py`, and the WebShop vector worker factory/index range
  in `agent_system/environments/env_package/webshop/envs.py`. The latter uses
  a subclass to exclude the fixed140 validation objectives from every training
  worker by content identity, rather than independently shuffled local indices.
  Training/validation loops, parser, prompt, search, goal generator and reward
  are unchanged. Wrapper code adds fixed140 selection, read-only telemetry,
  disjoint-pool assertions and transactional checkpoint serialization/loading.
  The same balanced140 task manifest is reused across checkpoints/model sizes.
  Original skills/webshop/skill_mapping.json and its eight skill markdown files
  are fetched from the same pinned public source and individually hash-locked.
  Current-Actor Teacher scoring, detached RLSD token advantages, lambda schedule,
  original GRPO outcome advantages and the native Actor update are unmodified.
- **WebShop**: https://github.com/princeton-nlp/WebShop .
  The embedded source retains its MIT license at
  `.deps/sdar/agent_system/environments/env_package/webshop/webshop/LICENSE.md`.
  Public product/instruction assets are downloaded, not redistributed in Git.
  Data remains subject to its original source terms; the wrapper's license
  does not relicense third-party assets.
- **Public asset mirrors**: https://huggingface.co/datasets/HongbangYuan/webshop
  and https://huggingface.co/datasets/YWZBrandon/webshop-data .
  Revisions and SHA256 are fixed in `configs/assets.json`.
  All three selected files were downloaded from both mirrors and byte-matched.
  This is a mirror-identity check, not an independent historical hash from the
  unavailable original Google Drive link.
- **spaCy / en_core_web_sm 3.7.1**:
  https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.7.1 .
  MIT; wheel URL/hash pinned. Native WebShop loads the small English model.
- **Qwen2.5-1.5B-Instruct**:
  https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct , Apache-2.0.
- **Qwen2.5-3B-Instruct**:
  https://huggingface.co/Qwen/Qwen2.5-3B-Instruct , Qwen Research License.
  Weights are downloaded from fixed public revisions; no weights ship in Git.

Please credit GRPO, RLSD, SDAR, veRL, verl-agent, WebShop and Qwen in derivative work.
The wrapper does not imply endorsement by the original authors.
