# CostGuard: the overnight GPU run (real-data demo)

CostGuard watches overnight training runs while the engineer sleeps. Free rules read loss and GPU stats, the AI
judges only when a rule fires, plain code (the policy) decides, and nothing restarts without the engineer's tap.

**Live demo:** https://sangyoon21.github.io/hci-agent-demos/costguard/ (one HTML file, runs offline, mock AI)

## The night in the demo (10:14 PM to 7:14 AM, three 8x A100 servers at $27.45/hr)

| Server | What really happened | What CostGuard does |
|---|---|---|
| `a100-train-2` | Peak learning rate 0.1 was too high. Val loss rose from 2.13, and from step ~29k the model collapsed to the chance level (3.31, the same as guessing letter frequencies). From step 34k on, 100% of its ReLU units are dead. GPUs stay ~92% busy the whole time. | Rule fires at 1:46 AM (smoothed loss at chance for 10 min). AI: failed. Health check finds that the 12 newest checkpoints are unhealthy and picks `step_6000`. Pauses at 1:48 AM. Morning: 1-line fix `--lr 0.1` → `--lr 3e-3 --resume ckpt/step_6000.pt`, which we actually ran (val 1.81–1.85). |
| `a100-train-1` | Healthy run. A full validation pass (217 batches) runs 12:53–1:26 AM; GPU drops to 0%. | Rule fires at 1:13 AM (no loss lines for 20 min). AI: healthy, it's an eval. Kept running. A fixed-threshold script would have killed it at 1:08 AM. |
| `a100-train-3` | Training finished at 2:03 AM; GPUs then sat at 0% until morning. | Rule fires at 2:33 AM. Final checkpoint verified. Paused. |

Without CostGuard: $741. With CostGuard: $464 ($277 avoided). Engineer controls: "do not kill" lock and sleep
budget cap per run (set before sleeping), Approve / Whitelist / Undo in the morning.

## What's real

| Piece | Status | Source |
|---|---|---|
| Training loss, val loss, grad norm, lr | **Real** | `pipeline/train_charlm.py`: char-level MLP language model (numpy, Adam, fp16-rounded matmuls) on Tiny Shakespeare |
| Checkpoints and their health (NaN, dead units, val loss) | **Real** | every checkpoint file re-loaded and re-evaluated in `pipeline/build_scenario.py` |
| The 1-line fix | **Real** | resumed run `--lr 3e-3 --resume step_6000` |
| Slow eval | **Real** | full validation pass on the real step-16000 checkpoint |
| GPU utilization | **Real** | per-minute `nvidia-smi` from 3 jobs in the Microsoft Philly trace (Jeon et al., ATC'19), via `pipeline/philly_extract.py` |
| Clock mapping | **Chosen** | the runs ran on a CPU in seconds, so their steps are spread evenly over the night (steps/min shown in the demo) |
| Price | **Public list price** | AWS p4de.24xlarge, $27.45/hr on-demand (us-east-1); check before quoting |
| AI answers | **Written in advance** | no API key in a public page; every number in them is computed from the data |
| Cloud API, phone message, IAM | **Simulated** | |

We could not make this small model produce a literal NaN (we tried SGD, Adam, fp16-rounded matmuls, ReLU and tanh,
learning rates up to 30). It fails the way small models do: the loss blows up and the network dies. NaN is the
extreme version of the same "still billing, no longer learning" failure.

## Real numbers from the Philly trace (117,325 jobs, Aug–Dec 2017)

* 33% of all GPU-hours went to jobs that ended **Failed**, 33% to jobs that were **Killed**, and 34% to jobs that passed.
* Of 140 overnight 8-GPU jobs (launched 8 PM–midnight, 5–30 h long), **73 (52%)** held all 8 GPUs under 5% busy for
  an hour or more, and **25%** of their GPU-minutes were under 5%. That excludes 14 jobs that read 0% nearly all
  night, which may be monitoring gaps. With them included: 87 of 154 jobs (56%) and 29.5%.

## Rebuild everything

```bash
# 1. raw data (outside the repo, ~1 GB)
mkdir -p ../costguard-datasets/{philly,shakespeare}
curl -L -o ../costguard-datasets/philly/trace-data.tar.gz \
  https://media.githubusercontent.com/media/msr-fiddle/philly-traces/master/trace-data.tar.gz
tar -xzf ../costguard-datasets/philly/trace-data.tar.gz -C ../costguard-datasets/philly \
  trace-data/cluster_job_log trace-data/cluster_machine_list        # gpu_util is streamed, not unpacked
curl -L -o ../costguard-datasets/shakespeare/input.txt \
  https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
pip install numpy pandas

# 2. real GPU traces
python pipeline/philly_extract.py --philly ../costguard-datasets/philly --out ../costguard-datasets/philly/extract

# 3. real training runs (each takes well under a minute on a laptop CPU)
R=../costguard-datasets/runs; C="--optim adam --precision fp16 --ckpt-every 2000 --eval-every 5000"
python pipeline/train_charlm.py $C --act relu --lr 0.1   --steps 100000 --out $R/relu01          # collapses
python pipeline/train_charlm.py $C --act relu --lr 0.003 --steps 60000 --resume $R/relu01/ckpt/step_6000.npz --out $R/relu01_fix_lr0003
python pipeline/train_charlm.py $C --act relu --lr 0.003 --steps 60000  --out $R/relu0003        # slow-eval run
python pipeline/train_charlm.py $C --act tanh --lr 0.003 --steps 60000  --out $R/tanh0003        # finishes early

# 4. scenario + page
python pipeline/build_scenario.py --datasets ../costguard-datasets --out data/scenario.json
python web/build.py         # -> web/costguard.html
```

## Files

| Path | What |
|---|---|
| `pipeline/train_charlm.py` | the real training runs |
| `pipeline/philly_extract.py` | Philly job stats + per-minute GPU util for overnight 8-GPU jobs |
| `pipeline/build_scenario.py` | maps runs and GPU traces onto the night, runs the watcher rules, checks checkpoint health |
| `data/scenario.json` | everything the page shows |
| `web/template.html` | the demo: rules → AI → checkpoint check → policy → action, morning message |
| `legacy/` | the earlier idle-machine prototype (Streamlit, live Claude option), kept as it was |
