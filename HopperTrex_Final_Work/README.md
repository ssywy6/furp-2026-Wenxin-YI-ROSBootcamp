# HopperTrex CPU Reinforcement Learning

## Final Submission

This project trains a two-wheel HopperTrex robot to climb three low steps in a MuJoCo simulation. The final policy was fine-tuned with random sudden external pushes to improve recovery behavior.

### Final model

`policy/cpu_stairs_up_2cm_push.zip`

This is a Stable-Baselines3 PPO model. Do not unzip it.

## Quick start on Windows

Install Python 3.11, open PowerShell in this folder, and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe train_curriculum.py --stage stairs --mode eval --stair-height .02 --model policy/cpu_stairs_up_2cm_push.zip
.\.venv\Scripts\python.exe train_curriculum.py --stage stairs --mode view --stair-height .02 --model policy/cpu_stairs_up_2cm_push.zip
```

The `view` command opens the MuJoCo viewer. The `eval` command runs the standard deterministic evaluation.

## External-force training

During fine-tuning, the environment applied a random horizontal force of 8–18 N for one 0.02-second control interval. The force direction was randomized. Training used 100,352 additional simulation steps, starting from a policy that already climbed 2 cm steps.

## Results

- Standard deterministic stair evaluation: 18/20 successful episodes.
- Random-push evaluation: 9/10 successful episodes.
- Full random-push results are in `reports/robust_evaluation.json`.

These are small simulation evaluations. They do not prove physical-robot performance. No manipulator, QP whole-body controller, or real-robot deployment is included in this submission.

## Repository contents

- `policy/`: the final PPO model.
- `assets/hoppertrex/`: MuJoCo robot XML and mesh files.
- `cpu_*.py`: simulation and controller source code.
- `train_curriculum.py`: training, evaluation, and viewer entry point.
- `requirements-cpu.txt`: Python dependencies.
- `reports/`: evaluation output.

