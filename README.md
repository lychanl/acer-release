# Actor-Critic with Experience Replay and autocorrelated actions
This repository contains original implementation of **Actor-Critic with
 Experience Replay and autocorrelated actions** algorithm.
 Implementation of original **Actor-Critic with Experience Replay** is also present.
## Installation

### Prerequisites
**Python3** is required.  
Note that, steps bellow won't install 
all of the OpenAI Gym environments. Visit
[OpenAI Gym repository](https://github.com/openai/gym) for more details.

### Installation steps
1. Create new virtual environment:
```shell script
python3.7 -m venv {name}
```
Note: it will create the environment folder in your current directory.

2. Activate the virtual environment (should be run from the same directory as above
or full path should be passed):
```shell script
source {name}/bin/activate 
```
3. While in the repository's root directory, update pip version:
```shell script
python -m pip install --upgrade pip
``` 
4. Install the requirements:
```shell script
pip install -r requirements.txt
``` 

5. Run the agent:
```shell script
python run.py {args...}
``` 

## Example runs

```shell script
python acer/run.py --algo acer --env_name Pendulum-v0 --gamma 0.95 \
    --lam 0.9 --b 3 --c0 0.3 --c 10 --actor_lr 0.001 --critic_lr 0.002  \
    --actor_layers 20 --critic_layers 50 --memory_size 1000000 \
    --num_parallel_envs 10  --actor_beta_penalty 0.1 --batches_per_env 10
```

```shell script
python3.7 acer/run.py --algo acerac --env_name HalfCheetahBulletEnv-v0 \
    --gamma 0.99 --lam 0.9 --b 2 --c0 0.1 --c 10 --actor_lr 0.00003 --critic_lr 0.00006 \
    --actor_layers 256 256  --critic_layers 256 256 --memory_size 1000000 \
    --num_parallel_envs 10 --actor_beta_penalty 0.001 --batches_per_env 10 \
    --num_evaluation_runs 5  --std 0.4  --max_time_steps 3000000 --tau 4 --alpha 0.5
```

## Parameters
TBA

## TensorBoard
During the training some statistics like 'loss', mean penalty value and return are being
collected and logged into TensorBoard files (*logs/* folder).  
To view the dashboard run
```shell script
tensorboard --logdir logs
```
in the repository's root directory. The dashboard will be available in the browser under
the addres http://localhost:6006/


## References
 
TBA
 
Wawrzyński, Paweł.
*Real-time reinforcement learning by sequential actor–critics
and experience replay.*
Neural Networks 22.10 (2009): 1484-1497.

Wawrzyński, Paweł, and Ajay Kumar Tanwani.
*Autonomous reinforcement learning with experience replay.*
Neural Networks 41 (2013): 156-167.



