# Actor-Critic with Experience Replay and Pessimistic Critic
This repository contains original implementation of **Actor-Critic with
 Experience Replay and Pessimistic Critic** algorithm.
 Implementation of **Actor-Critic with Experience Replay** is also present.
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
3. While in the repository's root directory, install the requirements:
```shell script
pip install -r requirements.txt
``` 

4. Run the agent:
```shell script
python run.py {args...}
``` 

## Example runs
```shell script
python3 acer/run.py --env_name LunarLander-v2 --gamma 0.99 --alpha 0.9 --p 0.1 --b 3 --c0 0.3 \
    --c 50 --actor_lr 0.001 --critic_lr 0.002  --actor_layers 60 --critic_layers 100 \
    --memory_size 1000000 --num_parallel_envs 10 --actor_adam_beta1 0.9 --actor_adam_beta2 0.999 \
    --actor_adam_epsilon 0.1 --critic_adam_beta1 0.9 --critic_adam_beta2 0.999 \
    --critic_adam_epsilon 0.1 --actor_beta_penalty 0.001 --standardize_obs \
    --evaluate_time_steps_interval 1500 --num_evaluation_runs 10 \
```
```shell script
python run.py --env_name Pendulum-v0 --gamma 0.95 --alpha 0.9 \
    --p 0.1 --b 5 --c0 0.3 --c 50 --actor_lr 0.001 --critic_lr 0.002  \
    --actor_layers 20 --critic_layers 50 --memory_size 1000000 \
    --num_parallel_envs 10 --actor_adam_beta1 0.9 --actor_adam_beta2 0.999 \
    --actor_adam_epsilon 0.1 --critic_adam_beta1 0.9 --critic_adam_beta2 0.999 \
    --critic_adam_epsilon 0.1 --actor_beta_penalty 0.1
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

## Notes
1. *num_parallel_envs* denotes number of parallel OpenAI Gym environments. Right now,
no multithreading/multiprocessing is implemented (actions are executed sequentially in every env).
Asynchronous workers will be required to handle more complex environments
 (where *.step()* call is computationally expensive). 
 Consider [Ray](https://github.com/ray-project/ray)  
2. New replay buffer is created for each environment. One trajectory per buffer is sampled
 in the experience replay process. Those trajectories are combined into single batch (i.e.
 single batch consists of *num_parallel_envs* trajectories).
 
 
 #### Improvements to be considered
 * Multivariate Normal distribution diagonal as learned parameter  
    * PoC experiments resulted in fast converge to some low values and
    further in no exploration and learning at all. 
    Maybe with entropy penalty the mechanism can work.
 * more meaningful statistics for TensorBoard
 * reward clipping / scaling
 * sampling more trajectories per environment
 * weights initialization (*normalized column* initialization from the OpenAI Baselines
 is used right now, some sources propose Orthogonal initialization)
 * probably, some optimizations around *@tf.function* usage can be done
 
 
 ## References
 
TBA
 
Wawrzyński, Paweł.
*Real-time reinforcement learning by sequential actor–critics
and experience replay.*
Neural Networks 22.10 (2009): 1484-1497.

Wawrzyński, Paweł, and Ajay Kumar Tanwani.
*Autonomous reinforcement learning with experience replay.*
Neural Networks 41 (2013): 156-167.



