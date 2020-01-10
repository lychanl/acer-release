# Actor-Critic with experience replay

## Example run
```
python3 run.py --env_name Pendulum-v0 --gamma 0.95 --alpha 0.9 --p 0.1 --b 3 --min_steps_learn 100 --c 100 --actor_lr 0.001 --critic_lr 0.001  --actor_layers 20 --critic_layers 50 --memory_size 1000000 --num_parallel_envs 1 --actor_beta1 0.9 --actor_beta2 0.999 --actor_epsilon 0.00001 --critic_beta1 0.9 --critic_beta2 0.999 --critic_epsilon 0.00001 --actor_beta_penalty 0.1 --max_episodes 200 --std 0.4
```

## Notes
1. *num_parallel_envs* denotes number of parallel OpenAI Gym environments. Right now,
no multithreading/multiprocessing is implemented (actions are executed sequentially in every env).
Asynchronous workers will be required to handle more complex environments
 (where *.step()* call is computationally expensive).  
 New replay buffer is created for each environment. One trajectory per buffer is being sampled
 in the experience replay process. Those trajectories are combined into single batch (single batch
 consists of *num_parallel_envs* trajectories).
 2. *tf.function* signatures are now hardcoded to handle *Pendulum-v0* env only. Those signatures
 are required to tell Tensorflow not to *retrace* (retracing results in 
 creation of redundant computational graphs)
