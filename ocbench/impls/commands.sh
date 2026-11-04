# block-single-task1-v0 (BC)
python main.py --env_name=block-single-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# block-single-task1-v0 (FBC)
python main.py --env_name=block-single-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# block-single-task1-v0 (FSQBC)
python main.py --env_name=block-single-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# block-double-task1-v0 (BC)
python main.py --env_name=block-double-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# block-double-task1-v0 (FBC)
python main.py --env_name=block-double-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# block-double-task1-v0 (FSQBC)
python main.py --env_name=block-double-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# block-double-task2-v0 (BC)
python main.py --env_name=block-double-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# block-double-task2-v0 (FBC)
python main.py --env_name=block-double-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# block-double-task2-v0 (FSQBC)
python main.py --env_name=block-double-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# block-triple-task1-v0 (BC)
python main.py --env_name=block-triple-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# block-triple-task1-v0 (FBC)
python main.py --env_name=block-triple-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# block-triple-task1-v0 (FSQBC)
python main.py --env_name=block-triple-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# block-triple-task2-v0 (BC)
python main.py --env_name=block-triple-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# block-triple-task2-v0 (FBC)
python main.py --env_name=block-triple-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# block-triple-task2-v0 (FSQBC)
python main.py --env_name=block-triple-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# block-quadruple-task1-v0 (BC)
python main.py --env_name=block-quadruple-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# block-quadruple-task1-v0 (FBC)
python main.py --env_name=block-quadruple-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# block-quadruple-task1-v0 (FSQBC)
python main.py --env_name=block-quadruple-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# block-quadruple-task2-v0 (BC)
python main.py --env_name=block-quadruple-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# block-quadruple-task2-v0 (FBC)
python main.py --env_name=block-quadruple-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# block-quadruple-task2-v0 (FSQBC)
python main.py --env_name=block-quadruple-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# block-quadruple-task3-v0 (BC)
python main.py --env_name=block-quadruple-task3-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# block-quadruple-task3-v0 (FBC)
python main.py --env_name=block-quadruple-task3-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# block-quadruple-task3-v0 (FSQBC)
python main.py --env_name=block-quadruple-task3-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# chamber-easy-task1-v0 (BC)
python main.py --env_name=chamber-easy-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# chamber-easy-task1-v0 (FBC)
python main.py --env_name=chamber-easy-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# chamber-easy-task1-v0 (FSQBC)
python main.py --env_name=chamber-easy-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# chamber-easy-task2-v0 (BC)
python main.py --env_name=chamber-easy-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# chamber-easy-task2-v0 (FBC)
python main.py --env_name=chamber-easy-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# chamber-easy-task2-v0 (FSQBC)
python main.py --env_name=chamber-easy-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# chamber-easy-task3-v0 (BC)
python main.py --env_name=chamber-easy-task3-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# chamber-easy-task3-v0 (FBC)
python main.py --env_name=chamber-easy-task3-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# chamber-easy-task3-v0 (FSQBC)
python main.py --env_name=chamber-easy-task3-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# chamber-medium-task1-v0 (BC)
python main.py --env_name=chamber-medium-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# chamber-medium-task1-v0 (FBC)
python main.py --env_name=chamber-medium-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# chamber-medium-task1-v0 (FSQBC)
python main.py --env_name=chamber-medium-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# chamber-medium-task2-v0 (BC)
python main.py --env_name=chamber-medium-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# chamber-medium-task2-v0 (FBC)
python main.py --env_name=chamber-medium-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# chamber-medium-task2-v0 (FSQBC)
python main.py --env_name=chamber-medium-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# chamber-hard-task1-v0 (BC)
python main.py --env_name=chamber-hard-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# chamber-hard-task1-v0 (FBC)
python main.py --env_name=chamber-hard-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# chamber-hard-task1-v0 (FSQBC)
python main.py --env_name=chamber-hard-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# chamber-hard-task2-v0 (BC)
python main.py --env_name=chamber-hard-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# chamber-hard-task2-v0 (FBC)
python main.py --env_name=chamber-hard-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# chamber-hard-task2-v0 (FSQBC)
python main.py --env_name=chamber-hard-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# switch-3x3-task1-v0 (BC)
python main.py --env_name=switch-3x3-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# switch-3x3-task1-v0 (FBC)
python main.py --env_name=switch-3x3-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# switch-3x3-task1-v0 (FSQBC)
python main.py --env_name=switch-3x3-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# switch-3x3-task2-v0 (BC)
python main.py --env_name=switch-3x3-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# switch-3x3-task2-v0 (FBC)
python main.py --env_name=switch-3x3-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# switch-3x3-task2-v0 (FSQBC)
python main.py --env_name=switch-3x3-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# switch-4x4-task1-v0 (BC)
python main.py --env_name=switch-4x4-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# switch-4x4-task1-v0 (FBC)
python main.py --env_name=switch-4x4-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# switch-4x4-task1-v0 (FSQBC)
python main.py --env_name=switch-4x4-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# switch-4x4-task2-v0 (BC)
python main.py --env_name=switch-4x4-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# switch-4x4-task2-v0 (FBC)
python main.py --env_name=switch-4x4-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# switch-4x4-task2-v0 (FSQBC)
python main.py --env_name=switch-4x4-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# switch-5x5-task1-v0 (BC)
python main.py --env_name=switch-5x5-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# switch-5x5-task1-v0 (FBC)
python main.py --env_name=switch-5x5-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# switch-5x5-task1-v0 (FSQBC)
python main.py --env_name=switch-5x5-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# switch-5x5-task2-v0 (BC)
python main.py --env_name=switch-5x5-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# switch-5x5-task2-v0 (FBC)
python main.py --env_name=switch-5x5-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# switch-5x5-task2-v0 (FSQBC)
python main.py --env_name=switch-5x5-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# hanoi-single-task1-v0 (BC)
python main.py --env_name=hanoi-single-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# hanoi-single-task1-v0 (FBC)
python main.py --env_name=hanoi-single-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# hanoi-single-task1-v0 (FSQBC)
python main.py --env_name=hanoi-single-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# hanoi-single-task2-v0 (BC)
python main.py --env_name=hanoi-single-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# hanoi-single-task2-v0 (FBC)
python main.py --env_name=hanoi-single-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# hanoi-single-task2-v0 (FSQBC)
python main.py --env_name=hanoi-single-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# hanoi-double-task1-v0 (BC)
python main.py --env_name=hanoi-double-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# hanoi-double-task1-v0 (FBC)
python main.py --env_name=hanoi-double-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# hanoi-double-task1-v0 (FSQBC)
python main.py --env_name=hanoi-double-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# hanoi-double-task2-v0 (BC)
python main.py --env_name=hanoi-double-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# hanoi-double-task2-v0 (FBC)
python main.py --env_name=hanoi-double-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# hanoi-double-task2-v0 (FSQBC)
python main.py --env_name=hanoi-double-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# hanoi-triple-task1-v0 (BC)
python main.py --env_name=hanoi-triple-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# hanoi-triple-task1-v0 (FBC)
python main.py --env_name=hanoi-triple-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# hanoi-triple-task1-v0 (FSQBC)
python main.py --env_name=hanoi-triple-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# hanoi-triple-task2-v0 (BC)
python main.py --env_name=hanoi-triple-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# hanoi-triple-task2-v0 (FBC)
python main.py --env_name=hanoi-triple-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# hanoi-triple-task2-v0 (FSQBC)
python main.py --env_name=hanoi-triple-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# bowling-task1-v0 (BC)
python main.py --env_name=bowling-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=0 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp --agent.loss=l2
# bowling-task1-v0 (FBC)
python main.py --env_name=bowling-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=0 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(4096, 4096, 4096, 4096, 4096, 4096, 4096, 4096)" --agent.mlp_class=resmlp
# bowling-task1-v0 (FSQBC)
python main.py --env_name=bowling-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=25 --success_only=0 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.actor_hidden_dims="(2048, 2048, 2048, 2048, 2048, 2048, 2048, 2048)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# block-lite-single-task1-v0 (BC)
python main.py --env_name=block-lite-single-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# block-lite-single-task1-v0 (FBC)
python main.py --env_name=block-lite-single-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# block-lite-single-task1-v0 (FSQBC)
python main.py --env_name=block-lite-single-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# block-lite-double-task1-v0 (BC)
python main.py --env_name=block-lite-double-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# block-lite-double-task1-v0 (FBC)
python main.py --env_name=block-lite-double-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# block-lite-double-task1-v0 (FSQBC)
python main.py --env_name=block-lite-double-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# block-lite-double-task2-v0 (BC)
python main.py --env_name=block-lite-double-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# block-lite-double-task2-v0 (FBC)
python main.py --env_name=block-lite-double-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# block-lite-double-task2-v0 (FSQBC)
python main.py --env_name=block-lite-double-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# block-lite-triple-task1-v0 (BC)
python main.py --env_name=block-lite-triple-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# block-lite-triple-task1-v0 (FBC)
python main.py --env_name=block-lite-triple-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# block-lite-triple-task1-v0 (FSQBC)
python main.py --env_name=block-lite-triple-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# block-lite-triple-task2-v0 (BC)
python main.py --env_name=block-lite-triple-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# block-lite-triple-task2-v0 (FBC)
python main.py --env_name=block-lite-triple-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# block-lite-triple-task2-v0 (FSQBC)
python main.py --env_name=block-lite-triple-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# block-lite-quadruple-task1-v0 (BC)
python main.py --env_name=block-lite-quadruple-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# block-lite-quadruple-task1-v0 (FBC)
python main.py --env_name=block-lite-quadruple-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# block-lite-quadruple-task1-v0 (FSQBC)
python main.py --env_name=block-lite-quadruple-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# block-lite-quadruple-task2-v0 (BC)
python main.py --env_name=block-lite-quadruple-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# block-lite-quadruple-task2-v0 (FBC)
python main.py --env_name=block-lite-quadruple-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# block-lite-quadruple-task2-v0 (FSQBC)
python main.py --env_name=block-lite-quadruple-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# block-lite-quadruple-task3-v0 (BC)
python main.py --env_name=block-lite-quadruple-task3-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# block-lite-quadruple-task3-v0 (FBC)
python main.py --env_name=block-lite-quadruple-task3-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# block-lite-quadruple-task3-v0 (FSQBC)
python main.py --env_name=block-lite-quadruple-task3-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# chamber-lite-easy-task1-v0 (BC)
python main.py --env_name=chamber-lite-easy-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# chamber-lite-easy-task1-v0 (FBC)
python main.py --env_name=chamber-lite-easy-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# chamber-lite-easy-task1-v0 (FSQBC)
python main.py --env_name=chamber-lite-easy-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# chamber-lite-easy-task2-v0 (BC)
python main.py --env_name=chamber-lite-easy-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# chamber-lite-easy-task2-v0 (FBC)
python main.py --env_name=chamber-lite-easy-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# chamber-lite-easy-task2-v0 (FSQBC)
python main.py --env_name=chamber-lite-easy-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# chamber-lite-easy-task3-v0 (BC)
python main.py --env_name=chamber-lite-easy-task3-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# chamber-lite-easy-task3-v0 (FBC)
python main.py --env_name=chamber-lite-easy-task3-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# chamber-lite-easy-task3-v0 (FSQBC)
python main.py --env_name=chamber-lite-easy-task3-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# chamber-lite-medium-task1-v0 (BC)
python main.py --env_name=chamber-lite-medium-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# chamber-lite-medium-task1-v0 (FBC)
python main.py --env_name=chamber-lite-medium-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# chamber-lite-medium-task1-v0 (FSQBC)
python main.py --env_name=chamber-lite-medium-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# chamber-lite-medium-task2-v0 (BC)
python main.py --env_name=chamber-lite-medium-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# chamber-lite-medium-task2-v0 (FBC)
python main.py --env_name=chamber-lite-medium-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# chamber-lite-medium-task2-v0 (FSQBC)
python main.py --env_name=chamber-lite-medium-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# chamber-lite-hard-task1-v0 (BC)
python main.py --env_name=chamber-lite-hard-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# chamber-lite-hard-task1-v0 (FBC)
python main.py --env_name=chamber-lite-hard-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# chamber-lite-hard-task1-v0 (FSQBC)
python main.py --env_name=chamber-lite-hard-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# chamber-lite-hard-task2-v0 (BC)
python main.py --env_name=chamber-lite-hard-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# chamber-lite-hard-task2-v0 (FBC)
python main.py --env_name=chamber-lite-hard-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# chamber-lite-hard-task2-v0 (FSQBC)
python main.py --env_name=chamber-lite-hard-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# switch-lite-3x3-task1-v0 (BC)
python main.py --env_name=switch-lite-3x3-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# switch-lite-3x3-task1-v0 (FBC)
python main.py --env_name=switch-lite-3x3-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# switch-lite-3x3-task1-v0 (FSQBC)
python main.py --env_name=switch-lite-3x3-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# switch-lite-3x3-task2-v0 (BC)
python main.py --env_name=switch-lite-3x3-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# switch-lite-3x3-task2-v0 (FBC)
python main.py --env_name=switch-lite-3x3-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# switch-lite-3x3-task2-v0 (FSQBC)
python main.py --env_name=switch-lite-3x3-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# switch-lite-4x4-task1-v0 (BC)
python main.py --env_name=switch-lite-4x4-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# switch-lite-4x4-task1-v0 (FBC)
python main.py --env_name=switch-lite-4x4-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# switch-lite-4x4-task1-v0 (FSQBC)
python main.py --env_name=switch-lite-4x4-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# switch-lite-4x4-task2-v0 (BC)
python main.py --env_name=switch-lite-4x4-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# switch-lite-4x4-task2-v0 (FBC)
python main.py --env_name=switch-lite-4x4-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# switch-lite-4x4-task2-v0 (FSQBC)
python main.py --env_name=switch-lite-4x4-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# switch-lite-5x5-task1-v0 (BC)
python main.py --env_name=switch-lite-5x5-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# switch-lite-5x5-task1-v0 (FBC)
python main.py --env_name=switch-lite-5x5-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# switch-lite-5x5-task1-v0 (FSQBC)
python main.py --env_name=switch-lite-5x5-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# switch-lite-5x5-task2-v0 (BC)
python main.py --env_name=switch-lite-5x5-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# switch-lite-5x5-task2-v0 (FBC)
python main.py --env_name=switch-lite-5x5-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# switch-lite-5x5-task2-v0 (FSQBC)
python main.py --env_name=switch-lite-5x5-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# hanoi-lite-single-task1-v0 (BC)
python main.py --env_name=hanoi-lite-single-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# hanoi-lite-single-task1-v0 (FBC)
python main.py --env_name=hanoi-lite-single-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# hanoi-lite-single-task1-v0 (FSQBC)
python main.py --env_name=hanoi-lite-single-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# hanoi-lite-single-task2-v0 (BC)
python main.py --env_name=hanoi-lite-single-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# hanoi-lite-single-task2-v0 (FBC)
python main.py --env_name=hanoi-lite-single-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# hanoi-lite-single-task2-v0 (FSQBC)
python main.py --env_name=hanoi-lite-single-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# hanoi-lite-double-task1-v0 (BC)
python main.py --env_name=hanoi-lite-double-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# hanoi-lite-double-task1-v0 (FBC)
python main.py --env_name=hanoi-lite-double-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# hanoi-lite-double-task1-v0 (FSQBC)
python main.py --env_name=hanoi-lite-double-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# hanoi-lite-double-task2-v0 (BC)
python main.py --env_name=hanoi-lite-double-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# hanoi-lite-double-task2-v0 (FBC)
python main.py --env_name=hanoi-lite-double-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# hanoi-lite-double-task2-v0 (FSQBC)
python main.py --env_name=hanoi-lite-double-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# hanoi-lite-triple-task1-v0 (BC)
python main.py --env_name=hanoi-lite-triple-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# hanoi-lite-triple-task1-v0 (FBC)
python main.py --env_name=hanoi-lite-triple-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# hanoi-lite-triple-task1-v0 (FSQBC)
python main.py --env_name=hanoi-lite-triple-task1-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True

# hanoi-lite-triple-task2-v0 (BC)
python main.py --env_name=hanoi-lite-triple-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/bc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp --agent.loss=l2
# hanoi-lite-triple-task2-v0 (FBC)
python main.py --env_name=hanoi-lite-triple-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fbc.py --agent.lr=0.0001 --agent.batch_size=1024 --agent.actor_hidden_dims="(1024, 1024, 1024, 1024, 1024, 1024, 1024, 1024)" --agent.mlp_class=resmlp
# hanoi-lite-triple-task2-v0 (FSQBC)
python main.py --env_name=hanoi-lite-triple-task2-v0 --offline_steps=2000000 --eval_interval=50000 --eval_episodes=500 --action_chunk_length=3 --success_only=1 --agent=agents/fsqbc.py --agent.lr=0.0001 --agent.batch_size=256 --agent.fsq_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.actor_hidden_dims="(512, 512, 512, 512, 512, 512, 512, 512)" --agent.mlp_class=resmlp --agent.num_tokens=4 --agent.fsq_levels="(8, 5, 5, 5)" --agent.state_cond=True
