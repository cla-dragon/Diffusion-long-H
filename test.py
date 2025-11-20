import ogbench

env, dataset, _ = ogbench.make_env_and_datasets(
    "cube-single-play-v0",
    compact_dataset=True,
)
print(env.action_space)