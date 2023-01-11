from typing import Dict, Tuple
import argparse
import numpy as np
import wandb
import math 
import os
import ray
from datetime import datetime
from ray import air, tune
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import Episode, RolloutWorker
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch

class CustomCallbacks(DefaultCallbacks):
	
	def on_algorithm_init(self, *, algorithm, **kwargs):
		# wandb server 
		# wandb.init(project='RLlib')
		now = datetime.now()
		name = now.strftime("_%m_%d_%Y_%H_%M_%S")
		# wandb.run.name='PPO'+name
		# monitoring 
		self.time=0

	# def on_episode_start(
	# 	self,
	# 	*,
	# 	worker: RolloutWorker,
	# 	base_env: BaseEnv,
	# 	policies: Dict[str, Policy],
	# 	episode: Episode,
	# 	env_index: int,
	# 	**kwargs
	# ):
	# 	# Make sure this episode has just been started (only initial obs
	# 	# logged so far).
	# 	assert episode.length == 0, (
	# 		"ERROR: `on_episode_start()` callback should be called right "
	# 		"after env reset!"
	# 	)
	# 	print("episode {} (env-idx={}) started.".format(episode.episode_id, env_index))
		# episode.user_data["pole_angles"] = []
		# episode.hist_data["pole_angles"] = []

	# def on_episode_step(
	# 	self,
	# 	*,
	# 	worker: RolloutWorker,
	# 	base_env: BaseEnv,
	# 	policies: Dict[str, Policy],
	# 	episode: Episode,
	# 	env_index: int,
	# 	**kwargs
	# ):
	# 	# Make sure this episode is ongoing.
	# 	assert episode.length > 0, (
	# 		"ERROR: `on_episode_step()` callback should not be called right "
	# 		"after env reset!"
	# 	)
		# print("Last info {}".format(episode.last_info_for()))
		# pole_angle = abs(episode.last_observation_for()[2])
		# raw_angle = abs(episode.last_raw_obs_for()[2])
		# assert pole_angle == raw_angle
		# episode.user_data["pole_angles"].append(pole_angle)

	# def on_episode_end(
	# 	self,
	# 	*,
	# 	worker: RolloutWorker,
	# 	base_env: BaseEnv,
	# 	policies: Dict[str, Policy],
	# 	episode: Episode,
	# 	env_index: int,
	# 	**kwargs
	# ):
	# 	# Check if there are multiple episodes in a batch, i.e.
	# 	# "batch_mode": "truncate_episodes".
	# 	if worker.policy_config["batch_mode"] == "truncate_episodes":
	# 		# Make sure this episode is really done.
	# 		assert episode.batch_builder.policy_collectors["default_policy"].batches[
	# 			-1
	# 		]["dones"][-1], (
	# 			"ERROR: `on_episode_end()` should only be called "
	# 			"after episode is done!"
	# 		)
	# 	print('Total reward episode : ', episode.total_reward)
		# pole_angle = np.mean(episode.user_data["pole_angles"])
		# print(
		# 	"episode {} (env-idx={}) ended with length {} and pole "
		# 	"angles {}".format(
		# 		episode.episode_id, env_index, episode.length, pole_angle
		# 	)
		# )
		# episode.custom_metrics["pole_angle"] = pole_angle
		# episode.hist_data["pole_angles"] = episode.user_data["pole_angles"]

	# def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch, **kwargs):
	# 	print("returned sample batch of size {}".format(samples.count))

	# def on_train_result(self, *, algorithm, result: dict, **kwargs):
	# 	print(
	# 		"Algorithm.train() result: {} -> {} episodes".format(
	# 			algorithm, result["episodes_this_iter"]
	# 		)
	# 	)
	# 	# you can mutate the result dict to add new fields to return
	# 	result["callback_ok"] = True
	# 	print("TRAINING COMPLETED !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
		# print('VAR : ',self.VAR)

	# def on_learn_on_batch(
	# 	self, *, policy: Policy, train_batch: SampleBatch, result: dict, **kwargs
	# ) -> None:
	# 	result["sum_actions_in_train_batch"] = np.sum(train_batch["actions"])
	# 	print(
	# 		"policy.learn_on_batch() result: {} -> sum actions: {}".format(
	# 			policy, result["sum_actions_in_train_batch"]
	# 		)
	# 	)

	# def on_postprocess_trajectory(
	# 	self,
	# 	*,
	# 	worker: RolloutWorker,
	# 	episode: Episode,
	# 	agent_id: str,
	# 	policy_id: str,
	# 	policies: Dict[str, Policy],
	# 	postprocessed_batch: SampleBatch,
	# 	original_batches: Dict[str, Tuple[Policy, SampleBatch]],
	# 	**kwargs
	# ):
	# 	print("postprocessed {} steps".format(postprocessed_batch.count))
	# 	if "num_batches" not in episode.custom_metrics:
	# 		episode.custom_metrics["num_batches"] = 0
	# 	episode.custom_metrics["num_batches"] += 1


	def on_train_result(self,*,algorithm,result: dict,**kwargs,) -> None:
		"""Called at the end of Algorithm.train().

		Args:
			algorithm: Current Algorithm instance.
			result: Dict of results returned from Algorithm.train() call.
				You can mutate this object to add additional metrics.
			kwargs: Forward compatibility placeholder.
		"""
		num_healthy_workers=result['num_healthy_workers']
		# wandb
		data={}
		recursif_dict(result)
		# time 
		data['training/time']=self.time
		episode_reward=result['sampler_results']['hist_stats']['episode_reward']
		# reward
		if len(episode_reward)>0 :
			current_episode_reward=episode_reward[-num_healthy_workers:]
			# mean reward 
			data['training/episodic_reward_mean']=np.mean(current_episode_reward)
			# max
			data['training/episodic_reward_max']=np.max(current_episode_reward)
			# min
			data['training/episodic_reward_min']=np.min(current_episode_reward)
			# std 
			data['training/episodic_reward_std']=np.std(current_episode_reward)

		if 'default_policy' in result['info']['learner'].keys():
			loss_data=result['info']['learner']['default_policy']['learner_stats']
			# policy_loss
			data['training/policy_loss']=loss_data['policy_loss']
			# total_loss
			data['training/total_loss']=loss_data['total_loss']
			# vf_explained_var
			data['training/vf_explained_var']=loss_data['vf_explained_var']
			# vf_loss
			data['training/vf_loss']=loss_data['vf_loss']
		# log 
		# wandb.log(data)

def recursif_dict(d,k=0):
	if isinstance(d,dict) :
		for key in d.keys():
			print(k*str('   ')+key) if isinstance(d[key],dict) else print(k*str('   ')+key+' : '+str(d[key]))
			recursif_dict(d[key],k+1)