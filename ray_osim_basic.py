
import ray
from ray.tune.registry import register_env
from ray.tune import run_experiments
import numpy as np
from ray.rllib.models import ModelCatalog
from ray.rllib.models.preprocessors import Preprocessor
from osim.env import ProstheticsEnv
import gym

env_name = "osim_env"


class OsimPreprocessor(Preprocessor):
    def _init(self):
        self.dict_keys = ["body_acc_calcn_l_0",
"body_acc_calcn_l_1",
"body_acc_calcn_l_2",
"body_acc_femur_l_0",
"body_acc_femur_l_1",
"body_acc_femur_l_2",
"body_acc_femur_r_0",
"body_acc_femur_r_1",
"body_acc_femur_r_2",
"body_acc_head_0",
"body_acc_head_1",
"body_acc_head_2",
"body_acc_pelvis_0",
"body_acc_pelvis_1",
"body_acc_pelvis_2",
"body_acc_pros_foot_r_0",
"body_acc_pros_foot_r_1",
"body_acc_pros_foot_r_2",
"body_acc_pros_tibia_r_0",
"body_acc_pros_tibia_r_1",
"body_acc_pros_tibia_r_2",
"body_acc_talus_l_0",
"body_acc_talus_l_1",
"body_acc_talus_l_2",
"body_acc_tibia_l_0",
"body_acc_tibia_l_1",
"body_acc_tibia_l_2",
"body_acc_toes_l_0",
"body_acc_toes_l_1",
"body_acc_toes_l_2",
"body_acc_torso_0",
"body_acc_torso_1",
"body_acc_torso_2",
"body_acc_rot_calcn_l_0",
"body_acc_rot_calcn_l_1",
"body_acc_rot_calcn_l_2",
"body_acc_rot_femur_l_0",
"body_acc_rot_femur_l_1",
"body_acc_rot_femur_l_2",
"body_acc_rot_femur_r_0",
"body_acc_rot_femur_r_1",
"body_acc_rot_femur_r_2",
"body_acc_rot_head_0",
"body_acc_rot_head_1",
"body_acc_rot_head_2",
"body_acc_rot_pelvis_0",
"body_acc_rot_pelvis_1",
"body_acc_rot_pelvis_2",
"body_acc_rot_pros_foot_r_0",
"body_acc_rot_pros_foot_r_1",
"body_acc_rot_pros_foot_r_2",
"body_acc_rot_pros_tibia_r_0",
"body_acc_rot_pros_tibia_r_1",
"body_acc_rot_pros_tibia_r_2",
"body_acc_rot_talus_l_0",
"body_acc_rot_talus_l_1",
"body_acc_rot_talus_l_2",
"body_acc_rot_tibia_l_0",
"body_acc_rot_tibia_l_1",
"body_acc_rot_tibia_l_2",
"body_acc_rot_toes_l_0",
"body_acc_rot_toes_l_1",
"body_acc_rot_toes_l_2",
"body_acc_rot_torso_0",
"body_acc_rot_torso_1",
"body_acc_rot_torso_2",
"body_pos_calcn_l_0",
"body_pos_calcn_l_1",
"body_pos_calcn_l_2",
"body_pos_femur_l_0",
"body_pos_femur_l_1",
"body_pos_femur_l_2",
"body_pos_femur_r_0",
"body_pos_femur_r_1",
"body_pos_femur_r_2",
"body_pos_head_0",
"body_pos_head_1",
"body_pos_head_2",
"body_pos_pelvis_0",
"body_pos_pelvis_1",
"body_pos_pelvis_2",
"body_pos_pros_foot_r_0",
"body_pos_pros_foot_r_1",
"body_pos_pros_foot_r_2",
"body_pos_pros_tibia_r_0",
"body_pos_pros_tibia_r_1",
"body_pos_pros_tibia_r_2",
"body_pos_talus_l_0",
"body_pos_talus_l_1",
"body_pos_talus_l_2",
"body_pos_tibia_l_0",
"body_pos_tibia_l_1",
"body_pos_tibia_l_2",
"body_pos_toes_l_0",
"body_pos_toes_l_1",
"body_pos_toes_l_2",
"body_pos_torso_0",
"body_pos_torso_1",
"body_pos_torso_2",
"body_pos_rot_calcn_l_0",
"body_pos_rot_calcn_l_1",
"body_pos_rot_calcn_l_2",
"body_pos_rot_femur_l_0",
"body_pos_rot_femur_l_1",
"body_pos_rot_femur_l_2",
"body_pos_rot_femur_r_0",
"body_pos_rot_femur_r_1",
"body_pos_rot_femur_r_2",
"body_pos_rot_head_0",
"body_pos_rot_head_1",
"body_pos_rot_head_2",
"body_pos_rot_pelvis_0",
"body_pos_rot_pelvis_1",
"body_pos_rot_pelvis_2",
"body_pos_rot_pros_foot_r_0",
"body_pos_rot_pros_foot_r_1",
"body_pos_rot_pros_foot_r_2",
"body_pos_rot_pros_tibia_r_0",
"body_pos_rot_pros_tibia_r_1",
"body_pos_rot_pros_tibia_r_2",
"body_pos_rot_talus_l_0",
"body_pos_rot_talus_l_1",
"body_pos_rot_talus_l_2",
"body_pos_rot_tibia_l_0",
"body_pos_rot_tibia_l_1",
"body_pos_rot_tibia_l_2",
"body_pos_rot_toes_l_0",
"body_pos_rot_toes_l_1",
"body_pos_rot_toes_l_2",
"body_pos_rot_torso_0",
"body_pos_rot_torso_1",
"body_pos_rot_torso_2",
"body_vel_calcn_l_0",
"body_vel_calcn_l_1",
"body_vel_calcn_l_2",
"body_vel_femur_l_0",
"body_vel_femur_l_1",
"body_vel_femur_l_2",
"body_vel_femur_r_0",
"body_vel_femur_r_1",
"body_vel_femur_r_2",
"body_vel_head_0",
"body_vel_head_1",
"body_vel_head_2",
"body_vel_pelvis_0",
"body_vel_pelvis_1",
"body_vel_pelvis_2",
"body_vel_pros_foot_r_0",
"body_vel_pros_foot_r_1",
"body_vel_pros_foot_r_2",
"body_vel_pros_tibia_r_0",
"body_vel_pros_tibia_r_1",
"body_vel_pros_tibia_r_2",
"body_vel_talus_l_0",
"body_vel_talus_l_1",
"body_vel_talus_l_2",
"body_vel_tibia_l_0",
"body_vel_tibia_l_1",
"body_vel_tibia_l_2",
"body_vel_toes_l_0",
"body_vel_toes_l_1",
"body_vel_toes_l_2",
"body_vel_torso_0",
"body_vel_torso_1",
"body_vel_torso_2",
"body_vel_rot_calcn_l_0",
"body_vel_rot_calcn_l_1",
"body_vel_rot_calcn_l_2",
"body_vel_rot_femur_l_0",
"body_vel_rot_femur_l_1",
"body_vel_rot_femur_l_2",
"body_vel_rot_femur_r_0",
"body_vel_rot_femur_r_1",
"body_vel_rot_femur_r_2",
"body_vel_rot_head_0",
"body_vel_rot_head_1",
"body_vel_rot_head_2",
"body_vel_rot_pelvis_0",
"body_vel_rot_pelvis_1",
"body_vel_rot_pelvis_2",
"body_vel_rot_pros_foot_r_0",
"body_vel_rot_pros_foot_r_1",
"body_vel_rot_pros_foot_r_2",
"body_vel_rot_pros_tibia_r_0",
"body_vel_rot_pros_tibia_r_1",
"body_vel_rot_pros_tibia_r_2",
"body_vel_rot_talus_l_0",
"body_vel_rot_talus_l_1",
"body_vel_rot_talus_l_2",
"body_vel_rot_tibia_l_0",
"body_vel_rot_tibia_l_1",
"body_vel_rot_tibia_l_2",
"body_vel_rot_toes_l_0",
"body_vel_rot_toes_l_1",
"body_vel_rot_toes_l_2",
"body_vel_rot_torso_0",
"body_vel_rot_torso_1",
"body_vel_rot_torso_2",
"forces_AnkleLimit_l_0",
"forces_AnkleLimit_l_1",
"forces_AnkleLimit_r_0",
"forces_AnkleLimit_r_1",
"forces_HipAddLimit_l_0",
"forces_HipAddLimit_l_1",
"forces_HipAddLimit_r_0",
"forces_HipAddLimit_r_1",
"forces_HipLimit_l_0",
"forces_HipLimit_l_1",
"forces_HipLimit_r_0",
"forces_HipLimit_r_1",
"forces_KneeLimit_l_0",
"forces_KneeLimit_l_1",
"forces_KneeLimit_r_0",
"forces_KneeLimit_r_1",
"forces_abd_l_0",
"forces_abd_r_0",
"forces_add_l_0",
"forces_add_r_0",
"forces_ankleSpring_0",
"forces_bifemsh_l_0",
"forces_bifemsh_r_0",
"forces_foot_l_0",
"forces_foot_l_1",
"forces_foot_l_2",
"forces_foot_l_3",
"forces_foot_l_4",
"forces_foot_l_5",
"forces_foot_l_6",
"forces_foot_l_7",
"forces_foot_l_8",
"forces_foot_l_9",
"forces_foot_l_10",
"forces_foot_l_11",
"forces_foot_l_12",
"forces_foot_l_13",
"forces_foot_l_14",
"forces_foot_l_15",
"forces_foot_l_16",
"forces_foot_l_17",
"forces_foot_l_18",
"forces_foot_l_19",
"forces_foot_l_20",
"forces_foot_l_21",
"forces_foot_l_22",
"forces_foot_l_23",
"forces_gastroc_l_0",
"forces_glut_max_l_0",
"forces_glut_max_r_0",
"forces_hamstrings_l_0",
"forces_hamstrings_r_0",
"forces_iliopsoas_l_0",
"forces_iliopsoas_r_0",
"forces_pros_foot_r_0_0",
"forces_pros_foot_r_0_1",
"forces_pros_foot_r_0_2",
"forces_pros_foot_r_0_3",
"forces_pros_foot_r_0_4",
"forces_pros_foot_r_0_5",
"forces_pros_foot_r_0_6",
"forces_pros_foot_r_0_7",
"forces_pros_foot_r_0_8",
"forces_pros_foot_r_0_9",
"forces_pros_foot_r_0_10",
"forces_pros_foot_r_0_11",
"forces_pros_foot_r_0_12",
"forces_pros_foot_r_0_13",
"forces_pros_foot_r_0_14",
"forces_pros_foot_r_0_15",
"forces_pros_foot_r_0_16",
"forces_pros_foot_r_0_17",
"forces_rect_fem_l_0",
"forces_rect_fem_r_0",
"forces_soleus_l_0",
"forces_tib_ant_l_0",
"forces_vasti_l_0",
"forces_vasti_r_0",
"joint_acc_ankle_l_0",
"joint_acc_ankle_r_0",
"joint_acc_back_0",
"joint_acc_ground_pelvis_0",
"joint_acc_ground_pelvis_1",
"joint_acc_ground_pelvis_2",
"joint_acc_ground_pelvis_3",
"joint_acc_ground_pelvis_4",
"joint_acc_ground_pelvis_5",
"joint_acc_hip_l_0",
"joint_acc_hip_l_1",
"joint_acc_hip_l_2",
"joint_acc_hip_r_0",
"joint_acc_hip_r_1",
"joint_acc_hip_r_2",
"joint_acc_knee_l_0",
"joint_acc_knee_r_0",
"joint_pos_ankle_l_0",
"joint_pos_ankle_r_0",
"joint_pos_back_0",
"joint_pos_ground_pelvis_0",
"joint_pos_ground_pelvis_1",
"joint_pos_ground_pelvis_2",
"joint_pos_ground_pelvis_3",
"joint_pos_ground_pelvis_4",
"joint_pos_ground_pelvis_5",
"joint_pos_hip_l_0",
"joint_pos_hip_l_1",
"joint_pos_hip_l_2",
"joint_pos_hip_r_0",
"joint_pos_hip_r_1",
"joint_pos_hip_r_2",
"joint_pos_knee_l_0",
"joint_pos_knee_r_0",
"joint_vel_ankle_l_0",
"joint_vel_ankle_r_0",
"joint_vel_back_0",
"joint_vel_ground_pelvis_0",
"joint_vel_ground_pelvis_1",
"joint_vel_ground_pelvis_2",
"joint_vel_ground_pelvis_3",
"joint_vel_ground_pelvis_4",
"joint_vel_ground_pelvis_5",
"joint_vel_hip_l_0",
"joint_vel_hip_l_1",
"joint_vel_hip_l_2",
"joint_vel_hip_r_0",
"joint_vel_hip_r_1",
"joint_vel_hip_r_2",
"joint_vel_knee_l_0",
"joint_vel_knee_r_0",
"misc_mass_center_acc_0",
"misc_mass_center_acc_1",
"misc_mass_center_pos_0",
"misc_mass_center_pos_1",
"misc_mass_center_vel_0",
"misc_mass_center_vel_1",
"muscles_abd_l_activation",
"muscles_abd_l_fiber_force",
"muscles_abd_l_fiber_length",
"muscles_abd_l_fiber_velocity",
"muscles_abd_r_activation",
"muscles_abd_r_fiber_force",
"muscles_abd_r_fiber_length",
"muscles_abd_r_fiber_velocity",
"muscles_add_l_activation",
"muscles_add_l_fiber_force",
"muscles_add_l_fiber_length",
"muscles_add_l_fiber_velocity",
"muscles_add_r_activation",
"muscles_add_r_fiber_force",
"muscles_add_r_fiber_length",
"muscles_add_r_fiber_velocity",
"muscles_bifemsh_l_activation",
"muscles_bifemsh_l_fiber_force",
"muscles_bifemsh_l_fiber_length",
"muscles_bifemsh_l_fiber_velocity",
"muscles_bifemsh_r_activation",
"muscles_bifemsh_r_fiber_force",
"muscles_bifemsh_r_fiber_length",
"muscles_bifemsh_r_fiber_velocity",
"muscles_gastroc_l_activation",
"muscles_gastroc_l_fiber_force",
"muscles_gastroc_l_fiber_length",
"muscles_gastroc_l_fiber_velocity",
"muscles_glut_max_l_activation",
"muscles_glut_max_l_fiber_force",
"muscles_glut_max_l_fiber_length",
"muscles_glut_max_l_fiber_velocity",
"muscles_glut_max_r_activation",
"muscles_glut_max_r_fiber_force",
"muscles_glut_max_r_fiber_length",
"muscles_glut_max_r_fiber_velocity",
"muscles_hamstrings_l_activation",
"muscles_hamstrings_l_fiber_force",
"muscles_hamstrings_l_fiber_length",
"muscles_hamstrings_l_fiber_velocity",
"muscles_hamstrings_r_activation",
"muscles_hamstrings_r_fiber_force",
"muscles_hamstrings_r_fiber_length",
"muscles_hamstrings_r_fiber_velocity",
"muscles_iliopsoas_l_activation",
"muscles_iliopsoas_l_fiber_force",
"muscles_iliopsoas_l_fiber_length",
"muscles_iliopsoas_l_fiber_velocity",
"muscles_iliopsoas_r_activation",
"muscles_iliopsoas_r_fiber_force",
"muscles_iliopsoas_r_fiber_length",
"muscles_iliopsoas_r_fiber_velocity",
"muscles_rect_fem_l_activation",
"muscles_rect_fem_l_fiber_force",
"muscles_rect_fem_l_fiber_length",
"muscles_rect_fem_l_fiber_velocity",
"muscles_rect_fem_r_activation",
"muscles_rect_fem_r_fiber_force",
"muscles_rect_fem_r_fiber_length",
"muscles_rect_fem_r_fiber_velocity",
"muscles_soleus_l_activation",
"muscles_soleus_l_fiber_force",
"muscles_soleus_l_fiber_length",
"muscles_soleus_l_fiber_velocity",
"muscles_tib_ant_l_activation",
"muscles_tib_ant_l_fiber_force",
"muscles_tib_ant_l_fiber_length",
"muscles_tib_ant_l_fiber_velocity",
"muscles_vasti_l_activation",
"muscles_vasti_l_fiber_force",
"muscles_vasti_l_fiber_length",
"muscles_vasti_l_fiber_velocity",
"muscles_vasti_r_activation",
"muscles_vasti_r_fiber_force",
"muscles_vasti_r_fiber_length",
"muscles_vasti_r_fiber_velocity"]
        self.shape = len(self.dict_keys) # self._obs_space.shape

    def transform(self, observation):
        obs = self.relative_features(observation)
        obs = self.flatten_dict({},obs,[])
        obs_list = []
        for i in self.dict_keys:
            obs_list.append(obs[i])
        return obs_list

    #from observation to relative observation
    def relative_features(self, state_desc):

        # way osim.py does, not sure why so many fewer positions and why they cut out so many observation points
        # for body_part in ["pelvis", "head", "torso", "toes_l", "toes_r", "talus_l", "talus_r"]:
        #     if is_prosthetic and body_part in ['toes_r','talus_r']:
        #         continue

        for body_part in ['pelvis', 'calcn_l', 'femur_l', 'femur_r', 'head', 'pros_foot_r', 'pros_tibia_r', 'talus_l',
                          'tibia_l', 'toes_l', 'torso']:

            # pelvis used for relative positions and position rotations
            if body_part == "pelvis":
                pelvis_list = []
                pelvis_list += state_desc["body_pos"][body_part]  # [0:2]
                pelvis_list += state_desc["body_pos_rot"][body_part]  # [2:]

            else:
                # other body parts have to following updated
                # positions x,y,z
                # position rotation x, y, z

                for i in range(3):
                    state_desc["body_pos"][body_part][i] -= pelvis_list[i]
                    state_desc["body_pos_rot"][body_part][i] -= pelvis_list[i + 3]

        for i in range(2):
            state_desc["misc"]["mass_center_pos"][i] -= pelvis_list[
                i]  # why only 2 values in this? raised in issues but not clarified

        return state_desc

    #flattens dictionary
    def flatten_dict(self,ret_dict,d,key_list=[]):
        for k, v in d.items():
            key_list.append(k)
            if isinstance(v, dict):
                flatten_dict(ret_dict, v, key_list)
            else:
                zString = "_".join(key_list)

                if isinstance(v, list):
                    v_count = len(d[k])
                    for j in range(v_count):
                        tempString = zString + "_" + str(j)
                        ret_dict[tempString] = v[j]
                else:
                    ret_dict[zString] = v

            key_list.pop()
        return ret_dict

ModelCatalog.register_custom_preprocessor("osim_prep", OsimPreprocessor)

# class OsimEnv(gym.Env):
#     def __init__(self,env_config):
#         self.accuracy_setting = 1e-1
#         self.nstep_hold = 4
#         self.env = ProstheticsEnv(visualize=False,integrator_accuracy=self.accuracy_setting)
#
#     # def step(self,action):
#     #     for j in range(self.nstep_hold):
#     #         obs, reward, done, info = env.step(action, project=False)
#     #         if done:
#     #             break

#register_env(env_name, lambda c: OsimEnv(c))
self.accuracy_setting = 1e-1
register_env(env_name, lambda config: ProstheticsEnv(visualize=False,integrator_accuracy=self.accuracy_setting))


ray.init()

max_timesteps = 4000000
experiment_name = "osim_{}".format(int(time.time()))

run_experiments({
    experiment_name: {
        'run': 'APEX_DDPG',
        'env':env_name,
        'stop':{'timesteps_total': max_timesteps},
        'repeat':1,
        'checkpoint_freq': 500,
        "trial_resources": {
                            'cpu': 1,#lambda spec: spec.config.num_workers,#lambda spec: spec.config.num_workers,
                            'extra_cpu': 7,
                            "gpu": 1
        },
        'config': {
            'num_workers': 4,
            'observation_filter': 'MeanStdFilter',
            #defaults commented out
            # "n_step": 3,
            # "gpu": False,
            # "buffer_size": 2000000,
            "buffer_size": 100000,
            # "learning_starts": 50000,
            "learning_starts": 10000,
            # "train_batch_size": 512,
            # "sample_batch_size": 50,
            # "max_weight_sync_delay": 400,
            # "target_network_update_freq": 500000,
            "target_network_update_freq": 10000,
            # "timesteps_per_iteration": 25000,
            "timesteps_per_iteration": 2500,
            # "per_worker_exploration": True,
            # "worker_side_prioritization": True,
            # "min_iter_time_s": 30,
            'optimizer':{
                'num_replay_buffer_shards':1,
            },
            'model':{
                "custom_preprocessor": "osim_prep",
            },
            'tf_session_args': {
                'gpu_options': {'allow_growth': True}
            },
        },
    },
})
