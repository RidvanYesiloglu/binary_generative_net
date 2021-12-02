import sys
import parameters
import os

params_dict = parameters.decode_arguments_dictionary("/home/groups/gracegao/Low_Corr_Bin_Code_Design/params_dictionary")
args = parameters.get_arguments(params_dict)
repr_str = parameters.create_repr_str(args, [info.name for info in params_dict.param_infos])
waitings_folder = os.path.join("/scratch/groups/gracegao/Low_Corr_Bin_Code_Design/waiting_jobs", repr_str)
os.rmdir(waitings_folder)