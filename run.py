import os

cmd_prefix = 'python main.py'

cmd_list = [
    '--vis_env var_loss_mix',
    '--vis_env var_loss_org',
    '--vis_env var_loss_std',
    '--vis_env var_loss_alpha --alpha 0.3',
    '--vis_env var_loss_alpha --alpha 0.4',
    '--vis_env var_loss_course --alpha 0.5',
    '--vis_env var_loss_course --alpha 0.4',
]

for cmd in cmd_list:
    os.system(cmd_prefix+' '+cmd)
<<<<<<< HEAD




# asfdsa



# temp
=======
>>>>>>> 412de234568b561feaa7797402026dd7a832d801
