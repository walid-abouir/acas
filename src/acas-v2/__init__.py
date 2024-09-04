from acas_xu import *
from gymnasium.envs.registration import register

register(
    id='AcasEnv',
    entry_point='acas_v2.acas_xu:AcasEnv',
    
)

register(
    id='Acas-continuous-v0',
    entry_point='acas-v2.acas_xu:AcasEnvContinuous',
    
)
