from gymnasium.envs.registration import register

register(
    id='Acas-discrete-v1',
    entry_point='acas-v2.acas_xu:AcasEnv',
    
)

register(
    id='Acas-continuous-v0',
    entry_point='acas-v2.acas_xu:AcasEnvContinuous',
    
)
