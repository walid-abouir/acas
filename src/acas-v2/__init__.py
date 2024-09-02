from gymnasium.envs.registration import register

register(
    id='acas-v2',
    entry_point='acas-v2.acas_xu:AcasEnv',
    
)
