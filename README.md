# nmt

# Execution:

To get set up, just run ``` sh run_script.sh```

# TO-DO
Sampling:
    
    - Fix Vocabulary - For now, only bilingual vocabulary is present. Need to add Monolingual vocab also.

    - Fix uniform probability sampling - len(mono) >> len(bi). Hence, majority samples are mono for now
        - Use WeightedRandomSampler (torch.utils.sampler)
Loss:
    
    - Add Mono Loss 
