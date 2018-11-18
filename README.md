# nmt

# Execution:

To get set up, just run ``` sh run_scripts.sh```

# TO-DO

1) Fix Vocabulary - For now, only bilingual vocabulary is present. Need to add Monolingual vocab also.

2) Fix uniform probability sampling - len(mono) >> len(bi). Hence, majority samples are mono for now
    - Use WeightedRandomSampler (torch.utils.sampler)
