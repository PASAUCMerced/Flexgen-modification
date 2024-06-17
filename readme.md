### Structure

```plaintext
|   
| - decentralized_model -     
|   
|   |- hivemind related, e.g., build connections via IP
|   |- server.py
|   |- jobs, clients
|   |- (pipeline parallelism) models/layers (opt, llama)
|
| - dist_model
|   |- tensor model parallelism -
|   |   |- models/layers (opt, llama)
|   |- sequence model parallelism -
|       |- models/layers (opt, llama)
|
| - single_gpu_model -
|   |- models/
|       |- llama/layers (pass)
|       |- opt/layers
|
| - examples
|   |- decentralized_model_scripts (configs)
|   |- dist_model_scripts
|   |- single_gpu_model_scripts
|       |- models
|           |- llama
|           |   |- flex_llama.py (pass)
|           |- opt
|               |- flex_opt.py (todo)
|
| - utils
|
```

