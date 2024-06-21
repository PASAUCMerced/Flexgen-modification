### Structure

```plaintext
|   
| - decentralized_model -    (todo)
|   |- hivemind/
│       |─ __init__.py
│       |─ connection_manager.py
│       |─ peer_discovery.py        # e.g., build connections via IP addresses
│       |─ utils.py 
|   |- cli/
│       |─ run_dht.py
|
|   |- server/
│       ├── __init__.py
│       ├── server.py
│       ├── backend.py
│       ├── handlers.py

|   |- client/
│       ├── __init__.py
│       ├── client.py   # e.g., clientConfig
│       ├── client_manager.py # e.g., sequence manager
│       ├── sequential_generation.py 

|   |- pipeline parallelism
|       |- models/
|              |- llama/layers (todo)
|              |- opt/layers (todo)
|
| - dist_model -    (todo)
|   |- tensor model parallelism -
|   |   |- models/
|   |           |- llama/layers (todo)
|   |           |- opt/layers (todo)
|   |
|   |- sequence model parallelism -
|       |- models/
|               |- llama/layers (todo)
|           |- opt/layers (todo)
|
| - single_gpu_model -
|   |- models/
|       |- llama/layers (pass)
|       |- opt/layers (pass)
|
| - examples
|   |- decentralized_model_scripts (configs)
|   |   |- models
|   |       |- llama
|   |       |   |- dece_flex_llama.py (todo)
|   |       |- opt
|   |       |   |- dece_flex_opt.py (todo)
|   |
|   |- dist_model_scripts
|   |    |- models
|   |        |- llama
|   |        |   |- dist_flex_llama.py (todo)
|   |        |- opt
|   |            |- dist_flex_opt.py (todo)
|   |
|   |- single_gpu_model_scripts
|      |- models
|           |- llama
|           |   |- flex_llama.py (pass)
|           |- opt
|               |- flex_opt.py (pass)
|
| - utils
|
```

