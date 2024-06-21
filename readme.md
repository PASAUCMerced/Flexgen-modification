### Structure

```plaintext
|   
| - decentralized_model -    
|   | hivemind/
│       |─ __init__.py
│       |─ connection_manager.py
│       |─ peer_discovery.py        # e.g., build connections via IP addresses
│       |─ utils.py 
|   
|   |- server/
│       ├── __init__.py
│       ├── server.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── endpoints.py
│       │   └── middleware.py
│       └── jobs/
│          ├── __init__.py
│          ├── job_manager.py
│          └── client_handler.py
|   
|   |- pipeline parallelism
|       |- models/
|              |- llama/layers (todo)
|              |- opt/layers (todo)
|
| - dist_model
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

