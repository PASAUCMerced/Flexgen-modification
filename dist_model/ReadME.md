In "pytorch_backend.py"
the tensor parallellism realted functions:  
`def mha_wo_layernorm_TP() `   
       split the layer norm from the original "mha" function, then we need add tensor parallelism related modifications.  
       the weight splitting has not add to this function yet.  
`def mha_gen_wo_layernorm_TP()`    
	split the layer norm from the original "mha_gen" function, then we need add tensor parallelism related modifications.   
 	the weight splitting has not add to this function yet.  

    
  the function `mha_gen`, with previous weight split is in `torch_device.py`  
