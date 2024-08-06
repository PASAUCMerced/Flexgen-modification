the tensor parallellism realted functions:  
def mha_wo_layernorm_TP()  
split the layer norm from the original "mha" function, then we need add tensor parallelism related modifications.  
def mha_gen_wo_layernorm_TP()    
split the layer norm from the original "mha_gen" function, then we need add tensor parallelism related modifications.  
