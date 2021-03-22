from numba.pycc import CC

cc = CC('module_truc')

@cc.export('plus_1', 'float64(float64)')
def plus_1(n):
    n += 1
    return n


 
    
    
    
    
    
    
    
    