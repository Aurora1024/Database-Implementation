import numpy as np
import pandas as pd

def test(dict,key,value):
    dict[key]=value
    return 1

if __name__ == '__main__':
    dict = {'1':'aaa','2':'bbb','3':'ccc'}
    test(dict,'1','ddd')
    print(dict)