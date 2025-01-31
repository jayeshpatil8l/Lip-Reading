
import os
from utils import get_corrupted

corrupted = get_corrupted()

if corrupted:
    for file in corrupted:
        if(file.endswith('.mpg')):
            os.remove(os.path.join('..','Data', 's1',f'{file}'))
            print(f"Removed File: {file}")      
            os.remove(os.path.join('..','Data', 'align',f"{file.split('.')[0]}.align"))
            print(f"Removed File: {file.split('.')[0]}.align")
        else:
            os.remove(os.path.join('..','Data', 's1',f'{file}'))
            print(f"Removed File: {file}")
else:
    print('No Currupted files to remove!')