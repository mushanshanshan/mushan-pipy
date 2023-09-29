import os

def log_line(file_path, line):
    _line = ""
    if isinstance(line, list):
        for i in line:
            _line += str(i)
            _line += " | "
        _line = _line[:-2]
    elif isinstance(line, str):
        _line = line
    else:
        print("Unknow input types!")
        return
    
    with open(file_path, 'a+') as f:
        f.write(_line)
        f.write("\n")
        
        
def set_debug_mode():
    os.environ['MUSHAN_DEBUG'] = 1
    
def debug(message):
    if os.environ['MUSHAN_DEBUG'] == 1:
        print(message)