import re

def re_match(pattern, text):
    res_dic = {}
    pat = re.compile(pattern, re.I | re.M)
    res = pat.findall(text)
    for r in res:
        v = r.split("{")[-1].split("}")[0]
        res_dic[r] = v
    return res_dic

def re_ip(text):
    res = re_match(r'~ip{[\s\S]*?}', text)
    
    for k, v in res.items():
        _v = ""
        v = v.split(".")
        for idx,i in enumerate(v):
            for j in i:
                _v += f"{j} "
            if idx != 3:
                _v += f" dot "
        text = text.replace(k, _v)
        
    return text

def re_web(text):
    res = re_match(r'~web{[\s\S]*?}', text)
    
    for k, v in res.items():
        _v = ""
        v = v.split(".")
        for idx,i in enumerate(v):
            for j in i:
                _v += f"{j} "
            if idx != len(v)-1:
                _v += f" dot "
        text = text.replace(k, _v)
        
    return text


def re_abbr(text):
    res = re_match(r'~abbr{[\s\S]*?}', text)
    
    for k, v in res.items():
        _v = ""
        for i in v:
            _v += f"{i} "
        text = text.replace(k, _v)
        
    return text
                      