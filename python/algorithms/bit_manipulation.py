"""
    Converting integer to binary. 
        - python: bin() 
        - Java: Integer.parseInt(binary.toString(), 2);
    
    Converting binary to integer
        - Python: just use int(binary, 2)
        - Java: Integer.toBinaryString(b)
"""


def int_to_bin(num):
    binary = ""
    while num > 0:
        remainder = num % 2
        num = num // 2
        binary = str(remainder) + binary
    return binary


def bin_to_int(binary: str):
    num = 0
    for i, v in enumerate(binary):
        num = num * 2 + int(v)
    return num
