"""
    Converting integer to binary. 
        - python: bin() 
        - Java: Integer.parseInt(binary.toString(), 2);
    
    Converting binary to integer
        - Python: just use int(binary, 2)
        - Java: Integer.toBinaryString(b)
        
    Even or Odd
        - odd when last bit is 1, event when 0
        
    Left Shift <<
        - shift bits to left, same as: num * 2^n
        - i.e. result = binary << 2 -> num * 2^2
    
    Right Shift >>
        - shift bigs to right, same as: num * 2^-n
        2
        
"""


def int_to_bin(num):
    binary = ""
    while num > 0:
        remainder = num % 2
        num = num // 2
        binary = str(remainder) + binary
    return binary or "0"


def bin_to_int(binary: str):
    num = 0
    for v in binary:
        num = num * 2 + int(v)
    return num


def decimal_to_bin(num: float, precision: int = 10) -> str:
    if num == 0:
        return "0"
    integer_part = int(num)
    fractional_part = num - integer_part
    binary_integer = ""

    if integer_part == 0:
        binary_integer = "0"
    else:
        while integer_part > 0:
            remainder = integer_part % 2
            integer_part = integer_part // 2
            binary_integer = str(remainder) + binary_integer

    binary_fractional = ""
    while fractional_part > 0 and len(binary_fractional) < precision:
        fractional_part *= 2
        bit = int(fractional_part)
        binary_fractional += str(bit)
        fractional_part -= bit

    if binary_fractional:
        return binary_integer + "." + binary_fractional
    else:
        return binary_integer


def bin_to_decimal(binary: str) -> float:
    if "." in binary:
        integer_part, fractional_part = binary.split(".")
    else:
        integer_part, fractional_part = binary, ""
    num = 0
    for v in integer_part:
        num = num * 2 + int(v)

    fractional_value = 0
    for i, v in enumerate(fractional_part):
        fractional_value += int(v) * (2 ** -(i + 1))

    return num + fractional_value


def addBinary(a, b) -> str:
    n = max(len(a), len(b))
    a, b = a.zfill(n), b.zfill(n)
    carry = 0
    answer = []

    for i in range(n - 1, -1, -1):
        if a[i] == "1":
            carry += 1
        if b[i] == "1":
            carry += 1
        if carry % 2 == 1:
            answer.append("1")
        else:
            answer.append("0")
        carry //= 2

    if carry == 1:
        answer.append("1")
    answer.reverse()
    return "".join(answer)
