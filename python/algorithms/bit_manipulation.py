"""
Converting integer to binary. 

1. The bin() function converts an integer to a binary string.

2. Start with the integer in question and divide it by 2 keeping notice of the quotient and the remainder. 
  Continue dividing the quotient by 2 until you get a quotient of zero. Then just write out the remainders in the reverse order.

  12 / 2 = 6 remainder 0
  6 / 2 = 3 remainder 0
  3 / 2 = 1 remainder 1
  1 / 2 = 0 remainder 1
  binary = 1100
"""


def int_to_bin(num):
    binary = ""
    while num > 0:
        remainder = num % 2
        num = num // 2
        binary = str(remainder) + binary
    return binary
