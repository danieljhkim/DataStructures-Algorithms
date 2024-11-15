def count_sysmmetric_nums(LOW, HIGH):
    count = 0
    for size in range(1, len(str(HIGH)) + 1):
        for digit in range(1, 10):
            sym = int(str(digit) * size)
            if LOW <= sym <= HIGH:
                count += 1
    return count


def count_symmetric_nums2(LOW, HIGH):
    def generate_same_digit_numbers(digit, length):
        return int(str(digit) * length)

    count = 0
    for digit in range(1, 10):  # Digits from 1 to 9
        length = 1
        while True:
            num = generate_same_digit_numbers(digit, length)
            if num > HIGH:
                break
            if num >= LOW:
                count += 1
            length += 1

    return count


def count_symmetric_nums(LOW, HIGH):
    def how_many(num):
        if num <= 10:
            return num - 1
        return 9

    n = HIGH
    count = 0
    while n > 0:
        count += how_many(n + 1)
        n //= 10
    n = LOW
    while n > 0:
        count -= how_many(n)
        n //= 10
    if LOW:
        return count
    return count


print(count_symmetric_nums(123, 9999999))

print(count_sysmmetric_nums(123, 9999999))

print(count_symmetric_nums2(123, 9999999))
