
message_file = open("test.txt", 'r')

def _create_staircase(nums):
  """
  1
  2 3
  4 5 7
  """
  nums.sort()
  subsets = []
  step = 1
  while len(nums) != 0:
    if len(nums) >= step:
      subsets.append(nums[0:step])
      nums = nums[step:]
      step += 1
    else:
      subsets.append(nums[0:len(nums)])
      break
  return subsets

def decode(message_file):
  if isinstance(message_file, str):
    encoded_file = open(message_file, 'r')
  else:
    encoded_file = message_file
  dictionary = {}
  numsList = []
  for line in encoded_file:
    segments = line.split(" ")
    if len(segments) == 2:
      dictionary[int(segments[0])] = segments[1].strip('\n')
      numsList.append(int(segments[0]))
  steps = _create_staircase(numsList)
  ans = ""
  for step in steps:
    last_num = step[-1]
    decoded_word = dictionary[last_num]
    ans += decoded_word + " "
  print(ans)
  return ans.strip(' ')

    
decode(message_file)