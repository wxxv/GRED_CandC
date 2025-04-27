import math


def entropy_calculation(probabilities):
    probabilities = [probability / sum(probabilities) for probability in probabilities]
    # print(probabilities)
    entropy = 0
    for probability in probabilities:
        if probability > 0:
            entropy -= probability * math.log10(probability)
    return entropy


# 示例概率列表
# probabilities = [0,0.25,0.4,0] 
# entropy = entropy_calculation(probabilities)
# print(f"给定概率的熵为: {entropy}")
    

# a = entropy_calculation([0.1,0.1,0.4,0.4]) - 0.1*entropy_calculation([1,0,0,0]) - 0.1*entropy_calculation([0,1,0,0]) - 0.8*entropy_calculation([0,0,0.8,0.8])    # RQ1
# a = entropy_calculation([0.1,0.1,0.4,0.4]) - 0.5*entropy_calculation([0.1, 0.1, 0.8, 0]) - 0.5*entropy_calculation([0.1, 0.1, 0.8, 0])     # RQ2

a = entropy_calculation([0.4,0.15,0.2,0.25]) - 0.15/0.35*entropy_calculation([0.4,0.35,0,0.25]) - 0.2/0.35*entropy_calculation([0.4,0.35,0,0.25])
print(a)


# b = entropy_calculation([0.1, 0.1 ,0.8])
# b = 0.8 * entropy_calculation([0.5, 0.5])

b = 0.35*entropy_calculation([0.2, 0.15])
print(b)