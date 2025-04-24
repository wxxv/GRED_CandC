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
    
a = entropy_calculation([0.4,0.15,0.2,0.25]) - 0.35*entropy_calculation([0,0.15,0.2,0]) - 0.65*entropy_calculation([0.4,0,0,0.25])
print(a)
b = entropy_calculation([0.35,0.65])
print(b)