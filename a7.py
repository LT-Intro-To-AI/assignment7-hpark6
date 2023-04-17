from neural import NeuralNet

sq_training_data = [
    ([0.2], [0.04]),
    ([0.3], [0.09]),
    ([0.5], [0.25]),
    ([0.7], [0.49]),
    ([0.1], [0.01]),
    ([.9], [.81])
]
sqn = NeuralNet(1, 20, 1)#(__, num of nodes, __)
sqn.train(sq_training_data)

print()
print(sqn.test_with_expected(sq_training_data))
print(sqn.evaluate([0.66]))
print(sqn.evaluate([0.95]))


or_data = [
    ([0,0],[0]),
    ([0,1],[1]),
    ([1,0],[1]),
    ([1,1],[1])]

orn = NeuralNet(2, 20, 1)
orn.train(or_data)

print()
print(orn.test_with_expected(or_data))


print()
print("\n\nTraining voter opinion\n\n")


voter_opinions = [
    ([.9, .6, .8, .3, .1], [1]),
    ([.8, .8, .4, .6, .4], [1]),
    ([.7, .2, .4, .6, .3], [1]),
    ([.5, .5, .8, .4, .8], [0]),
    ([.3, .1, .6, .8, .8], [0]),
    ([.6, .3, .4, .3, .6], [0])
]

von = NeuralNet(5, 6, 1)
von.train(voter_opinions)

print(von.test_with_expected(voter_opinions))

test_data = [
    [1, 1, 1, .1, .1],
    [.5, .2, .1, .7, .7]
]

print(f"case 1: {test_data[0]} evaluates to: {von.evaluate(test_data[0])}")