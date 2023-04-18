from neural import NeuralNet

sq_training_data = [
    ([0.2], [0.04]),
    ([0.3], [0.09]),
    ([0.5], [0.25]),
    ([0.7], [0.49]),
    ([0.1], [0.01]),
    ([.9], [.81])
]
sqn = NeuralNet(1, 1, 1)
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

or_test = NeuralNet(2, 9, 1)
or_test.train(or_data)

print()
print(or_test.test_with_expected(or_data))




print()
print("\n\nTraining voter opinion\n\n")

voter_op = [
    ([.9, .6, .8, .3, .1], [1]),
    ([.8, .8, .4, .6, .4], [1]),
    ([.7, .2, .4, .6, .3], [1]),
    ([.5, .5, .8, .4, .8], [0]),
    ([.3, .1, .6, .8, .8], [0]),
    ([.6, .3, .4, .3, .6], [0])
]

von = NeuralNet(5, 6, 1)
von.train(voter_op)

print(von.test_with_expected(voter_op))

test_data = [
    [1, 1, 1, .1, .1],
    [.5, .2, .1, .7, .7],
    [.8, .3, .3, .3, .8],
    [.8, .3, .3, .8, .3],
    [.9, .8, .8, .3, .6]
]
print()
print(f"case 1: {test_data[0]} evaluates to: {von.evaluate(test_data[0])}")
print(f"case 2: {test_data[1]} evaluates to: {von.evaluate(test_data[1])}")
print(f"case 3: {test_data[2]} evaluates to: {von.evaluate(test_data[2])}")
print(f"case 4: {test_data[3]} evaluates to: {von.evaluate(test_data[3])}")
print(f"case 5: {test_data[4]} evaluates to: {von.evaluate(test_data[4])}")
