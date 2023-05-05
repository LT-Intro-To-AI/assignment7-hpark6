from neural import NeuralNet

forest_fire_training_data = [
    ([0.2], [0.04]),
    ([0.3], [0.09]),
    ([0.5], [0.25]),
    ([0.7], [0.49]),
    ([0.1], [0.01]),
    ([.9], [.81])
]
ffn = NeuralNet(1, 1, 1)
ffn.train(forest_fire_training_data)

print()
print(ffn.test_with_expected(forest_fire_training_data))
print(ffn.evaluate([0.66]))
print(ffn.evaluate([0.95]))

n_data = [
    ([0,0],[0]),
    ([0,1],[1]),
    ([1,0],[1]),
    ([1,1],[1])]

n_test = NeuralNet(2, 9, 1)
n_test.train(or_data)

print()
print(n_test.test_with_expected(n_data))




print()
print("\n\nTraining forest fire data\n\n")

fires = [
    ([7, 5, mar, fri, 86.2, 26.2, 94.3, 5.1, 8.2, 51, 6.7, 0], [0]), #line 2
    ([3, 4, sep, tue, 91, 129.5, 692.6, 7, 13.9, 59, 6.3, 0], [11.24]), #line 200
    ([8, 8, aug, wed, 91.7, 191.4, 635.9, 7.8, 26.2, 36, 4.5, 0], [185.76]) #line 422
    ([8, 6, aug, thu, 94.8, 222.4, 698.6, 13.9, 27.5, 27, 4.9, 0], [746.28]),
    ([.3, .1, .6, .8, .8], [0]),
    ([.6, .3, .4, .3, .6], [0])
]

ff = NeuralNet(5, 6, 1)
ff.train(voter_op)

print(ff.test_with_expected(fires))

test_data = [
    [4, 7, jul, thu, 89.5, 203.4, 663.1, 8, 25, 43, 3.99, 0], #12 inputs
    [8, 4, oct, fri, 90.4, 245.6, 559.5, 4, 30, 50, 2.4, 0],
   
]
print()
print(f"case 1: {test_data[0]} evaluates to: {ff.evaluate(test_data[0])}")
print(f"case 2: {test_data[1]} evaluates to: {ff.evaluate(test_data[1])}")
#print(f"case 3: {test_data[2]} evaluates to: {ff.evaluate(test_data[2])}")
#print(f"case 4: {test_data[3]} evaluates to: {ff.evaluate(test_data[3])}")
#print(f"case 5: {test_data[4]} evaluates to: {ff.evaluate(test_data[4])}")
