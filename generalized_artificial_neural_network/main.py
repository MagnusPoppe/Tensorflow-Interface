import os

from generalized_artificial_neural_network.trainer import Trainer

file = "bit-counter.json"  # "glass.json" #"one-hot.json"
configuration_file = os.path.join("configurations", file)

coach = Trainer(configuration_file, display_graph=False)
coach.train(epochs=coach.config.epochs)
print("\nTRAINING COMPLETE!")
print("\tERROR AFTER TRAINING: " + str(coach.error_history[-1][1]))

# Running tests:
training_score = coach.test(coach.config.manager.get_training_cases(), in_top_k=True)
validation_score = coach.test(coach.config.manager.get_validation_cases(), in_top_k=True)
testing_score = coach.test(coach.config.manager.get_testing_cases(), in_top_k=True)

print("\nPERFORMING TESTS:")
print("\tSCORE ON TRAINING CASES:   " + str(training_score)   + "%")
print("\tSCORE ON VALIDATION CASES: " + str(validation_score) + "%")
print("\tSCORE ON TESTING CASES:    " + str(testing_score)    + "%")

run = True
waited = 0
while run:
    x = input(">>> ")

    if x in ["q", "quit"]:
        run = False

    elif x == "run more":
        pass

    elif x in "run tests":
        coach.test(coach.config.manager.get_testing_cases())