from src.features.features_selection import elitism
from sklearn.ensemble import RandomForestClassifier
from src.features.features_selection.LoanDefaultRisk import LoanDefaultRisk
from sklearn import model_selection
from src.config import Consts as Cs
from deap import base
from deap import creator
from deap import tools
import random
import numpy
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from boruta import BorutaPy
import numpy as np
import json



class LoanFeatureSelector:
    """
    Select most import features using genetic and boruta algorithm
    """
    def __init__(self, method, parameters=None):

        if parameters is None:
            parameters = Cs.FS_PARAMETERS
        self.parameters = parameters
        self.method = method  # takes either ga or boruta
        self.loan = LoanDefaultRisk(self.method, self.parameters)  # load data, cv, model parameters
        self.best_features = None
        self.second_best_features = None

    def fitnessFunction(self, individual):
        """
        fitness function used by genetic algorithm to choose the best chromosomes (solution)

        :param individual: [0,1,1,0, ...,1] chromosome, potential solution, selected features

        :return  value : number representing fitness of the solution based on auc and regularisation equation
        """
        num_features_used = sum(individual)
        if num_features_used == 0:
            return 0.0,
        else:
            auc = self.loan.getMeanAuc(individual)
            return auc - self.parameters['OPTIMISATION']['FEATURE_PENALTY_FACTOR'] * num_features_used,

    def GeneticAlgorithm(self):
        """
        fix multiple methods and pipeline that will be used in the GA selection process of features

        :return  toolbox : GA architecture
        """
        toolbox = base.Toolbox()
        # define a single objective, maximizing fitness strategy:
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        # create the Individual class based on list:
        creator.create("Individual", list, fitness=creator.FitnessMax)
        # create an operator that randomly returns 0 or 1:
        toolbox.register("zeroOrOne", random.randint, 0, 1)
        # create the individual operator to fill up an Individual instance:
        toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.zeroOrOne, len(self.loan))
        # create the population operator to generate a list of individuals:
        toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)
        toolbox.register("evaluate", self.fitnessFunction)
        # Tournament selection : method of selecting potential solutions for the next generation
        toolbox.register("select", tools.selTournament, tournsize=self.parameters['OPTIMISATION']['TOURNAMENT_SIZE'])
        # Single-point crossover: method of combining two solutions into hopefully more better one
        toolbox.register("mate", tools.cxTwoPoint)
        # Flip-bit : method of tweaking a solution by mutation, help explore new areas in the optimisation search space
        # indpb: Independent probability for each attribute to be flipped
        toolbox.register("mutate", tools.mutFlipBit,
                         indpb=self.parameters['OPTIMISATION']['MUTATION_FLIP'] / len(self.loan))

        return toolbox

    def find_optimal_features(self, visualize=False):
        """
        launch the feature selector algorithm and saving the best features

        :param visualize: if True visualize the evolution of GA population and best solution across generations
        """

        if self.method == 'boruta':

            fs_boruta = BorutaPy(estimator=self.loan.classifier, n_estimators='auto', max_iter=50)
            fs_boruta.fit(np.array(self.loan.X), np.array(self.loan.y))

            self.best_features = self.loan.X.columns[fs_boruta.support_].to_list()  # features confirmed important
            self.second_best_features = self.loan.X.columns[fs_boruta.support_weak_].to_list()

        elif self.method == 'ga':

            toolbox = self.GeneticAlgorithm()

            # create initial population (generation 0):
            population = toolbox.populationCreator(n=self.parameters['OPTIMISATION']['POPULATION_SIZE'])

            # prepare the statistics object:
            stats = tools.Statistics(lambda ind: ind.fitness.values)
            stats.register("max", numpy.max)
            stats.register("avg", numpy.mean)

            # define the hall-of-fame object:
            hof = tools.HallOfFame(self.parameters['OPTIMISATION']['HALL_OF_FAME_SIZE'])

            # perform the Genetic Algorithm flow with hof feature added:
            population, logbook = elitism.eaSimpleWithElitism(population, toolbox,
                                                              cxpb=self.parameters['OPTIMISATION']['P_CROSSOVER'],
                                                              mutpb=self.parameters['OPTIMISATION']['P_MUTATION'],
                                                              ngen=self.parameters['OPTIMISATION']['MAX_GENERATIONS'],
                                                              stats=stats, halloffame=hof, verbose=True)

            self.best_features = hof.items[0]
            self.best_features = [self.loan.features[i] for i, value in enumerate(self.best_features) if value == 1]

            # extract statistics:
            maxFitnessValues, meanFitnessValues = logbook.select("max", "avg")

            if visualize is True:

                # print best solution found:
                print("- Best solutions are:")
                for i in range(self.parameters['OPTIMISATION']['HALL_OF_FAME_SIZE']):
                    print(i, ": ", hof.items[i], ", fitness = ", hof.items[i].fitness.values[0],
                          ", accuracy = ", self.loan.getMeanAuc(hof.items[i]), ", features = ", sum(hof.items[i]))

                # plot statistics:
                sns.set_style("whitegrid")
                plt.plot(maxFitnessValues, color='red')
                plt.plot(meanFitnessValues, color='green')
                plt.xlabel('Generation')
                plt.ylabel('Max / Average Fitness')
                plt.title('Max and Average fitness over Generations')
                plt.show()

    def evaluate_best_features(self):
        """
        evaluate best features by using cross validation and estimating auc on unseen data
        """
        X = self.loan.X[self.best_features]
        y = self.loan.y

        model_rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, max_depth=10)
        model_ev = model_selection.cross_val_score(model_rf, X, y, cv=5, scoring='roc_auc')

        print("Estimated Model Auc performance on unseen data %a" % (model_ev.mean().round(3)))

    def save_best_features(self):
        """
        save best features
        """
        with open(Cs.PATH_SAVE_BEST_FEATURES, 'w') as file:
            json.dump(self.best_features, file)