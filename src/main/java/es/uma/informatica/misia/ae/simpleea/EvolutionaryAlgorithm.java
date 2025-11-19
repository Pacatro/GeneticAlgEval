package es.uma.informatica.misia.ae.simpleea;

import java.util.*;
import java.util.function.BooleanSupplier;

public class EvolutionaryAlgorithm {
    public static final String MAX_FUNCTION_EVALUATIONS_PARAM = "maxFunctionEvaluations";
    public static final String RANDOM_SEED_PARAM = "randomSeed";
    public static final String POPULATION_SIZE_PARAM = "populationSize";

    private Problem problem;
    private int functionEvaluations;
    private int maxFunctionEvaluations;
    private List<Individual> population;
    private int populationSize;
    private Random rnd;

    private Individual bestSolution;

    private Selection selection;
    private Replacement replacement;
    private Mutation mutation;
    private Crossover recombination;

    public EvolutionaryAlgorithm(Map<String, Double> parameters, Problem problem) {
        configureAlgorithm(parameters, problem);
    }

    private void configureAlgorithm(Map<String, Double> parameters, Problem problem) {
        populationSize = parameters.get(POPULATION_SIZE_PARAM).intValue();
        maxFunctionEvaluations = parameters.get(MAX_FUNCTION_EVALUATIONS_PARAM).intValue();
        double bitFlipProb = parameters.get(BitFlipMutation.BIT_FLIP_PROBABILITY_PARAM);
        double crossProb = parameters.get(SinglePointCrossover.CROSS_PROBABILITY_PARAM);
        long randomSeed = parameters.get(RANDOM_SEED_PARAM).longValue();

        this.problem = problem;

        rnd = new Random(randomSeed);

        // Selection
        selection = new BinaryTournament(rnd);
        // Recombination
        recombination = new SinglePointCrossover(rnd, crossProb);
        // Variation
        mutation = new BitFlipMutation(rnd, bitFlipProb);
        // Replacement
        replacement = new ElitistReplacement();
    }

    public int getFuncEvaluations() {
        return this.functionEvaluations;
    }

    public Individual run() {
        population = generateInitialPopulation();
        functionEvaluations = 0;

        evaluatePopulation(population);

        BooleanSupplier continueCondition = (maxFunctionEvaluations > 0)
                ? () -> functionEvaluations < maxFunctionEvaluations
                : () -> !hasGlobalOptimum(problem.getProblemSize());

        while (continueCondition.getAsBoolean()) {
            Individual parent1 = selection.selectParent(population);
            Individual parent2 = selection.selectParent(population);
            Individual child = recombination.apply(parent1, parent2);
            child = mutation.apply(child);
            evaluateIndividual(child);
            population = replacement.replacement(population, Collections.singletonList(child));
        }

        return bestSolution;
    }

    private boolean hasGlobalOptimum(int problemSize) {
        for (Individual ind : population) {
            if (ind.getFitness() == (double) problemSize) {
                return true;
            }
        }
        return false;
    }

    private void evaluateIndividual(Individual individual) {
        double fitness = problem.evaluate(individual);
        individual.setFitness(fitness);
        functionEvaluations++;
        checkIfBest(individual);
    }

    private void checkIfBest(Individual individual) {
        if (bestSolution == null || individual.getFitness() > bestSolution.getFitness()) {
            bestSolution = individual;
        }
    }

    private void evaluatePopulation(List<Individual> population) {
        for (Individual individual : population) {
            evaluateIndividual(individual);
        }
    }

    private List<Individual> generateInitialPopulation() {
        List<Individual> population = new ArrayList<>();
        for (int i = 0; i < populationSize; i++) {
            population.add(problem.generateRandomIndividual(rnd));
        }
        return population;
    }
}
