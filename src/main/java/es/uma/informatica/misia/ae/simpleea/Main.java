package es.uma.informatica.misia.ae.simpleea;

import java.util.HashMap;
import java.util.Map;

public class Main {
    public static void main(String[] args) {
        if (args.length < 5) {
            System.err.println("Invalid number of arguments");
            System.err.println("Arguments: <population size> <function evaluations> <bitflip probability> <cross probability> <problem size> [<random seed>]");
            System.err.println("Note: If <functionEvaluations> is 0, the algorithm will run until the global optimum is found.");
            return;
        }

        int n = Integer.parseInt(args[4]);
        Problem problem = new Onemax(n);

        Map<String, Double> parameters = readEAParameters(args);
        EvolutionaryAlgorithm evolutionaryAlgorithm = new EvolutionaryAlgorithm(parameters, problem);

        Individual bestSolution = evolutionaryAlgorithm.run();

        Double maxFunctionEvaluations = parameters.get(EvolutionaryAlgorithm.MAX_FUNCTION_EVALUATIONS_PARAM);

        // If we are looking for the global optimum, we want to know the number of evaluations instead of the fitness value
        Double result = maxFunctionEvaluations > 0 ? bestSolution.fitness : evolutionaryAlgorithm.getFuncEvaluations();

        // We print only the result number, so the python script can collect it from stdout
        System.out.println(result);
    }

    private static Map<String, Double> readEAParameters(String[] args) {
        Map<String, Double> parameters = new HashMap<>();
        parameters.put(EvolutionaryAlgorithm.POPULATION_SIZE_PARAM, Double.parseDouble(args[0]));
        parameters.put(EvolutionaryAlgorithm.MAX_FUNCTION_EVALUATIONS_PARAM, Double.parseDouble(args[1]));
        parameters.put(BitFlipMutation.BIT_FLIP_PROBABILITY_PARAM, Double.parseDouble(args[2]));
        parameters.put(SinglePointCrossover.CROSS_PROBABILITY_PARAM, Double.parseDouble(args[3]));

        long randomSeed = System.currentTimeMillis();
        if (args.length > 5) {
            randomSeed = Long.parseLong(args[5]);
        }
        parameters.put(EvolutionaryAlgorithm.RANDOM_SEED_PARAM, (double) randomSeed);
        return parameters;
    }
}
