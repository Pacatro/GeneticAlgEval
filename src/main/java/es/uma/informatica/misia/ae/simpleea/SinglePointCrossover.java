package es.uma.informatica.misia.ae.simpleea;

import java.util.Random;

public class SinglePointCrossover implements Crossover {
    private final Double crossProb;
    private final Random rnd;
    public static final String CROSS_PROBABILITY_PARAM = "crossProbability";

    public SinglePointCrossover(Random rnd, Double crossProb) {
        this.rnd = rnd;
        this.crossProb = crossProb;
    }

    @Override
    public BinaryString apply(Individual individual1, Individual individual2) {
        BinaryString binaryParent1 = (BinaryString) individual1;

        if (rnd.nextDouble() > crossProb) {
            // No cross: we return one of the parents
            return binaryParent1;
        }

        BinaryString binaryParent2 = (BinaryString) individual2;

        BinaryString child = new BinaryString(binaryParent1);
        int cutPoint = rnd.nextInt(binaryParent1.getChromosome().length + 1);

        for (int i = cutPoint; i < binaryParent1.getChromosome().length; i++) {
            child.getChromosome()[i] = binaryParent2.getChromosome()[i];
        }
        return child;
    }

}
