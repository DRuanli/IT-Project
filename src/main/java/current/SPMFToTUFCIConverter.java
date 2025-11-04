package current;

import java.io.*;
import java.util.*;

/**
 * SPMFToTUFCIConverter: Converts SPMF transaction format to TUFCI uncertain database format.
 *
 * <p>SPMF format is deterministic (items appear with certainty), while TUFCI format
 * is probabilistic (each item has an occurrence probability). This converter bridges
 * the two formats by applying configurable probability assignment strategies.
 *
 * <p><b>SPMF Format</b>:
 * <pre>
 * item1 item2 item3:utility:metadata
 * item4 item5
 * </pre>
 *
 * <p><b>TUFCI Format</b>:
 * <pre>
 * n_transactions n_items
 * 0 item1:0.8 item2:0.6 item3:0.9
 * 1 item4:0.7 item5:0.5
 * </pre>
 *
 * <p><b>Usage Examples</b>:
 * <pre>
 * # Uniform probability (all items = 0.8)
 * java SPMFToTUFCIConverter input.txt output.txt --strategy uniform --uniform-prob 0.8
 *
 * # Random probability in range [0.5, 1.0] with seed
 * java SPMFToTUFCIConverter input.txt output.txt --strategy random --random-min 0.5 --random-max 1.0 --random-seed 42
 *
 * # Frequency-based (frequent items get higher probability)
 * java SPMFToTUFCIConverter input.txt output.txt --strategy frequency
 * </pre>
 *
 * @author TUFCI Research Team
 * @version 1.0
 */
public class SPMFToTUFCIConverter {

    // ========================================================================
    // PROBABILITY ASSIGNMENT STRATEGIES
    // ========================================================================

    /**
     * Strategy interface for assigning probabilities to items in transactions.
     *
     * <p>Implementations define different approaches for converting deterministic
     * SPMF items to probabilistic TUFCI items.
     */
    public interface ProbabilityStrategy {
        /**
         * Get probability for an item in a transaction.
         *
         * @param item the item identifier
         * @param transactionId the transaction index (0-based)
         * @param globalItemFrequency map of item → total occurrence count across all transactions
         * @return probability in range [0.0, 1.0]
         */
        double getProbability(String item, int transactionId, Map<String, Integer> globalItemFrequency);

        /**
         * Get descriptive name of this strategy.
         */
        String getName();
    }

    /**
     * Assigns the same fixed probability to all items.
     *
     * <p>Simplest strategy - useful for creating uniform uncertain databases.
     *
     * <p><b>Example</b>: UniformStrategy(0.8) assigns 80% probability to every item.
     */
    public static class UniformStrategy implements ProbabilityStrategy {
        private final double probability;

        /**
         * Creates uniform strategy with specified probability.
         *
         * @param probability the fixed probability for all items (must be in [0.0, 1.0])
         * @throws IllegalArgumentException if probability out of range
         */
        public UniformStrategy(double probability) {
            if (probability < 0.0 || probability > 1.0) {
                throw new IllegalArgumentException(
                    "Probability must be in [0.0, 1.0], got: " + probability
                );
            }
            this.probability = probability;
        }

        @Override
        public double getProbability(String item, int tid, Map<String, Integer> freq) {
            return probability;
        }

        @Override
        public String getName() {
            return "Uniform(" + probability + ")";
        }
    }

    /**
     * Assigns random probabilities within a specified range.
     *
     * <p>Uses seeded Random for reproducibility. Each item gets a random probability
     * independently drawn from [min, max].
     *
     * <p><b>Example</b>: RandomStrategy(0.5, 1.0, 42) assigns random probabilities
     * between 50% and 100% using seed 42.
     */
    public static class RandomStrategy implements ProbabilityStrategy {
        private final Random random;
        private final double min;
        private final double max;

        /**
         * Creates random strategy with specified range and seed.
         *
         * @param min minimum probability (inclusive)
         * @param max maximum probability (inclusive)
         * @param seed random seed for reproducibility
         * @throws IllegalArgumentException if min/max out of range or min > max
         */
        public RandomStrategy(double min, double max, long seed) {
            if (min < 0.0 || max > 1.0 || min > max) {
                throw new IllegalArgumentException(
                    "Invalid range [" + min + ", " + max + "] - must satisfy 0 ≤ min ≤ max ≤ 1"
                );
            }
            this.min = min;
            this.max = max;
            this.random = new Random(seed);
        }

        @Override
        public double getProbability(String item, int tid, Map<String, Integer> freq) {
            return min + (max - min) * random.nextDouble();
        }

        @Override
        public String getName() {
            return "Random[" + min + ", " + max + "]";
        }
    }

    /**
     * Assigns probabilities based on global item frequency.
     *
     * <p>Frequent items get higher probabilities, rare items get lower probabilities.
     * Uses linear scaling: prob = 0.3 + 0.7 * (freq / maxFreq).
     *
     * <p><b>Rationale</b>: Items appearing in many transactions are likely "core" items
     * with higher certainty, while rare items may be noise or outliers.
     *
     * <p><b>Example</b>: If item A appears in 100 transactions (max frequency) and
     * item B appears in 10, then:
     * <ul>
     *   <li>P(A) = 0.3 + 0.7 * (100/100) = 1.0</li>
     *   <li>P(B) = 0.3 + 0.7 * (10/100) = 0.37</li>
     * </ul>
     */
    public static class FrequencyStrategy implements ProbabilityStrategy {
        private static final double MIN_PROB = 0.3;  // Minimum probability for rare items
        private static final double PROB_RANGE = 0.7; // Range for scaling [0.3, 1.0]

        @Override
        public double getProbability(String item, int tid, Map<String, Integer> globalFreq) {
            int itemFreq = globalFreq.getOrDefault(item, 1);
            int maxFreq = globalFreq.values().stream().max(Integer::compareTo).orElse(1);

            // Linear scaling: 0.3 + 0.7 * (freq / maxFreq)
            double ratio = (double) itemFreq / maxFreq;
            return MIN_PROB + PROB_RANGE * ratio;
        }

        @Override
        public String getName() {
            return "Frequency-based";
        }
    }

    // ========================================================================
    // CORE CONVERSION METHODS
    // ========================================================================

    /**
     * Parses SPMF transaction database file.
     *
     * <p>SPMF format supports optional metadata after colons:
     * <pre>
     * item1 item2 item3:utility:metadata
     * item4 item5
     * </pre>
     *
     * <p>This parser strips metadata and returns pure item sets.
     *
     * @param filename path to SPMF file
     * @return list of transactions, each transaction is a set of item strings
     * @throws IOException if file cannot be read
     */
    public static List<Set<String>> parseSPMF(String filename) throws IOException {
        List<Set<String>> transactions = new ArrayList<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) {
                    continue; // Skip empty lines
                }

                Set<String> transaction = new HashSet<>();
                String[] tokens = line.split("\\s+");

                for (String token : tokens) {
                    // Strip metadata after ':' if present
                    // Example: "A:100:metadata" → "A"
                    int colonIndex = token.indexOf(':');
                    String item = colonIndex >= 0 ? token.substring(0, colonIndex) : token;

                    if (!item.isEmpty()) {
                        transaction.add(item);
                    }
                }

                if (!transaction.isEmpty()) {
                    transactions.add(transaction);
                }
            }
        }

        return transactions;
    }

    /**
     * Applies probability strategy to deterministic transactions.
     *
     * <p>Converts each transaction from Set&lt;String&gt; to Map&lt;String, Double&gt;
     * where each item is mapped to its assigned probability.
     *
     * @param transactions list of deterministic transactions
     * @param strategy probability assignment strategy
     * @return list of probabilistic transactions (item → probability mappings)
     */
    public static List<Map<String, Double>> applyProbabilities(
            List<Set<String>> transactions,
            ProbabilityStrategy strategy) {

        // Compute global item frequencies for frequency-based strategy
        Map<String, Integer> globalFreq = new HashMap<>();
        for (Set<String> transaction : transactions) {
            for (String item : transaction) {
                globalFreq.put(item, globalFreq.getOrDefault(item, 0) + 1);
            }
        }

        // Apply strategy to each transaction
        List<Map<String, Double>> probabilisticTransactions = new ArrayList<>();
        for (int tid = 0; tid < transactions.size(); tid++) {
            Map<String, Double> probTransaction = new HashMap<>();
            for (String item : transactions.get(tid)) {
                double prob = strategy.getProbability(item, tid, globalFreq);
                probTransaction.put(item, prob);
            }
            probabilisticTransactions.add(probTransaction);
        }

        return probabilisticTransactions;
    }

    /**
     * Writes transactions to TUFCI format file.
     *
     * <p>TUFCI format:
     * <pre>
     * n_transactions n_items
     * tid1 item1:prob1 item2:prob2 ...
     * tid2 item3:prob3 item4:prob4 ...
     * </pre>
     *
     * @param filename output file path
     * @param transactions list of probabilistic transactions
     * @param allItems set of all unique items (for header)
     * @throws IOException if file cannot be written
     */
    public static void writeTUFCI(
            String filename,
            List<Map<String, Double>> transactions,
            Set<String> allItems) throws IOException {

        try (BufferedWriter writer = new BufferedWriter(new FileWriter(filename))) {
            // Write header: n_transactions n_items
            writer.write(transactions.size() + " " + allItems.size());
            writer.newLine();

            // Write transactions
            for (int tid = 0; tid < transactions.size(); tid++) {
                Map<String, Double> transaction = transactions.get(tid);

                if (transaction.isEmpty()) {
                    // Empty transaction: just write tid
                    writer.write(String.valueOf(tid));
                } else {
                    // Write tid followed by item:probability pairs
                    writer.write(tid + " ");

                    List<String> items = new ArrayList<>(transaction.keySet());
                    Collections.sort(items); // Sort for deterministic output

                    for (int i = 0; i < items.size(); i++) {
                        String item = items.get(i);
                        double prob = transaction.get(item);

                        writer.write(item + ":" + String.format("%.4f", prob));

                        if (i < items.size() - 1) {
                            writer.write(" ");
                        }
                    }
                }

                writer.newLine();
            }
        }
    }

    /**
     * Converts SPMF file to TUFCI file using specified strategy.
     *
     * <p>Main orchestrator method that chains parsing, probability application, and writing.
     *
     * @param inputFile SPMF input file path
     * @param outputFile TUFCI output file path
     * @param strategy probability assignment strategy
     * @param verbose if true, prints conversion statistics
     * @throws IOException if file I/O fails
     */
    public static void convert(
            String inputFile,
            String outputFile,
            ProbabilityStrategy strategy,
            boolean verbose) throws IOException {

        if (verbose) {
            System.out.println("=== SPMF to TUFCI Converter ===");
            System.out.println("Input:    " + inputFile);
            System.out.println("Output:   " + outputFile);
            System.out.println("Strategy: " + strategy.getName());
            System.out.println();
        }

        // Phase 1: Parse SPMF
        long startTime = System.currentTimeMillis();
        List<Set<String>> transactions = parseSPMF(inputFile);
        long parseTime = System.currentTimeMillis() - startTime;

        // Collect all unique items
        Set<String> allItems = new HashSet<>();
        for (Set<String> transaction : transactions) {
            allItems.addAll(transaction);
        }

        if (verbose) {
            System.out.println("Parsed SPMF database:");
            System.out.println("  Transactions: " + transactions.size());
            System.out.println("  Unique items: " + allItems.size());
            System.out.println("  Parse time:   " + parseTime + " ms");
            System.out.println();
        }

        // Phase 2: Apply probabilities
        startTime = System.currentTimeMillis();
        List<Map<String, Double>> probabilisticTransactions =
            applyProbabilities(transactions, strategy);
        long applyTime = System.currentTimeMillis() - startTime;

        if (verbose) {
            System.out.println("Applied probability strategy:");
            System.out.println("  Strategy:     " + strategy.getName());
            System.out.println("  Apply time:   " + applyTime + " ms");
            System.out.println();
        }

        // Phase 3: Write TUFCI
        startTime = System.currentTimeMillis();
        writeTUFCI(outputFile, probabilisticTransactions, allItems);
        long writeTime = System.currentTimeMillis() - startTime;

        if (verbose) {
            System.out.println("Wrote TUFCI database:");
            System.out.println("  File:         " + outputFile);
            System.out.println("  Write time:   " + writeTime + " ms");
            System.out.println();
            System.out.println("Conversion complete! Total time: " +
                (parseTime + applyTime + writeTime) + " ms");
        }
    }

    // ========================================================================
    // COMMAND-LINE INTERFACE
    // ========================================================================

    /**
     * Prints usage instructions to stderr.
     */
    private static void printUsage() {
        System.err.println("Usage: java SPMFToTUFCIConverter <input.txt> <output.txt> [OPTIONS]");
        System.err.println();
        System.err.println("Converts SPMF transaction format to TUFCI uncertain database format.");
        System.err.println();
        System.err.println("Options:");
        System.err.println("  --strategy <uniform|random|frequency>");
        System.err.println("      Probability assignment strategy (default: uniform)");
        System.err.println();
        System.err.println("  --uniform-prob <value>");
        System.err.println("      Fixed probability for uniform strategy (default: 0.8)");
        System.err.println("      Example: --uniform-prob 0.9");
        System.err.println();
        System.err.println("  --random-min <value>");
        System.err.println("      Minimum probability for random strategy (default: 0.5)");
        System.err.println();
        System.err.println("  --random-max <value>");
        System.err.println("      Maximum probability for random strategy (default: 1.0)");
        System.err.println();
        System.err.println("  --random-seed <value>");
        System.err.println("      Random seed for reproducibility (default: 42)");
        System.err.println();
        System.err.println("  --verbose");
        System.err.println("      Print conversion statistics");
        System.err.println();
        System.err.println("Examples:");
        System.err.println("  # Uniform probability 0.8");
        System.err.println("  java SPMFToTUFCIConverter input.txt output.txt --strategy uniform --uniform-prob 0.8");
        System.err.println();
        System.err.println("  # Random probability [0.5, 1.0] with seed 42");
        System.err.println("  java SPMFToTUFCIConverter input.txt output.txt --strategy random --random-min 0.5 --random-max 1.0 --random-seed 42");
        System.err.println();
        System.err.println("  # Frequency-based probability");
        System.err.println("  java SPMFToTUFCIConverter input.txt output.txt --strategy frequency --verbose");
    }

    /**
     * Main entry point for command-line usage.
     *
     * @param args command-line arguments
     */
    public static void main(String[] args) {
        // Validate minimum arguments
        if (args.length < 2) {
            printUsage();
            System.exit(1);
        }

        String inputFile = args[0];
        String outputFile = args[1];

        // Parse options
        String strategyType = "uniform";
        double uniformProb = 0.8;
        double randomMin = 0.5;
        double randomMax = 1.0;
        long randomSeed = 42;
        boolean verbose = false;

        for (int i = 2; i < args.length; i++) {
            switch (args[i]) {
                case "--strategy":
                    if (i + 1 >= args.length) {
                        System.err.println("ERROR: --strategy requires an argument");
                        System.exit(1);
                    }
                    strategyType = args[++i];
                    break;

                case "--uniform-prob":
                    if (i + 1 >= args.length) {
                        System.err.println("ERROR: --uniform-prob requires a value");
                        System.exit(1);
                    }
                    try {
                        uniformProb = Double.parseDouble(args[++i]);
                    } catch (NumberFormatException e) {
                        System.err.println("ERROR: Invalid value for --uniform-prob: " + args[i]);
                        System.exit(1);
                    }
                    break;

                case "--random-min":
                    if (i + 1 >= args.length) {
                        System.err.println("ERROR: --random-min requires a value");
                        System.exit(1);
                    }
                    try {
                        randomMin = Double.parseDouble(args[++i]);
                    } catch (NumberFormatException e) {
                        System.err.println("ERROR: Invalid value for --random-min: " + args[i]);
                        System.exit(1);
                    }
                    break;

                case "--random-max":
                    if (i + 1 >= args.length) {
                        System.err.println("ERROR: --random-max requires a value");
                        System.exit(1);
                    }
                    try {
                        randomMax = Double.parseDouble(args[++i]);
                    } catch (NumberFormatException e) {
                        System.err.println("ERROR: Invalid value for --random-max: " + args[i]);
                        System.exit(1);
                    }
                    break;

                case "--random-seed":
                    if (i + 1 >= args.length) {
                        System.err.println("ERROR: --random-seed requires a value");
                        System.exit(1);
                    }
                    try {
                        randomSeed = Long.parseLong(args[++i]);
                    } catch (NumberFormatException e) {
                        System.err.println("ERROR: Invalid value for --random-seed: " + args[i]);
                        System.exit(1);
                    }
                    break;

                case "--verbose":
                    verbose = true;
                    break;

                default:
                    System.err.println("ERROR: Unknown option: " + args[i]);
                    printUsage();
                    System.exit(1);
            }
        }

        // Create strategy based on type
        ProbabilityStrategy strategy;
        try {
            switch (strategyType.toLowerCase()) {
                case "uniform":
                    strategy = new UniformStrategy(uniformProb);
                    break;

                case "random":
                    strategy = new RandomStrategy(randomMin, randomMax, randomSeed);
                    break;

                case "frequency":
                    strategy = new FrequencyStrategy();
                    break;

                default:
                    System.err.println("ERROR: Unknown strategy: " + strategyType);
                    System.err.println("Valid strategies: uniform, random, frequency");
                    System.exit(1);
                    return; // Unreachable, but satisfies compiler
            }
        } catch (IllegalArgumentException e) {
            System.err.println("ERROR: " + e.getMessage());
            System.exit(1);
            return;
        }

        // Perform conversion
        try {
            convert(inputFile, outputFile, strategy, verbose);
        } catch (IOException e) {
            System.err.println("ERROR: Conversion failed: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
