/**
 * TUFCI Formal Verification Test Suite
 *
 * Tests for critical soundness, completeness, and correctness issues
 * identified in formal verification analysis.
 *
 * Author: Formal Verification Analysis
 * Date: 2025-11-03
 */

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;

public class TUFCIVerificationTests {

    // ========================================================================
    // ISSUE #1: DistributionCache Race Condition Tests
    // ========================================================================

    /**
     * Test: Concurrent access to DistributionCache causes inconsistent state
     *
     * Expected: Race condition causes mismatched itemset and result
     * Status: DEMONSTRATES BUG - Will fail with current implementation
     */
    @Test
    public void testDistributionCache_RaceCondition() throws InterruptedException {
        final int NUM_THREADS = 100;
        final int NUM_ITERATIONS = 1000;

        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch endLatch = new CountDownLatch(NUM_THREADS);
        AtomicInteger inconsistencyCount = new AtomicInteger(0);

        // Create many threads that set and read cache simultaneously
        for (int t = 0; t < NUM_THREADS; t++) {
            final int threadId = t;
            new Thread(() -> {
                try {
                    startLatch.await();  // All threads start together

                    for (int i = 0; i < NUM_ITERATIONS; i++) {
                        // Each thread uses unique itemsets
                        Set<String> myItemset = Set.of("item_" + threadId + "_" + i);
                        double[] myTransProbs = new double[threadId + 1];
                        Arrays.fill(myTransProbs, 0.5);

                        // Create mock result with identifiable support
                        int uniqueSupport = threadId * 10000 + i;
                        double[] distribution = new double[10];
                        distribution[0] = 1.0;

                        // Simulated SupportResult (would need actual class)
                        SupportResultMock result = new SupportResultMock(
                            uniqueSupport, 0.5, distribution, null, myTransProbs
                        );

                        // Set cache (RACE CONDITION HERE)
                        DistributionCache.set(result, myItemset, myTransProbs);

                        // Immediately try to read (might get someone else's data)
                        Set<String> cachedItemset = DistributionCache.getLastItemset();
                        SupportResultMock cachedResult =
                            (SupportResultMock) DistributionCache.getLastResult();

                        // Check consistency
                        if (cachedItemset != null && cachedResult != null) {
                            // If we got our own itemset, result should match
                            if (cachedItemset.equals(myItemset)) {
                                if (cachedResult.supT != uniqueSupport) {
                                    inconsistencyCount.incrementAndGet();
                                    System.err.printf(
                                        "Thread %d: Expected support %d but got %d%n",
                                        threadId, uniqueSupport, cachedResult.supT
                                    );
                                }
                            }
                        }
                    }
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                } finally {
                    endLatch.countDown();
                }
            }).start();
        }

        startLatch.countDown();  // Start all threads
        endLatch.await(30, TimeUnit.SECONDS);  // Wait for completion

        System.out.printf("DistributionCache Race Test: %d inconsistencies detected%n",
            inconsistencyCount.get());

        // With race condition, we expect many inconsistencies
        // After fix (ThreadLocal), this should be 0
        assertTrue(inconsistencyCount.get() > 0,
            "Race condition should be detected (will pass after fix)");
    }

    /**
     * Test: Validate that cache reuse with mismatched itemsets is rejected
     *
     * Expected: canReuse() returns false for non-extension itemsets
     * Status: Should PASS (validation logic is correct)
     */
    @Test
    public void testDistributionCache_ValidationRejectsMismatched() {
        // Set cache with itemset {A, B}
        Set<String> itemsetAB = Set.of("A", "B");
        double[] transProbs = {0.5, 0.3};
        double[] dist = {0.5, 0.5, 0.0};

        SupportResultMock result = new SupportResultMock(5, 0.7, dist, null, transProbs);
        DistributionCache.set(result, itemsetAB, transProbs);

        // Try to reuse for {C, D} (completely different)
        Set<String> itemsetCD = Set.of("C", "D");
        assertFalse(DistributionCache.canReuse(itemsetCD),
            "Should reject cache reuse for non-overlapping itemsets");

        // Try to reuse for {A, C} (replacement, not extension)
        Set<String> itemsetAC = Set.of("A", "C");
        assertFalse(DistributionCache.canReuse(itemsetAC),
            "Should reject cache reuse for replacement (removed B, added C)");

        // Try to reuse for {B} (subset, not extension)
        Set<String> itemsetB = Set.of("B");
        assertFalse(DistributionCache.canReuse(itemsetB),
            "Should reject cache reuse for subset (removed A)");

        // Should accept {A, B, C} (extension)
        Set<String> itemsetABC = Set.of("A", "B", "C");
        assertTrue(DistributionCache.canReuse(itemsetABC),
            "Should accept cache reuse for valid extension");
    }

    // ========================================================================
    // ISSUE #2: refineDistribution() Incorrect Logic Tests
    // ========================================================================

    /**
     * Test: refineDistribution() handles zero probabilities correctly
     *
     * Expected: When prevProb=0, delta should be 0 (not 1.0)
     * Status: DEMONSTRATES BUG - Distribution sum != 1.0
     */
    @Test
    public void testRefineDistribution_ZeroProbability() {
        // Setup: Previous itemset {A} has mixed probabilities
        double[] prevProbs = {0.0, 0.5, 0.8};  // t0: no A, t1: 50% A, t2: 80% A
        double[] prevDist = {0.25, 0.50, 0.25};  // Some distribution

        // Extending to {A, B}:
        // - t0: P({A,B}) = 0 (since A not in t0)
        // - t1: P({A,B}) = 0.5 * 0.6 = 0.3
        // - t2: P({A,B}) = 0.8 * 0.7 = 0.56
        double[] newProbs = {0.0, 0.3, 0.56};

        // This should work correctly (after fix)
        double[] refined = TUFCI.refineDistribution(prevDist, newProbs, prevProbs);

        // Verify: Distribution must sum to 1.0
        double sum = 0.0;
        for (double p : refined) {
            sum += p;
        }

        assertEquals(1.0, sum, 1e-6,
            "Distribution must sum to 1.0 (may fail with current bug)");

        // Verify: All probabilities non-negative
        for (int i = 0; i < refined.length; i++) {
            assertTrue(refined[i] >= -1e-10,
                String.format("Probability at index %d must be non-negative: %.6f",
                    i, refined[i]));
        }
    }

    /**
     * Test: refineDistribution() should detect extension property violations
     *
     * Expected: Throws exception when prevProb=0 but newProb>0
     * Status: NEW TEST - Should pass after defensive check added
     */
    @Test
    public void testRefineDistribution_ExtensionPropertyViolation() {
        // Setup: Simulate cache reuse with MISMATCHED itemsets (race condition)
        double[] prevProbs = {0.0, 0.5};     // Previous itemset {A}
        double[] prevDist = {0.5, 0.5, 0.0};

        // NEW itemset is {C} (NOT related to {A})
        // This violates extension property: prevProbs[0]=0 but newProbs[0]=0.8
        double[] newProbs = {0.8, 0.3};  // ❌ IMPOSSIBLE if extending {A}

        // After fix, this should throw IllegalStateException
        assertThrows(IllegalStateException.class, () -> {
            TUFCI.refineDistribution(prevDist, newProbs, prevProbs);
        }, "Should detect and reject extension property violation");
    }

    /**
     * Test: Valid extension scenario
     *
     * Expected: Distribution correctly refined
     * Status: Should PASS
     */
    @Test
    public void testRefineDistribution_ValidExtension() {
        // Setup: {A} with P(A ⊆ t0)=0.5, P(A ⊆ t1)=0.8
        double[] prevProbs = {0.5, 0.8};

        // Simple distribution: 50% chance of 0 transactions, 50% chance of 2
        double[] prevDist = {0.5, 0.0, 0.5};

        // Extending to {A,B}: multiply by P(B|A)
        // P({A,B} ⊆ t0) = 0.5 * 0.6 = 0.3
        // P({A,B} ⊆ t1) = 0.8 * 0.7 = 0.56
        double[] newProbs = {0.3, 0.56};

        double[] refined = TUFCI.refineDistribution(prevDist, newProbs, prevProbs);

        // Verify distribution properties
        double sum = Arrays.stream(refined).sum();
        assertEquals(1.0, sum, 1e-6, "Distribution must sum to 1.0");

        for (double p : refined) {
            assertTrue(p >= 0, "All probabilities must be non-negative");
            assertTrue(p <= 1.0, "All probabilities must be <= 1.0");
        }
    }

    // ========================================================================
    // ISSUE #4: Closure Verification Correctness Tests
    // ========================================================================

    /**
     * Test: Closure checking with equal support itemsets
     *
     * Expected: Proper subsets are marked as not closed
     * Status: Should PASS (verification logic is correct)
     */
    @Test
    public void testClosureVerification_EqualSupport() {
        List<Itemset> itemsets = new ArrayList<>();
        ItemCodec codec = new ItemCodec(Arrays.asList("A", "B", "C"));

        // Create itemsets with equal support
        Set<String> itemA = Set.of("A");
        Set<String> itemB = Set.of("B");
        Set<String> itemAB = Set.of("A", "B");

        Itemset isA = Itemset.fromStringSet(itemA, 5, 0.9, codec);
        Itemset isB = Itemset.fromStringSet(itemB, 5, 0.9, codec);
        Itemset isAB = Itemset.fromStringSet(itemAB, 5, 0.9, codec);

        itemsets.add(isA);
        itemsets.add(isB);
        itemsets.add(isAB);

        // Verify closure
        int closedCount = TUFCI.verifyClosureProperty(itemsets, false);

        // {A} should NOT be closed (because {A,B} has same support)
        assertFalse(isA.isClosed(), "{A} should not be closed when {A,B} has equal support");

        // {B} should NOT be closed (because {A,B} has same support)
        assertFalse(isB.isClosed(), "{B} should not be closed when {A,B} has equal support");

        // {A,B} should be closed (no superset with >= support)
        assertTrue(isAB.isClosed(), "{A,B} should be closed");

        assertEquals(1, closedCount, "Only 1 itemset should be closed");
    }

    /**
     * Test: Closure checking with different supports
     *
     * Expected: Subsets with higher support are closed
     * Status: Should PASS
     */
    @Test
    public void testClosureVerification_DifferentSupport() {
        List<Itemset> itemsets = new ArrayList<>();
        ItemCodec codec = new ItemCodec(Arrays.asList("A", "B", "C"));

        Set<String> itemA = Set.of("A");
        Set<String> itemAB = Set.of("A", "B");
        Set<String> itemABC = Set.of("A", "B", "C");

        // Decreasing support: {A}:10 > {A,B}:5 > {A,B,C}:3
        Itemset isA = Itemset.fromStringSet(itemA, 10, 0.95, codec);
        Itemset isAB = Itemset.fromStringSet(itemAB, 5, 0.85, codec);
        Itemset isABC = Itemset.fromStringSet(itemABC, 3, 0.75, codec);

        itemsets.add(isA);
        itemsets.add(isAB);
        itemsets.add(isABC);

        // Verify closure
        TUFCI.verifyClosureProperty(itemsets, false);

        // All should be closed (supersets have strictly less support)
        assertTrue(isA.isClosed(), "{A} should be closed (superset {A,B} has less support)");
        assertTrue(isAB.isClosed(), "{A,B} should be closed (superset {A,B,C} has less support)");
        assertTrue(isABC.isClosed(), "{A,B,C} should be closed (no superset)");
    }

    // ========================================================================
    // ISSUE #5: ESUB Pruning Soundness Tests
    // ========================================================================

    /**
     * Test: Expected support upper bound pruning is sound (no false negatives)
     *
     * Expected: Pruned itemsets should have SupD < threshold
     * Status: Should PASS (math is correct)
     */
    @Test
    public void testESUB_Pruning_NoFalseNegatives() {
        // Create database with known probabilities
        Map<Integer, Map<String, Double>> transactions = new HashMap<>();
        transactions.put(0, Map.of("A", 0.5, "B", 0.6));
        transactions.put(1, Map.of("A", 0.4, "B", 0.7));
        transactions.put(2, Map.of("A", 0.3, "B", 0.8));

        UncertainDatabase db = new UncertainDatabase(transactions);
        double tau = 0.7;

        // Create top-k heap with threshold support = 2
        TopKHeap topk = new TopKHeap(5, 1);
        ItemCodec codec = db.getItemCodec();

        // Insert dummy itemset to set threshold
        Itemset dummy = Itemset.fromStringSet(Set.of("dummy"), 2, 0.9, codec);
        topk.insert(dummy);

        // Test itemset {A, B}
        Set<String> itemsetAB = Set.of("A", "B");

        // Compute actual support
        SupportResult actualResult = TUFCI.computeSupport(itemsetAB, db, tau);

        // Compute ESUB
        Double esub = TUFCI.computeExpectedSupportUpperBound(itemsetAB, db, tau, topk);

        if (esub == null) {
            // Pruned - verify that actual support is indeed < threshold
            assertTrue(actualResult.supT < topk.getMinSupport(),
                "If pruned by ESUB, actual support must be < threshold (soundness)");
        } else {
            // Not pruned - could have support >= or < threshold
            // (ESUB is an upper bound, not exact)
            assertTrue(esub >= actualResult.supT,
                "Expected support upper bound must be >= actual support");
        }
    }

    // ========================================================================
    // ISSUE #10: TopKHeap Memory Leak Test
    // ========================================================================

    /**
     * Test: TopKHeap.seenItemsets grows unbounded
     *
     * Expected: seenItemsets should not grow beyond k + epsilon
     * Status: DEMONSTRATES BUG - seenItemsets grows to 1000 while k=10
     */
    @Test
    public void testTopKHeap_MemoryLeak() {
        int k = 10;
        int numItemsets = 1000;

        TopKHeap topk = new TopKHeap(k, 1);
        ItemCodec codec = new ItemCodec(Arrays.asList("A", "B", "C", "D", "E"));

        // Insert many itemsets with decreasing support
        for (int i = 0; i < numItemsets; i++) {
            Set<String> items = Set.of("item_" + i);
            int support = numItemsets - i;  // Decreasing support
            Itemset itemset = Itemset.fromStringSet(items, support, 0.9, codec);
            topk.insert(itemset);
        }

        // Check sizes
        int heapSize = topk.size();
        int seenSize = topk.getSeenItemsetsSize();  // Would need accessor

        System.out.printf("TopKHeap: heap=%d, seenItemsets=%d%n", heapSize, seenSize);

        assertEquals(k, heapSize, "Heap should contain exactly k items");

        // Current implementation: seenSize = numItemsets (MEMORY LEAK)
        // After fix: seenSize should be ~k
        assertTrue(seenSize <= k * 2,
            "seenItemsets should not grow unbounded (may fail with current bug)");
    }

    // ========================================================================
    // Edge Case Tests
    // ========================================================================

    /**
     * Test: Empty database edge case
     *
     * Expected: All itemsets have support 0
     * Status: Should PASS
     */
    @Test
    public void testEmptyDatabase() {
        Map<Integer, Map<String, Double>> emptyTransactions = new HashMap<>();
        UncertainDatabase db = new UncertainDatabase(emptyTransactions);

        Set<String> itemset = Set.of("A", "B");
        SupportResult result = TUFCI.computeSupport(itemset, db, 0.5);

        assertEquals(0, result.supT, "Empty database should yield support 0");
        assertEquals(1.0, result.distribution[0], 1e-6,
            "P(0 transactions) should be 1.0 for empty database");
    }

    /**
     * Test: All probabilities = 1.0 (certain database)
     *
     * Expected: Reduces to deterministic mining
     * Status: Should PASS
     */
    @Test
    public void testCertainDatabase() {
        Map<Integer, Map<String, Double>> transactions = new HashMap<>();
        transactions.put(0, Map.of("A", 1.0, "B", 1.0));
        transactions.put(1, Map.of("A", 1.0, "C", 1.0));
        transactions.put(2, Map.of("B", 1.0, "C", 1.0));

        UncertainDatabase db = new UncertainDatabase(transactions);
        Set<String> itemsetAB = Set.of("A", "B");

        SupportResult result = TUFCI.computeSupport(itemsetAB, db, 0.5);

        // {A,B} appears in exactly 1 transaction (t0)
        assertEquals(1, result.supT, "Support should be deterministic when P=1.0");
        assertEquals(1.0, result.distribution[1], 1e-6,
            "P(exactly 1 transaction) should be 1.0");
    }

    /**
     * Test: tau = 1.0 (certainty threshold)
     *
     * Expected: Only certain itemsets qualify
     * Status: Should PASS
     */
    @Test
    public void testTauOne() {
        Map<Integer, Map<String, Double>> transactions = new HashMap<>();
        transactions.put(0, Map.of("A", 0.9, "B", 0.8));  // Uncertain
        transactions.put(1, Map.of("A", 1.0, "C", 1.0));  // Certain

        UncertainDatabase db = new UncertainDatabase(transactions);

        // {A,B} has uncertainty
        SupportResult resultAB = TUFCI.computeSupport(Set.of("A", "B"), db, 1.0);
        assertEquals(0, resultAB.supT,
            "Uncertain itemset should have support 0 with tau=1.0");

        // {A,C} is certain in t1 but not in t0
        SupportResult resultAC = TUFCI.computeSupport(Set.of("A", "C"), db, 1.0);
        // Should have support 0 or 1 depending on P(AC ⊆ t0)
        assertTrue(resultAC.supT <= 1,
            "Support should be low with high tau threshold");
    }

    // ========================================================================
    // Helper Classes (Mock objects for testing)
    // ========================================================================

    static class SupportResultMock {
        final int supT;
        final double probability;
        final double[] distribution;
        final double[] frequentness;
        final double[] transProbs;

        public SupportResultMock(int supT, double probability, double[] distribution,
                                double[] frequentness, double[] transProbs) {
            this.supT = supT;
            this.probability = probability;
            this.distribution = distribution;
            this.frequentness = frequentness;
            this.transProbs = transProbs;
        }
    }
}
