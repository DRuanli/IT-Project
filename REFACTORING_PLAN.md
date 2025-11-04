# TUFCI Refactoring Plan

## Overview
Refactor TUFCI.java (3,103 lines) to improve readability and maintainability while preserving all logic.

## Refactoring Principles
1. **Remove all verbose comments** - Replace with clean JavaDoc
2. **Extract constants** - All magic numbers become named constants
3. **Logical organization** - Group related code
4. **Consistent naming** - Clear, self-documenting names
5. **No logic changes** - Only structure and documentation

## New File Structure

```
TUFCI.java (New Structure)
├── 1. FILE HEADER (50 lines)
│   ├── Package and imports
│   └── Class-level JavaDoc
│
├── 2. DATA STRUCTURES (600 lines)
│   ├── ItemCodec (40 lines) - String ↔ BitSet mapping
│   ├── Itemset (90 lines) - Itemset representation
│   ├── TopKHeap (90 lines) - Min-heap for top-k
│   ├── LRUCache (60 lines) - Bounded cache with LRU eviction
│   ├── SafeCache (80 lines) - Thread-safe cache wrapper
│   ├── CacheKey (50 lines) - Composite cache key
│   ├── UncertainDatabase (180 lines) - Database representation
│   └── ProjectedUncertainDatabase (180 lines) - Conditional DB
│
├── 3. TUFCI MAIN CLASS (2,400 lines)
│   │
│   ├── 3.1 CONFIGURATION CONSTANTS (100 lines)
│   │   ├── Cache configuration
│   │   ├── Parallelization thresholds
│   │   ├── Numerical stability constants
│   │   └── Algorithm parameters
│   │
│   ├── 3.2 SUPPORT COMPUTATION (400 lines)
│   │   ├── computeBinomialConvolution() - DP for P_i(X)
│   │   ├── computeFrequentness() - Cumulative probability
│   │   ├── findProbabilisticSupport() - Binary search
│   │   ├── refineDistribution() - P5A optimization
│   │   ├── computeSupport() - Main support computation
│   │   └── DistributionCache - ThreadLocal cache
│   │
│   ├── 3.3 NUMERICAL STABILITY (100 lines)
│   │   ├── NumericalStability class
│   │   ├── safeLog() - Log-space computation
│   │   ├── expSafe() - Safe exponential
│   │   └── logProductFromMap() - Product in log-space
│   │
│   ├── 3.4 FREQUENT 1-ITEMSETS (200 lines)
│   │   ├── computeFrequent1Itemsets() - Phase 1
│   │   └── collectItemProbabilities() - Single-scan optimization
│   │
│   ├── 3.5 PATTERN GROWTH MINING (800 lines)
│   │   ├── mineRecursiveCollect() - Main mining loop
│   │   ├── processExtensionCollect() - Process single extension
│   │   ├── generateExtensions() - Generate candidates
│   │   ├── MiningTaskCollect - Parallel task
│   │   ├── SupportComputationTask - Parallel support
│   │   └── Pruning methods (ESUB, geometric mean, etc.)
│   │
│   ├── 3.6 CLOSURE VERIFICATION (200 lines)
│   │   ├── verifyClosureProperty() - Phase 4 post-processing
│   │   ├── deduplicateItemsets() - Remove duplicates
│   │   └── checkClosure() - BitSet-based checking
│   │
│   ├── 3.7 MAIN ALGORITHM (300 lines)
│   │   ├── runTUFCI() - 5-phase algorithm
│   │   └── computeMaxCacheEntries() - Dynamic sizing
│   │
│   ├── 3.8 I/O & UTILITIES (200 lines)
│   │   ├── loadUncertainDatabase() - File parsing
│   │   ├── printResults() - Output formatting
│   │   └── Helper methods
│   │
│   └── 3.9 MAIN ENTRY POINT (100 lines)
│       └── main() - Command-line interface
│
└── 4. HELPER CLASSES (50 lines)
    ├── SupportResult - Support computation result
    ├── ExtensionCandidate - Extension data holder
    └── ClosureCheckTask - Parallel closure check
```

## Refactoring Changes

### Constants Extracted
```java
// Cache Configuration
private static final int DEFAULT_MAX_CACHE_ENTRIES = 100_000;
private static final long TARGET_CACHE_MEMORY_BYTES = 1_000_000_000L; // 1GB
private static final double CACHE_MEMORY_RATIO = 0.25; // 25% of available

// Parallelization
private static final int MIN_TASK_WORK_UNITS = 10_000;
private static final int MAX_PARALLELIZATION_DEPTH = 10;
private static final int MIN_EXTENSIONS_FOR_PARALLEL = 2;
private static final int MIN_CANDIDATES_FOR_PARALLEL_GENERATION = 5;

// Numerical Stability
private static final double EPSILON = 1e-9;
private static final double PROBABILITY_TOLERANCE = 1e-6;
private static final double UNDERFLOW_THRESHOLD = 1e-300;

// Algorithm
private static final int DEFAULT_MINSUP = 2;
private static final double DEFAULT_TAU = 0.7;
private static final int DEFAULT_K = 5;
```

### Method Naming Improvements
```java
// Before: unclear
private static void mineRecursiveCollect(...)

// After: descriptive
private static void explorePatternTree(...)

// Before: abbreviation
private static int computeGlobalUpperBound(...)

// After: clear
private static int computeMaxPossibleSupport(...)
```

### JavaDoc Style
```java
/**
 * Computes probabilistic support using dynamic programming.
 *
 * <p>Calculates P_i(X) = probability that itemset X appears in exactly i transactions
 * using binomial convolution to avoid exponential enumeration of possible worlds.
 *
 * @param transactionProbabilities probability of X in each transaction
 * @return distribution array where result[i] = P_i(X)
 * @complexity O(n²) time, O(n) space where n = number of transactions
 */
private static double[] computeBinomialConvolution(double[] transactionProbabilities) {
    // Implementation...
}
```

### Code Organization Example

**Before**: Mixed concerns
```java
public static List<Itemset> runTUFCI(...) {
    // 300 lines of mixed validation, initialization, phases, cleanup
}
```

**After**: Separated phases
```java
public static List<Itemset> runTUFCI(...) {
    validateParameters(minsup, tau, k);
    SafeCache<CacheKey, SupportResult> cache = initializeCache(database);

    List<Itemset> frequent1 = phase1_ComputeFrequent1Itemsets(database, minsup, tau, cache);
    TopKHeap topk = phase2_InitializeTopKHeap(k, minsup, frequent1);
    List<Itemset> allItemsets = phase3_PatternGrowthMining(database, frequent1, topk, ...);
    int closedCount = phase4_VerifyClosureProperty(allItemsets);
    List<Itemset> results = phase5_FilterAndRankResults(allItemsets, k);

    return results;
}
```

## Implementation Strategy

Due to file size (3,103 lines), refactoring will be done in stages:

### Stage 1: Data Structures (Complete rewrite with clean JavaDoc)
- ItemCodec, Itemset, TopKHeap, LRUCache, SafeCache, CacheKey
- UncertainDatabase, ProjectedUncertainDatabase
- SupportResult, ExtensionCandidate

### Stage 2: Core Algorithm (Refactor TUFCI class)
- Extract all constants
- Clean up method signatures
- Add comprehensive JavaDoc
- Reorganize by functionality

### Stage 3: Verification
- Compile and test
- Verify logic preservation
- Performance comparison

## Expected Outcomes

1. **Readability**: -50% noise from removing verbose comments
2. **Maintainability**: Constants extracted, logical organization
3. **Documentation**: Clean JavaDoc for all public methods
4. **Performance**: Identical (no logic changes)
5. **Lines of Code**: ~2,800 lines (down from 3,103)

## Timeline
- Stage 1 (Data Structures): 1-2 hours
- Stage 2 (Main Algorithm): 2-3 hours
- Stage 3 (Verification): 30 minutes
- **Total**: 3-6 hours for complete refactoring
