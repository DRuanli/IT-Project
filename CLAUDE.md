# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TUFCI (Top-k Uncertain Frequent Closed Itemsets) is a research implementation for mining top-k closed itemsets in uncertain databases using pattern growth with dynamic programming for support computation.

This is a **data mining research project** with multiple algorithm versions tracking incremental improvements and optimizations.

## Build & Test Commands

### Building the Project
```bash
mvn clean compile
```

### Running the Main Algorithm
```bash
# Compile first
mvn compile

# Run with defaults (minsup=2, tau=0.7, k=5)
java -cp target/classes current.TUFCI src/main/resources/example_100x25.txt

# Run with custom parameters
java -cp target/classes current.TUFCI <input_file> <minsup> <tau> <k>

# Example with explicit parameters
java -cp target/classes current.TUFCI src/main/resources/example_100x25.txt 3 0.8 10
```

### Package and Run as JAR
```bash
mvn package
java -jar target/TUCFI-1.0-SNAPSHOT.jar src/main/resources/example_100x25.txt
```

## Code Architecture

### Package Structure
- `src/main/java/current/` - **Current/latest implementation** (TUFCI.java)
- `src/main/java/version/` - **Historical versions** (TUFCI_1.java through TUFCI_10.java) tracking algorithmic evolution
- `src/main/resources/` - Test datasets in uncertain database format

### Core Algorithm Phases

The TUFCI algorithm operates in **5 phases** with a two-phase mining approach:

1. **Phase 1**: Compute frequent 1-itemsets with single-scan optimization
2. **Phase 2**: Initialize top-k heap with 1-itemsets
3. **Phase 3**: Recursive pattern growth mining (collects all candidates)
4. **Phase 4**: POST-PROCESSING - Verify closure property for all discovered itemsets
5. **Phase 5**: Filter and return top-k truly closed itemsets

**Critical Design**: Phase 4 is a post-processing step added to ensure mathematical correctness. During Phase 3, closure is NOT enforced; all itemsets are collected first, then verified against each other in Phase 4.

### Key Data Structures

**ItemCodec** (`TUFCI.java:31-69`)
- Maps string item identifiers to numeric indices for BitSet representation
- Provides 30-40% speedup with 50-70% memory reduction
- Maintains bijection between strings and indices

**Itemset** (`TUFCI.java:79-168`)
- Uses BitSet internally instead of Set<String> for O(1) operations
- Stores support, probability, and closure flag
- Memory: ~176 bytes vs ~432 bytes per itemset (59% reduction)

**UncertainDatabase** (`TUFCI.java:384-535`)
- Stores transactions with probabilistic item occurrences
- Uses BitSet-based inverted index for efficient set operations
- Supports conditional database projection for pattern growth
- Maintains prefix probabilities for correct support computation

**TopKHeap** (`TUFCI.java:173-233`)
- Min-heap to maintain top-k itemsets by support
- Thread-safe insertion with duplicate prevention
- Dynamic threshold adjustment based on heap state

**SafeCache<K,V>** (`TUFCI.java:305-379`)
- Thread-safe LRU cache with double-checked locking
- Prevents race conditions in parallel execution
- Atomic getOrCompute() prevents duplicate computation
- Bounded to MAX_CACHE_ENTRIES (100,000) to prevent OOM

### Support Computation

**Dynamic Programming Approach** (`TUFCI.java:725-750`)
- `computeBinomialConvolution()`: Computes P_i(X) - probability of exactly i transactions containing X
- Avoids exponential enumeration of possible worlds
- Time: O(n²) where n = number of transactions
- Space: O(n)

**Numerical Stability** (`TUFCI.java:637-715`)
- Uses log-space arithmetic to prevent underflow
- Implements log-sum-exp trick for stability
- Critical for itemsets with many items (product of many small probabilities)

**Distribution Reuse Cache** (P5A optimization, `TUFCI.java:589-626`)
- Caches last computed distribution for itemset extension
- Can refine cached distribution instead of full DP recomputation
- 50-100x faster for related itemsets

### Parallelization Strategy

**Work-Based Task Creation** (`TUFCI.java:1129-1149`)
- `TASK_WORK_THRESHOLD`: Minimum work units (10,000) to justify task creation
- `MAX_TASK_CREATION_DEPTH`: Maximum recursion depth for parallelization (10)
- `MIN_EXTENSIONS_FOR_PARALLEL`: Minimum extensions (2) to parallelize
- Dynamic work estimation prevents overhead on small tasks

**Parallel Tasks**
- `MiningTask` (`TUFCI.java:1155-1230`): Parallel pattern growth
- `SupportComputationTask` (`TUFCI.java:1240-1261`): Parallel support computation for extensions
- `ClosureCheckTask` (`TUFCI.java:1271-1303`): Parallel closure verification

**Usage**: Pass `parallel=true` to `runTUFCI()` to enable ForkJoinPool parallelization

### Closure Checking

**BitSet-Based Optimization** (P9, `TUFCI.java:946-978`)
- Uses BitSet intersections instead of transaction iteration
- OLD: O(|tids| × |items|) = O(1000 × 100) operations
- NEW: O(|items| × bitset_ops) = O(100 × 20) operations
- 50x faster closure checking

**Post-Processing Verification** (P1 fix, `TUFCI.java:1526-1572`)
- Mathematical correctness: X is closed ⟺ ∀Y ⊃ X: Sup(Y) < Sup(X)
- Checks each itemset against all discovered itemsets
- Updates closure flags after all mining completes

## Input File Format

Uncertain database format (see `src/main/resources/example_100x25.txt`):

```
<n_transactions> <n_items>
<tid> <item1>:<prob1> <item2>:<prob2> ...
```

Example:
```
100 25
0 B:0.4 D:0.9 G:0.4 L:0.2 M:0.6
1 C:0.2 D:0.8 E:0.8 F:0.8 J:0.4
```

- Line 1: Metadata (transaction count, item count)
- Line 2+: Transaction ID followed by item:probability pairs
- Probabilities must be in range [0.0, 1.0]

## Version History

The `version/` directory contains 10 incremental versions documenting algorithm evolution:
- TUFCI_1.java: Initial implementation (2191 lines)
- TUFCI_2 - TUFCI_9: Progressive optimizations
- TUFCI_10.java: Latest versioned implementation (3132 lines)
- current/TUFCI.java: Active development version (3138 lines)

**When modifying the algorithm**: Consider saving a snapshot in `version/` before major changes to preserve research history.

## Important Implementation Details

### Cache Management
- **Bounded LRU cache** prevents OutOfMemoryError on large databases
- MAX_CACHE_ENTRIES = 100,000 entries (~40 MB with 400 bytes/entry)
- Cache statistics available via `getCacheHits()`, `getCacheMisses()`, `getHitRate()`

### Thread Safety
- All parallel operations use SafeCache with double-checked locking
- TopKHeap has synchronized insert/getMinSupport methods
- No explicit user locking needed when calling runTUFCI with parallel=true

### Prefix Handling in Conditional Databases
- When projecting to conditional DB, prefix items are removed from transactions
- Prefix probabilities are stored separately and multiplied during support computation
- This ensures correct P(itemset ⊆ transaction) calculation

## Common Gotchas

1. **Do NOT enforce closure during mining** - Closure verification happens in Phase 4 post-processing
2. **ItemCodec is shared across conditional databases** - Don't create new codecs for each projection
3. **Support computation uses log-space** - Convert back to probability space via expSafe()
4. **1-itemsets are always closed by definition** - No verification needed
5. **Parallel execution requires work estimation** - Small tasks run sequentially to avoid overhead
