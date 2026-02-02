# Beam Search for Reconvergent Path Pair Detection

## Target
Given a circuit netlist, find all reconvergent path pairs. Sort these pairs based on a heuristic that allows recursive justification of all reconvergent regions.

## Methodology

### 1. Systematic Identification of Sinks

The exhaustive process begins with a global scan of the netlist to identify every possible point of reconvergence.

- Gate Filtering: The algorithm iterates through all gates in the circuit list, selecting only those with a fan-in count.

- Pairwise Initialization: For every input combination at these gates, a backward search is initiated from it as the starting point ($S$), ensuring that every physical merging point in the circuit is evaluated as a potential reconvergent node ($R$).

### 2. Beam search

- For every direct fan-in to $S$, the algorithm spawns an initial pair. If $S$ has fan-ins from gates $A, B,$ and $C$, the beam is initialized with the pairs $(S \to A, S \to B)$, $(S \to B, S \to C)$, and $(S \to A, S \to C)$.

- Expansion: At each step, the "frontier" gates of a path pair are expanded to their preceding gates in the netlist.

- Multiplication: A single path pair can spawn multiple new pairs if the gates have multiple fan-ins.

- The beams keep spawning backward until they 'meet' at the starting point of the reconvergent path pair ($R$).

### 3. Topological Bounding via LRR

Once a potential end point ($S$) and starting point ($R$) are identified, the algorithm no longer relies solely on beam search; it instead calculates a formal Local Reconvergent Region (LRR).

- Intersection Logic: The LRR is defined as the set of all gates that are both in the transitive fan-out of $R$ and the transitive fan-in of $S$.

- Search Scope: By strictly bounding the search within this "envelope," the algorithm can exhaustively explore every internal path sequence without the exponential complexity of searching the entire circuit.

### 4. Comprehensive Path Enumeration

The modified search uses the LRR to ensure no valid path pairs are ignored.

- Exhaustive Trace: Instead of pruning "low-scoring" paths, the algorithm uses the LRR as a guide to follow every connection from $S$ to $R$.

- Exit Line Tracking: Every signal that leaves the LRR before reaching $R$ is identified as an Exit Line. These are recorded to reduce the search space for the PathConsistencySolver.

### 5. Path Ranking

The Algorithm visits shorter path pairs closer to $S$ first. In the later recursive solving step, these short paths are prioritized to be justified first. 

## References
- [Maamari et al. 2017](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=46788)
- [Beam Search](https://www.geeksforgeeks.org/machine-learning/introduction-to-beam-search-algorithm/)