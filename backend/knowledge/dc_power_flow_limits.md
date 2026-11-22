---
title: What a DC power-flow approximation can and cannot do (reference note)
category: grid
source: POWERIQ author-written reference note (not an official publication)
---
# DC power-flow approximation

The DC approximation assumes flat voltage magnitudes, small angle differences and negligible resistance, so line flows become a linear function of bus injections. It is useful for fast screening of congestion and N-1 style what-if questions.

It does not model reactive power, voltage limits, losses or stability, so it cannot validate that a real system is secure. POWERIQ's network is an illustrative backbone with assumed capacities; results indicate where congestion may appear, not where it will. Use a validated AC study with utility data (for example with pandapower or PyPSA and a documented network model) before any operational decision.
