# Order Constraints in Optimal Transport - Digest

**Authors**: Fabian Lim, Laura Wynter, Shiau Hong Lim (IBM Research, Singapore)
**Venue**: ICML 2022 - 39th International Conference on Machine Learning, PMLR 162
**Source**: [Order Constraints in Optimal Transport](https://arxiv.org/abs/2110.07275)

## Overview

Adds explicit order constraints to the optimal-transport plan so that structure - not just feature cost - shapes the coupling, and does so with an efficient, explainable solver. Where earlier work induced structure by sparsifying the plan, this paper imposes ordering relations directly as constraints on the transport matrix, then solves the constrained program with an ADMM scheme that scales far better than generic constrained-OT approaches.

## Key Facts

- Introduces order constraints (OC) into the OT formulation to encode structure that a plain cost cannot express
- For convex costs with efficiently computable gradients, the order-constrained problem is solved by an ADMM variant and is δ-approximable, with derived computationally efficient lower bounds
- Provides the theoretical properties of the method (convergence, approximation)
- Demonstrated on e-SNLI (Stanford NLI with human-annotated rationales) for explainability and on image color-transfer examples
- Frames the structure motivation around text documents with given context, an exact match to ordered-statement documents

## Relevance To This Project

This is the principled, constraint-based counterpart to the regularizer-based OPW (E10-H55) and a direct reference for the E10 batch's central idea - that document structure is order information the content cost is blind to, and that it can be read off a transport plan made order-aware. The explainability result (order-constrained plans align with human rationales on NLI) reinforces the project's interpretable transport-map goal: a structure read should not only score arrangement but show which statements moved. It also frames E10-H54's design choice - rather than re-optimize a constrained plan, H54 pins the content plan and measures its order distortion, a cheaper route to the same structural signal.
