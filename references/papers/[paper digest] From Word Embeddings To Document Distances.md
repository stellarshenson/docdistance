# From Word Embeddings To Document Distances - Digest

**Authors**: Matt J. Kusner, Yu Sun, Nicholas I. Kolkin, Kilian Q. Weinberger (Washington University in St. Louis)
**Venue**: ICML 2015 (JMLR W&CP vol. 37)
**Source**: [From Word Embeddings To Document Distances](https://proceedings.mlr.press/v37/kusnerb15.html)

## Overview

The paper introduces the Word Mover's Distance (WMD), a distance function between text documents built directly on word2vec embeddings. It measures dissimilarity as the minimum cumulative distance that the embedded words of one document must "travel" to match the embedded words of another, casting the problem as a special case of the Earth Mover's Distance (EMD) transportation problem. The metric is hyperparameter-free, interpretable, and achieves the lowest k-nearest-neighbor classification error against seven baselines on six of eight datasets.

## The Problem

Bag-of-words (BOW) and TF-IDF are the two most common document representations, but they are poorly suited to measuring document distance because their high-dimensional vectors are near-orthogonal and they cannot capture similarity between individual words. The canonical example: `Obama speaks to the media in Illinois` and `The President greets the press in Chicago` share no words after stop-word removal, so BOW reports near-maximum distance, yet the sentences carry nearly identical meaning. Latent representations such as LSI and LDA reduce dimensionality but often do not improve empirical performance on distance-based tasks like nearest-neighbor classification.

## Key Facts

- WMD builds on word2vec (Mikolov et al. 2013), where semantic relationships survive vector arithmetic - `vec(Berlin) - vec(Germany) + vec(France) ≈ vec(Paris)`
- Documents are normalized bag-of-words (nBOW) vectors `d`, points on the (n-1)-simplex; stop words removed
- Word travel cost is the Euclidean distance between word vectors in embedding space: `c(i,j) = ||x_i - x_j||₂`
- The document distance is the optimal value of a linear program over a flow matrix `T`, where `T_ij ≥ 0` is how much of word `i` travels to word `j`
- WMD is a true metric because the word-level cost `c(i,j)` is a metric (Rubner et al. 1998)
- No hyperparameters and straightforward to implement; results are interpretable as sparse word-to-word flows

## How WMD Works

Each document is a weighted point cloud of embedded words. The distance from document A to document B is the minimum cumulative cost to move all of A's word mass onto B's word mass. Formally this is the transportation linear program: minimize the sum of `T_ij · c(i,j)` subject to two flow constraints - the outgoing flow from each word `i` equals its weight `d_i`, and the incoming flow to each word `j` equals its weight `d'_j`. This is exactly the EMD, a well-studied problem with specialized fast solvers.

The optimization moves word mass to semantically similar words. Transforming `Illinois` into `Chicago` is cheap because word2vec places those vectors close together, whereas moving `Japan` into `Chicago` is expensive. When the two documents have different word counts, mass from one word is split across several similar words.

## Speed and Lower Bounds

Solving WMD exactly costs `O(p³ log p)` where `p` is the number of unique words, which becomes prohibitive at scale. The paper introduces cheaper lower bounds used to prune candidates without computing exact WMD.

- **Word Centroid Distance (WCD)** - distance between weighted-average word vectors, `||Xd - Xd'||₂`; derived via the triangle inequality, costs `O(dp)`, very fast but loose
- **Relaxed WMD (RWMD)** - drop one of the two flow constraints; each word moves all its mass to its single nearest word in the other document, costs `O(p²)`, and is a tight lower bound. Taking the maximum of the two one-sided relaxations gives the full RWMD bound
- **Prefetch and prune** - sort all documents by cheap WCD, compute exact WMD for the first `k`, then for the rest prune any whose RWMD bound exceeds the current k-th nearest distance. RWMD tightness lets it prune up to 95% of documents on some datasets

Used directly as a distance, RWMD alone already yields a 0.45 relative-to-BOW kNN error, still better than every baseline. Prefetch-and-prune gives roughly 2x to 5x speedups for the exact method, larger for longer documents.

## Results

Evaluated as kNN classification on eight document datasets (BBCSPORT, TWITTER, RECIPE, OHSUMED, CLASSIC, REUTERS, AMAZON, 20NEWS) against seven baselines (BOW, TF-IDF, Okapi BM25, LSI, LDA, mSDA, Componential Counting Grid).

- WMD achieves the lowest test error on six of eight datasets (all except BBCSPORT and OHSUMED)
- On average WMD produces only 0.42 of the BOW test error, outperforming every competing metric
- Reaches error as low as 2.8% on CLASSIC and 3.5% on REUTERS, beating even transductively trained LDA
- The freely available 3-million-word Google News word2vec model is consistently competitive; in general more training data, not merely more relevant data, yields better embeddings
- OHSUMED is a weak spot, attributed to technical medical terms absent from the embedding vocabulary, which must be discarded

## Why It Works and Limitations

The authors attribute WMD's accuracy to its ability to inherit the high-quality knowledge encoded in word2vec, trained on billions of words - a form of "latent" supervision that benefits even tasks unrelated to the embedding's training corpus. Methods like LDA and LSI do not scale naturally to corpora of that size without approximations that erode the benefit of large-scale data.

Stated limitations and future directions: words absent from the embedding are dropped, which can harm accuracy on technical-vocabulary corpora; the exact metric is the slowest to compute; and document structure is ignored - a proposed extension would penalize word movements between different sections of similarly structured documents (for example, between the introduction and method sections of academic papers).

## Relevance To This Project

WMD gives an embedding-grounded, hyperparameter-free document distance that does not require access to model logits. For agentic pipelines that convert or extract information between documents through frontier models - where KL divergence from logits is unavailable - WMD and its RWMD/WCD lower bounds offer a practical, interpretable measure of how far an output document has drifted from its source.
