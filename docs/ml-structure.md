## Sections

1. Method
    1. Feature Extractor
    2. Graph Network Blocks
    3. Loss Function and Evaluation Metric
2. Implementation Details
    1. Graph Network Library
    2. Rank Predictor
    3. Human Evaluation Tool
3. Results
4. Analysis and Discussion
    1. Accuracy
    2. Model Understanding
    3. Information Extracted from the Model for Real-world Use
5. Conclusion
    1. Future Work


## Content

* Method
  * Overall architecture
  * Feature extractor (discussion why and how else, i.e. pure GN approach, still feature extraction fits into the GN framework)
    * Architecture
    * Weight sharing between mobile and desktop
  * GN blocks and variants
    * Math. definition
    * Intuition behind them
  * Loss function
    * Loss from background section
    * Huge matrix
    * Sample weighting
    * Batch size relevance and virtual batch size
  * Evaluation
    * Accuracy determination
    * Inference for a new sample
  * Training details (?)
* Implementation details (two Python libraries)
  * Framework choice
  * GN lib
    * Alternatives (DeepMind's)
    * Class diagram
    * Challenges, performance
    * Details
    * Future improvements: parallelization, GPU ops, more default classes
  * Rank predictor
    * Modules (dataset classes, training)
    * Details: parallelization
    * Sacred, TensorBoard
  * Human evaluation tool (optional)
* Results
  * Train/test split
  * Exact configuration (optimizer parameters, etc.)
  * Comparison of feature extraction vs. pure GN
  * GN variants comparison
  * v1 vs. v2 (?)
  * Graph structure usage (link information vs. fully connected)
* Analysis and discussion
  * Human score
  * Comparison with human scores --> discuss that the human score may change over time, it's a tendency of our time
  * Question of upper limit (aspects: plain impossibility, dataset noise, problem of similar and high ranks)
  * Is the model good at easy pairs and bad at hard ones? Tweaked accuracy score consideration and computation.
  * Dataset size, no overfitting, larger model helpful?
  * Is the graph structure relevant after all?
  * Network understanding
    * Activation maps
    * Filters not helpful
    * Hard and easy samples
    * Gradient
  * Helpfulness of the extracted information
  * Comparison with human behavior (qualitative)
* Conclusion
  * Future work
    * Straight-forward
        * Tweak feature extractor architecture
        * Fine-tune feature extractor
        * Larger model to overfit
        * Larger batch sizes
        * Take ranks into consideration (harder and easier pairwise tasks)
        * Transfer capabilities when training on 10,000 to 30,000 --> 50,000 to 90,000
    * More abstract
        * Not purely vision-based
