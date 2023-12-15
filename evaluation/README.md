# Evaluation Metrics

## Context Relevance
Compares the query with the retrieved context

## Grounded-ness / Faithful-ness
Compares the retrieved context to the response

## Answer Relevance
Compares the response to the query

## Context Recall
Compares the retrieved context to the ground-truth answer

## Answer Correctness
Compares the response to the ground-truth answer


# TruLens

## RAG Triad
Context Relevance: (query and context) Is the retrieved context relevant to the query?

Grounded-ness: (context and response) Is the response supported by the context?

Answer Relevance: (response and query) Is the response relevant to the query?

# Ragas

## Metrics-Driven Development¶
While creating a fundamental LLM application may be straightforward, the challenge lies in its ongoing maintenance and continuous enhancement. Ragas’ vision is to facilitate the continuous improvement of LLM and RAG applications by embracing the ideology of Metrics-Driven Development (MDD).

MDD is a product development approach that relies on data to make well-informed decisions. This approach entails the ongoing monitoring of essential metrics over time, providing valuable insights into an application’s performance.

Our mission is to establish an open-source standard for applying MDD to LLM and RAG applications.

Evaluation: This enables you to assess LLM applications and conduct experiments in a metric-assisted manner, ensuring high dependability and reproducibility.

Monitoring: It allows you to gain valuable and actionable insights from production data points, facilitating the continuous improvement of the quality of your LLM application.

## Primary Metrics

Faithfulness (TruLens: Grounded-ness) - the factual consistency of the answer to the context base on the question.

Context Precision (TruLens: Context Relevance) - a measure of how relevant the retrieved context is to the question. Conveys quality of the retrieval pipeline.

Answer Relevance (TruLens: Answer Relevance) - a measure of how relevant the answer is to the question

Context Recall: measures the ability of the retriever to retrieve all the necessary information needed to answer the question.

### Faithfulness
(TruLens: Grounded-ness)
This measures the factual consistency of the generated answer against the given context. It is calculated from answer and retrieved context. The answer is scaled to (0,1) range. Higher the better.

The generated answer is regarded as faithful if all the claims that are made in the answer can be inferred from the given context. To calculate this a set of claims from the generated answer is first identified. Then each one of these claims are cross checked with given context to determine if it can be inferred from given context or not. The faithfulness score is given by divided by

### Answer Relevance
(TruLens: Answer Relevance)
The evaluation metric, Answer Relevance, focuses on assessing how pertinent the generated answer is to the given prompt. A lower score is assigned to answers that are incomplete or contain redundant information. This metric is computed using the question and the answer, with values ranging between 0 and 1, where higher scores indicate better relevancy.

An answer is deemed relevant when it directly and appropriately addresses the original question. Importantly, our assessment of answer relevance does not consider factuality but instead penalizes cases where the answer lacks completeness or contains redundant details. To calculate this score, the LLM is prompted to generate an appropriate question for the generated answer multiple times, and the mean cosine similarity between these generated questions and the original question is measured. The underlying idea is that if the generated answer accurately addresses the initial question, the LLM should be able to generate questions from the answer that align with the original question.

### Context Precision
(TruLens: Context Relevance)
Context Precision is a metric that evaluates whether all of the ground-truth relevant items present in the contexts are ranked higher or not. Ideally all the relevant chunks must appear at the top ranks. This metric is computed using the question and the contexts, with values ranging between 0 and 1, where higher scores indicate better precision.

### Context Relevancy
This metric gauges the relevancy of the retrieved context, calculated based on both the question and contexts. The values fall within the range of (0, 1), with higher values indicating better relevancy.

Ideally, the retrieved context should exclusively contain essential information to address the provided query.

### Context Recall
Context recall measures the extent to which the retrieved context aligns with the annotated answer, treated as the ground truth. It is computed based on the ground truth and the retrieved context, and the values range between 0 and 1, with higher values indicating better performance.

To estimate context recall from the ground truth answer, each sentence in the ground truth answer is analyzed to determine whether it can be attributed to the retrieved context or not. In an ideal scenario, all sentences in the ground truth answer should be attributable to the retrieved context.

### Answer semantic similarity
The concept of Answer Semantic Similarity pertains to the assessment of the semantic resemblance between the generated answer and the ground truth. This evaluation is based on the ground truth and the answer, with values falling within the range of 0 to 1. A higher score signifies a better alignment between the generated answer and the ground truth.

Measuring the semantic similarity between answers can offer valuable insights into the quality of the generated response. This evaluation utilizes a cross-encoder model to calculate the semantic similarity score.

### Answer Correctness
The assessment of Answer Correctness involves gauging the accuracy of the generated answer when compared to the ground truth. This evaluation relies on the ground truth and the answer, with scores ranging from 0 to 1. A higher score indicates a closer alignment between the generated answer and the ground truth, signifying better correctness.

Answer correctness encompasses two critical aspects: semantic similarity between the generated answer and the ground truth, as well as factual similarity. These aspects are combined using a weighted scheme to formulate the answer correctness score. Users also have the option to employ a ‘threshold’ value to round the resulting score to binary, if desired.

### Aspect Critique
This is designed to assess submissions based on predefined aspects such as harmlessness and correctness. Additionally, users have the flexibility to define their own aspects for evaluating submissions according to their specific criteria. The output of aspect critiques is binary, indicating whether the submission aligns with the defined aspect or not. This evaluation is performed using the ‘answer’ as input.

Critiques within the LLM evaluators evaluate submissions based on the provided aspect. Ragas Critiques offers a range of predefined aspects like correctness, harmfulness, etc. (Please refer to SUPPORTED_ASPECTS for a complete list). If you prefer, you can also create custom aspects to evaluate submissions according to your unique requirements.

The strictness parameter plays a crucial role in maintaining a certain level of self-consistency in predictions, with an ideal range typically falling between 2 to 4. It’s important to note that the scores obtained from aspect critiques are binary and do not contribute to the final Ragas score due to their non-continuous nature.

* harmfulness, maliciousness, coherence, correctness, conciseness, etc...

