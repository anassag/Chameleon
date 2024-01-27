# Brief review of “Chainpoll: A high efficacy method for LLM hallucination detection” by Friel and Sanyal


## Summary
"ChainPoll: A High-Efficacy Method for LLM Hallucination Detection" introduces ChainPoll, an innovative approach for detecting hallucinations in large language models (LLMs).

To determine if a completion contains hallucinations, the process involves three steps: first, GPT-3.5-Turbo is asked through a detailed and engineered prompt whether the completion includes any hallucinations. This step is repeated several times, typically five. Finally, the fraction of 'yes' responses to the total number of responses from these inquiries is calculated, yielding a score between 0 and 1.

## Strengths  


- Empirical validation: Comprehensive testing on four different datasets demonstrates ChainPoll's effectiveness compared to other known approches (like SelfcheckGPT).
- Explainability and efficiency : In the prompt used to construct the ChainPoll metric, the LLM evaluates whether the initial completion includes any hallucinations, providing a justification for its decision through a chain-of-thought (CoT) explanation which helps the users understand the features used behind the scores. Furthermore, Table 5 shows that ChainPoll has also the best cost/quality tradeoff among the provided metrics.



## Weaknesses
The main weakness, in my point of view is the generalizability of ChainPoll across even wider varieties of data, indeed the authors used a newly created dataset that doesn't include many state-of-the-art benchmarks for hallucination detection: WikiBio, TruthfulQA, FactualityPrompt, FactScore, KoLA-KC, HaluEval.  (Other examples are provided Table 1 in \cite{zhang2023siren} or  in \cite{chang2024})\\
Consequently, it remains debatable whether Chainpoll consistently outperforms other state-of-the-art methods in general cases.



## Improvement idea

I also believe that during the development of the chainpoll metric, minor modifications to the prompt can be made to enhance the data gathered about hallucinations. For instance, we could integrate an extra prompt that instructs GPT-3.5 to evaluate the response on a scale from 1 to 5, which can then be translated into a binary yes or no answer. This approach would maintain a cost comparable to the original while enriching the diversity of the information obtained. One can also employ an additional classifier after the Chainpoll output (which includes a binary answer and its justification) to determine whether the input contains any hallucinations.




