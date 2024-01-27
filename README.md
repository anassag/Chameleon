# Chameleon Package : I can feel and mimic your sentence

Chameleon package provides an algorithm that finds the closest sentence such that, given a sentiment analysis model and a target sentence, identifies a different sentence which has same sentiment score when rounded to n decimal digits.


The algorithm works as follow :

- For each trial i :
  - Generate a list of i*10 synonyms of each words . The synonyms are ordered by descending cosine similarity.
  - Generate  randomly  a sentence set of synonyms  :
	- If the created sentence is closer to the target : sample from synonyms closer to this sentence for the next 100 iteration 
	- If no closer sentence is found in the closest words resample again from the whole synonyms list.

The algorithm stops after 5 trials or if it finds a sentence that meets the stopping criterion and is sufficiently different from the target (Levenshtein distance >= 30).

Here is an example on how to use :

```ruby

from chameleon.models import HuggingFaceModel
from chameleon.probes import SimpleButEfficientProbe

model = HuggingFaceModel(
        "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )

probe = SimpleButEfficientProbe(model, "My grandmother's secret sauce is the best ever made!")


result = probe.run(epsilon=1e-2)


print("Found sentence:", result.sentence)
print("Scores:", result.scores)

```





