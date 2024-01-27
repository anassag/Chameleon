from chameleon.models import HuggingFaceModel
from chameleon.probes import SimpleButEfficientProbe

target = "The tragedy of only thinking up hilarious tweets for the Summer Olympics."
model = HuggingFaceModel(
        "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )

probe = SimpleButEfficientProbe(model, "I feel it should be a positive thing for us to look")


result = probe.run(epsilon=1e-2)

print("Found sentence:", result.sentence)
print("Scores:", result.scores)
