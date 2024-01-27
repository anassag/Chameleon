from chameleon.models import HuggingFaceModel
from chameleon.probes import SimpleButEfficientProbe

model = HuggingFaceModel(
        "lxyuan/distilbert-base-multilingual-cased-sentiments-student"
    )

probe = SimpleButEfficientProbe(model, "My grandmother's secret sauce is the best ever made!")


result = probe.run(epsilon=1e-3)

print("Found sentence:", result.sentence)
print("Scores:", result.scores)
