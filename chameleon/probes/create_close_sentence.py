import spacy
from pytest import approx
import numpy as np
from .base import BaseProbe, ProbeResult
from ..models.base import BaseModel
from chameleon.models import BaseModel


class SimpleButEfficientProbe(BaseProbe):
    """This class defines the abstract interface for all probes.

    Parameters
    ----------
    model : BaseModel
        A wrapper of the model to probe.
    target : str
        The target sentence which sentiment scores should be matched.
    """

    def __init__(self, model: BaseModel, target: str):
        self.model = model
        self.target = target
        self.target_scores = self.model.predict(self.target)
        self.spacy_model = spacy.load("en_core_web_lg")

        for string in self.spacy_model.vocab.strings:
            self.spacy_model.vocab[string]

    def fetch_tag(self, tag):
        """This method returns

        1. 'np' for proper nouns, 'n' for all other nouns

        2. 'a' for adjectives

        3. 'v' for verbs

        4. 'r' for adverbs

        5. None for all other tag"""

        if tag in ["NNP", "NNPS"]:

            return "np"

        elif tag in ["JJ", "JJR", "JJS"]:

            return "a"

        elif tag in ["RB", "RBR", "RBS"]:

            return "r"

        elif tag in ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]:

            return "v"

        elif tag in ["NN", "NNS"]:

            return "n"

        else:
            return None

    def get_synonyms(self, word, n_syn):
        "this method return n_syn closest tokens to a given word ordered by  descending similarity"
        word = self.spacy_model.vocab[word]

        queries = [
            w
            for w in word.vocab
            if w.is_lower == word.is_lower
            and w.prob >= -20
            and np.count_nonzero(w.vector)
        ]

        by_similarity = sorted(queries, key=lambda w: word.similarity(w), reverse=True)

        return [word.lower_] + [
            w.lower_ for w in by_similarity[: n_syn + 1] if w.lower_ != word.lower_
        ]

    def levenshtein_distance(self, str1, str2):
        "Computes the levenshtein_distance"
        if len(str1) > len(str2):
            str1, str2 = str2, str1

        distances = range(len(str1) + 1)
        for index2, char2 in enumerate(str2):
            new_distances = [index2 + 1]
            for index1, char1 in enumerate(str1):
                if char1 == char2:
                    new_distances.append(distances[index1])
                else:
                    new_distances.append(
                        1
                        + min(
                            (
                                distances[index1],
                                distances[index1 + 1],
                                new_distances[-1],
                            )
                        )
                    )
            distances = new_distances

        return distances[-1]

    def is_close(self, sentence, epsilon):
        """This method returns

        1- True  if the sentiment scores of the input and target sentences rounded up to epsilon are equal

        2- False otherwise"""

        scores = self.model.predict(sentence)

        for k, v in scores.items():

            if scores[k] != approx(self.target_scores[k], abs=epsilon):

                return False

        return True

    def distance(self, sentence):
        """This methods returns the L_1 distance between the sentence and target scores"""

        scores = self.model.predict(sentence)
        distance = 0
        for k, v in scores.items():
            distance += abs(self.target_scores[k] - scores[k])

        return distance

    def get_sampling_index(self, high, i):
        """this method is usefull to get a sampling interval around an index i<high
        it returns :
        - (i-1,i+2) if i<high-1
        - (high-3,high-1)  i==high-1
        - (0,3)  if i==0"""

        if high == 1:
            return (0, 1)

        if i == 0:
            return (0, 3)

        if 1 <= i < high - 1:
            return (i - 1, i + 2)

        if i == high - 1:
            return (high - 3, high)

    def run_single_trial(self, epsilon, n_syn, max_iter) -> ProbeResult:
        """
        This method returns an alternate version(s) of the sentence passed by replacing words with their closest synonyms.
        At each iterations:
        1-  draw a random combination of synonym.
        2-  If the created sentence has the closest sentiment score to the target sentence : sample from synonyms closer to this sentence for the next 100 iteration
        3-  return to step 1

        the latter algorithms stops when the created sentence and target sentence rounded to epsilon decimals are equal.
          (the  created sentence should have a levenshtein_distance >=30 from the target sentence)
        args:

        sentence (String) = the input sentence
        epsilon (int) = the number of decimals set by the user
        max_iter= maximum number of iterations

        returns:

        Closest sentence to the target sentence, or the sentence that verifies the stopping criterion .
        """

        sentence_combination = []

        doc = self.spacy_model(self.target)

        for token in doc:

            short_pos = self.fetch_tag(token.tag_)

            # ignore stopwords
            if token.is_stop:
                sentence_combination.append([token.text])
                continue
            else:
                sentence_combination.append(self.get_synonyms(token.text, n_syn))
            # if POS is noun, adj, adv, or verb - get similar words
            # if short_pos is not None:
            #     sentence_combination.append(self.get_synonyms(token.text, n_syn))
            # else do nothing
            # else:
            #     sentence_combination.append([token.text])
            #     continue

        syn_number_array = np.array(
            [len(synonyms) for synonyms in sentence_combination]
        )
        total_combination_possible = np.prod(syn_number_array)
        smallest_dist = np.inf
        best_sentence = ""
        low_high_list = [(0, len_syn) for len_syn in syn_number_array]
        n_passed = 0
        for i in range(min(total_combination_possible, max_iter)):
            n_passed += 1
            if n_passed == int(1e2):
                low_high_list = [(0, len_syn) for len_syn in syn_number_array]

            sampled_index = [np.random.randint(*low_high) for low_high in low_high_list]
            alternate_sentence = [
                synonym[index]
                for synonym, index in zip(sentence_combination, sampled_index)
            ]
            alternate_sentence = " ".join(alternate_sentence)

            if self.levenshtein_distance(alternate_sentence, self.target) >= 30:

                distance = self.distance(alternate_sentence)

                if self.is_close(alternate_sentence, epsilon):

                    print(
                        "==============SENTENCE FOUND ENDING TRIALS====================="
                    )
                    probe_result = ProbeResult(
                        alternate_sentence, self.model.predict(alternate_sentence)
                    )

                    return probe_result

                if distance < smallest_dist:
                    best_sentence = alternate_sentence
                    smallest_dist = distance
                    low_high_list = [
                        self.get_sampling_index(len_syn, index)
                        for len_syn, index in zip(syn_number_array, sampled_index)
                    ]
                    n_passed = 0

        probe_result = ProbeResult(best_sentence, self.model.predict(best_sentence))

        return probe_result

    def run(self, epsilon) -> ProbeResult:
        """
        This method runs multiple trials : it first create 10 synonyms per word, run the single trial method to find the best sentence within this set.
        If no sentence is found generate 20 synonyms per word and so on.

        After 5 trials if no sentence is found return the sentence that has the closest sentiment score with
        """

        smallest_dist = np.inf
        best_sentence = " "
        max_iter = int(1e4)
        for i, n_syn in enumerate(range(10, 60, 10)):

            print(
                f"============Trial N°{i+1}, generating {n_syn} synonyms per word========="
            )
            # As the number of synonym increaseas we raise maximum number of iterations
            trial_prob = self.run_single_trial(epsilon, n_syn, max_iter * (i + 1))
            distance = self.distance(trial_prob.sentence)

            if distance < smallest_dist:
                best_sentence = trial_prob.sentence
                smallest_dist = distance

            if self.is_close(trial_prob.sentence, epsilon):
                print(best_sentence)
                print("Scores :", self.model.predict(best_sentence))
                print(
                    f"Reminder of target sentence ",
                    self.target,
                    "\n with scores ",
                    self.model.predict(self.target),
                )

                return trial_prob
            elif best_sentence == "":
                print("No sentence found that is far enough from target")
            else:
                print(
                    f"Reminder of target sentence ",
                    self.target,
                    "\n with scores ",
                    self.model.predict(self.target),
                )
                print(
                    f"Best sentence found after trial N°{i+1} : ",
                    best_sentence,
                    "\n with scores ",
                    self.model.predict(best_sentence),
                )

        return ProbeResult(best_sentence, self.model.predict(best_sentence))
