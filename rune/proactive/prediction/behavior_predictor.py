"""N-gram based tool sequence prediction for RUNE.

Records tool-call sequences and uses n-gram models to predict
the next likely tool invocation.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from rune.utils.logger import get_logger

log = get_logger(__name__)

_MAX_HISTORY = 500


class BehaviorPredictor:
    """Predicts next tool calls using n-gram frequency analysis.

    Records a stream of tool names and builds bigram/trigram frequency
    tables to predict the most likely next tool invocation.
    """

    __slots__ = ("_history",)

    def __init__(self) -> None:
        self._history: list[str] = []

    def record_tool_call(self, tool_name: str) -> None:
        """Record a tool invocation.

        Parameters:
            tool_name: The name of the tool that was called.
        """
        self._history.append(tool_name)

        # Trim history to prevent unbounded growth
        if len(self._history) > _MAX_HISTORY:
            self._history = self._history[-_MAX_HISTORY:]

        log.debug("behavior_recorded", tool=tool_name, history_len=len(self._history))

    def predict_next(self, n: int = 3) -> list[tuple[str, float]]:
        """Predict the next tool call based on recent history.

        Uses n-gram matching (tries trigram first, falls back to bigram).

        Parameters:
            n: Maximum number of predictions to return.

        Returns:
            List of (tool_name, probability) tuples, sorted by probability descending.
        """
        # Require sufficient history for meaningful predictions.
        # With <5 items, n-gram probabilities are unreliable (e.g., 2 items
        # = single bigram = 100% confidence, which is noise, not a pattern).
        if len(self._history) < 5:
            return []

        predictions: Counter[str] = Counter()

        # Try trigram prediction (most specific)
        if len(self._history) >= 3:
            trigrams = self._build_ngrams(self._history, 3)
            key = tuple(self._history[-2:])
            if key in trigrams:
                for tool, count in trigrams[key].items():
                    total = sum(trigrams[key].values())
                    predictions[tool] += (count / total) * 2.0  # weighted higher

        # Bigram prediction (fallback)
        bigrams = self._build_ngrams(self._history, 2)
        key_bi = (self._history[-1],)
        if key_bi in bigrams:
            for tool, count in bigrams[key_bi].items():
                total = sum(bigrams[key_bi].values())
                predictions[tool] += count / total

        # Normalise scores to probabilities
        total_score = sum(predictions.values())
        if total_score == 0:
            return []

        result = [
            (tool, round(score / total_score, 4))
            for tool, score in predictions.most_common(n)
        ]

        log.debug("behavior_predicted", predictions=result)
        return result

    def _build_ngrams(
        self,
        sequence: list[str],
        n: int,
    ) -> dict[tuple[str, ...], Counter[str]]:
        """Build an n-gram frequency table from a sequence.

        Parameters:
            sequence: The list of tool names.
            n: The n-gram size (2 for bigram, 3 for trigram).

        Returns:
            A dict mapping context tuples to Counter of next-token frequencies.
        """
        ngrams: dict[tuple[str, ...], Counter[str]] = defaultdict(Counter)

        for i in range(len(sequence) - n + 1):
            window = sequence[i : i + n]
            context = tuple(window[:-1])
            target = window[-1]
            ngrams[context][target] += 1

        return ngrams

    def get_frequent_sequences(self, min_count: int = 3) -> list[tuple[str, ...]]:
        """Return frequently occurring tool sequences.

        Parameters:
            min_count: Minimum occurrence count to include a sequence.

        Returns:
            List of frequent tool-name tuples (length 2 or 3).
        """
        sequences: Counter[tuple[str, ...]] = Counter()

        # Count bigrams
        for i in range(len(self._history) - 1):
            seq = tuple(self._history[i : i + 2])
            sequences[seq] += 1

        # Count trigrams
        for i in range(len(self._history) - 2):
            seq = tuple(self._history[i : i + 3])
            sequences[seq] += 1

        return [seq for seq, count in sequences.most_common() if count >= min_count]
