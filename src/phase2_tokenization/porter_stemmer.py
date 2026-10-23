"""
Porter Stemmer — Hand-coded implementation.

Based on the original algorithm by Martin F. Porter (1980).
This is a faithful implementation of the classic 5-step suffix stripping
algorithm. We hand-code it rather than importing nltk.stem.PorterStemmer
to stay consistent with the project's "core must be self-coded" philosophy
and to allow fine-tuning for technical tokens.

Reference: https://tartarus.org/martin/PorterStemmer/def.txt
"""

from __future__ import annotations


def _is_consonant(word: str, i: int) -> bool:
    """Check if character at position i in word is a consonant."""
    c = word[i]
    if c in "aeiou":
        return False
    if c == "y":
        if i == 0:
            return True
        return not _is_consonant(word, i - 1)
    return True


def _measure(word: str) -> int:
    """
    Compute the 'measure' m of a word — the number of VC (vowel-consonant)
    sequences in the word after stripping trailing vowels/consonants.

    [C](VC)^m[V]  where m is the measure.
    """
    n = len(word)
    i = 0
    # Skip leading consonants
    while i < n and _is_consonant(word, i):
        i += 1

    m = 0
    while i < n:
        # Skip vowels
        while i < n and not _is_consonant(word, i):
            i += 1
        if i >= n:
            break
        # Skip consonants
        while i < n and _is_consonant(word, i):
            i += 1
        m += 1
    return m


def _has_vowel(stem: str) -> bool:
    """Check if stem contains at least one vowel."""
    return any(not _is_consonant(stem, i) for i in range(len(stem)))


def _ends_double_consonant(word: str) -> bool:
    """Check if word ends with a double consonant (e.g., 'hopping' -> pp)."""
    if len(word) < 2:
        return False
    return word[-1] == word[-2] and _is_consonant(word, len(word) - 1)


def _ends_cvc(word: str) -> bool:
    """
    Check if word ends with consonant-vowel-consonant where the last
    consonant is not w, x, or y.
    """
    if len(word) < 3:
        return False
    if (
        _is_consonant(word, len(word) - 1)
        and not _is_consonant(word, len(word) - 2)
        and _is_consonant(word, len(word) - 3)
    ):
        return word[-1] not in "wxy"
    return False


def _replace_suffix(word: str, suffix: str, replacement: str) -> str:
    """Replace suffix of word with replacement string."""
    return word[: -len(suffix)] + replacement if suffix else word + replacement


def _step1a(word: str) -> str:
    """Step 1a: plural morphology."""
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s"):
        return word[:-1]
    return word


def _step1b(word: str) -> str:
    """Step 1b: -ed, -ing morphology."""
    if word.endswith("eed"):
        stem = word[:-3]
        if _measure(stem) > 0:
            return word[:-1]  # Remove just the 'd'
        return word

    modified = False
    if word.endswith("ed"):
        stem = word[:-2]
        if _has_vowel(stem):
            word = stem
            modified = True
    elif word.endswith("ing"):
        stem = word[:-3]
        if _has_vowel(stem):
            word = stem
            modified = True

    if modified:
        if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
            return word + "e"
        if _ends_double_consonant(word) and word[-1] not in "lsz":
            return word[:-1]
        if _measure(word) == 1 and _ends_cvc(word):
            return word + "e"

    return word


def _step1c(word: str) -> str:
    """Step 1c: y -> i when stem contains vowel."""
    if word.endswith("y"):
        stem = word[:-1]
        if _has_vowel(stem):
            return stem + "i"
    return word


def _step2(word: str) -> str:
    """Step 2: map double suffixes to single suffixes."""
    pairs = [
        ("ational", "ate"), ("tional", "tion"), ("enci", "ence"),
        ("anci", "ance"), ("izer", "ize"), ("abli", "able"),
        ("alli", "al"), ("entli", "ent"), ("eli", "e"),
        ("ousli", "ous"), ("ization", "ize"), ("ation", "ate"),
        ("ator", "ate"), ("alism", "al"), ("iveness", "ive"),
        ("fulness", "ful"), ("ousness", "ous"), ("aliti", "al"),
        ("iviti", "ive"), ("biliti", "ble"),
    ]
    for suffix, replacement in pairs:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if _measure(stem) > 0:
                return stem + replacement
            return word
    return word


def _step3(word: str) -> str:
    """Step 3: further suffix normalization."""
    pairs = [
        ("icate", "ic"), ("ative", ""), ("alize", "al"),
        ("iciti", "ic"), ("ical", "ic"), ("ful", ""),
        ("ness", ""),
    ]
    for suffix, replacement in pairs:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if _measure(stem) > 0:
                return stem + replacement
            return word
    return word


def _step4(word: str) -> str:
    """Step 4: remove suffixes where measure > 1."""
    suffixes = [
        "al", "ance", "ence", "er", "ic", "able", "ible",
        "ant", "ement", "ment", "ent", "ion", "ou", "ism",
        "ate", "iti", "ous", "ive", "ize",
    ]
    for suffix in suffixes:
        if word.endswith(suffix):
            stem = word[: -len(suffix)]
            if suffix == "ion":
                if _measure(stem) > 1 and len(stem) > 0 and stem[-1] in "st":
                    return stem
            else:
                if _measure(stem) > 1:
                    return stem
            return word
    return word


def _step5a(word: str) -> str:
    """Step 5a: remove trailing 'e' if measure > 1, or measure == 1 and not *o."""
    if word.endswith("e"):
        stem = word[:-1]
        m = _measure(stem)
        if m > 1:
            return stem
        if m == 1 and not _ends_cvc(stem):
            return stem
    return word


def _step5b(word: str) -> str:
    """Step 5b: remove double consonant 'll' -> 'l' if measure > 1."""
    if word.endswith("ll") and _measure(word) > 1:
        return word[:-1]
    return word


def porter_stem(word: str) -> str:
    """
    Apply the Porter stemming algorithm to a single word.

    Args:
        word: The lowercase word to stem.

    Returns:
        The stemmed form.
    """
    if len(word) <= 2:
        return word

    word = _step1a(word)
    word = _step1b(word)
    word = _step1c(word)
    word = _step2(word)
    word = _step3(word)
    word = _step4(word)
    word = _step5a(word)
    word = _step5b(word)

    return word
