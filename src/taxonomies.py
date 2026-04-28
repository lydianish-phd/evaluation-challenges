# UGC taxonomy codes
GRAMMAR = "grammar"
SPELLING = "spelling"
WORD_ELONGATION = "word_elongation"
CAPITALISATION = "capitalisation"
INFORMAL_ABBREVIATIONS = "informal_abbreviations"
INFORMAL_ACRONYMS = "informal_acronyms"
HASHTAGS_SUBREDDITS = "hashtags_subreddits"
URLS_MENTIONS_RTS = "urls_mentions_rts"
EMOTICONS_EMOJIS = "emoticons_emojis"
ATYPICAL_PUNCTUATION = "atypical_punctuation"
OVERT_PROFANITY = "overt_profanity"
SELF_CENSORED_PROFANITY = "self_censored_profanity"


UGC_TAXONOMY = {
    GRAMMAR: {
        "id": 1,
        "label": "Grammar",
        "description": "Grammar",
    },
    SPELLING: {
        "id": 2,
        "label": "Spelling",
        "description": "Spelling",
    },
    WORD_ELONGATION: {
        "id": 3,
        "label": "Word elongation or letter repetitions",
        "description": "Word elongation or letter repetitions",
    },
    CAPITALISATION: {
        "id": 4,
        "label": "Capitalisation",
        "description": (
            "Capitalisation, e.g. missing at the beginning of a sentence or a "
            "proper noun, using all caps or case swapping for emphasis"
        ),
    },
    INFORMAL_ABBREVIATIONS: {
        "id": 5,
        "label": "Informal abbreviations",
        "description": "Informal abbreviations",
    },
    INFORMAL_ACRONYMS: {
        "id": 6,
        "label": "Informal acronyms",
        "description": "Informal acronyms",
    },
    HASHTAGS_SUBREDDITS: {
        "id": 7,
        "label": "Hashtags and subreddits",
        "description": "Hashtags and subreddits",
    },
    URLS_MENTIONS_RTS: {
        "id": 8,
        "label": "URLs, @-mentions, user IDs, and retweet marks",
        "description": "URLs, @-mentions to user IDs, and retweet marks (RT)",
    },
    EMOTICONS_EMOJIS: {
        "id": 9,
        "label": "Emoticons and emojis",
        "description": "Emoticons and emojis",
    },
    ATYPICAL_PUNCTUATION: {
        "id": 10,
        "label": "Atypical punctuation",
        "description": "Atypical punctuation, e.g. missing or repeated",
    },
    OVERT_PROFANITY: {
        "id": 11,
        "label": "Overt profanity",
        "description": "Overt profanity",
    },
    SELF_CENSORED_PROFANITY: {
        "id": 12,
        "label": "Self-censored profanity",
        "description": "Self-censored profanity",
    },
}

# PMUMT taxonomy codes
PMUMT_LETTER_DELETION_ADDITION = "1"
PMUMT_MISSING_DIACRITICS = "2"
PMUMT_PHONETIC_WRITING = "3"
PMUMT_TOKENISATION_ERROR = "4"
PMUMT_WRONG_VERB_TENSE = "5"
PMUMT_HASHTAG_MENTION_URL = "6"
PMUMT_WRONG_GENDER_NUMBER = "7"
PMUMT_INCONSISTENT_CASING = "8"
PMUMT_EMOJI = "9"
PMUMT_NAMED_ENTITY = "10"
PMUMT_CONTRACTION = "11"
PMUMT_GRAPHEMIC_PUNCTUATION_STRETCHING = "12"
PMUMT_INTERJECTIONS = "13"


PMUMT_TAXONOMY = {
    PMUMT_LETTER_DELETION_ADDITION: {
        "label": "Letter deletion/addition",
        "description": "Letter deletion or addition.",
    },
    PMUMT_MISSING_DIACRITICS: {
        "label": "Missing diacritics",
        "description": "Missing accents or diacritic marks.",
    },
    PMUMT_PHONETIC_WRITING: {
        "label": "Phonetic writing",
        "description": "Words written phonetically rather than according to standard spelling.",
    },
    PMUMT_TOKENISATION_ERROR: {
        "label": "Tokenisation error",
        "description": "Non-standard tokenisation, including missing or extra spaces.",
    },
    PMUMT_WRONG_VERB_TENSE: {
        "label": "Wrong verb tense",
        "description": "Non-standard or incorrect verb tense.",
    },
    PMUMT_HASHTAG_MENTION_URL: {
        "label": "#, @, URL",
        "description": "Hashtags, @-mentions, or URLs.",
    },
    PMUMT_WRONG_GENDER_NUMBER: {
        "label": "Wrong gender/grammatical number",
        "description": "Non-standard gender or grammatical number agreement.",
    },
    PMUMT_INCONSISTENT_CASING: {
        "label": "Inconsistent casing",
        "description": "Non-standard or inconsistent capitalisation.",
    },
    PMUMT_EMOJI: {
        "label": "Emoji",
        "description": "Emoji or emoticon-like symbols.",
    },
    PMUMT_NAMED_ENTITY: {
        "label": "Named Entity",
        "description": "Named entities.",
    },
    PMUMT_CONTRACTION: {
        "label": "Contraction",
        "description": "Informal or non-standard contractions.",
    },
    PMUMT_GRAPHEMIC_PUNCTUATION_STRETCHING: {
        "label": "Graphemic/punctuation stretching",
        "description": "Letter or punctuation repetition for emphasis.",
    },
    PMUMT_INTERJECTIONS: {
        "label": "Interjections",
        "description": "Interjections or discourse markers typical of informal writing.",
    },
}

PMUMT_TO_UGC_TAXONOMY = {
    PMUMT_LETTER_DELETION_ADDITION: [SPELLING],
    PMUMT_MISSING_DIACRITICS: [SPELLING],
    PMUMT_PHONETIC_WRITING: [
        GRAMMAR,
        INFORMAL_ABBREVIATIONS,
    ],
    PMUMT_TOKENISATION_ERROR: [SPELLING],
    PMUMT_WRONG_VERB_TENSE: [GRAMMAR],
    PMUMT_HASHTAG_MENTION_URL: [
        HASHTAGS_SUBREDDITS,
        URLS_MENTIONS_RTS,
    ],
    PMUMT_WRONG_GENDER_NUMBER: [GRAMMAR],
    PMUMT_INCONSISTENT_CASING: [CAPITALISATION],
    PMUMT_EMOJI: [EMOTICONS_EMOJIS],
    PMUMT_NAMED_ENTITY: [],
    PMUMT_CONTRACTION: [INFORMAL_ABBREVIATIONS],
    PMUMT_GRAPHEMIC_PUNCTUATION_STRETCHING: [
        WORD_ELONGATION,
        ATYPICAL_PUNCTUATION,
    ],
    PMUMT_INTERJECTIONS: [
        INFORMAL_ABBREVIATIONS,
        INFORMAL_ACRONYMS,
    ],
}

# RoCS-MT taxonomy constants

ROCSMT_ERROR = "ERROR"

ROCSMT_ABBREVIATION = "abbreviation"
ROCSMT_ACRONYMISATION = "acronymisation"
ROCSMT_ARTICLE_ADD = "article_add"
ROCSMT_ARTICLE_DROP = "article_drop"
ROCSMT_ASTERISKS = "asterisks"
ROCSMT_CAMELCASE = "camelcase"
ROCSMT_CAPITALISATION = "capitalisation"
ROCSMT_CENSURE = "censure"
ROCSMT_CONTRACTION = "contraction"
ROCSMT_CUTE = "cute"
ROCSMT_DEVOWELLING = "devowelling"
ROCSMT_DIALECTISM = "dialectism"
ROCSMT_DIGIT_LETTER_SIM = "digit_letter_sim"
ROCSMT_DIGITS_TO_WORDS = "digits_to_words"
ROCSMT_DIMINUTIVE = "dimunitive"
ROCSMT_DOUBLE_TO_SINGLE = "double_to_single_character"
ROCSMT_ELONGATION = "elongation"
ROCSMT_EMOTICON = "emoticon"
ROCSMT_GRAMMAR = "grammar"
ROCSMT_INFLECTION = "inflection"
ROCSMT_INTERJECTION = "interjection"
ROCSMT_LEX_CHOICE = "lex_choice"
ROCSMT_MIMIC_SPOKEN = "mimic_spoken"

ROCSMT_NORM_PUNCT = "norm_punct"  # unify variants

ROCSMT_PHONETIC_DISTANCE = "phonetic_distance"
ROCSMT_PRONOUN_DROP = "pronoun_drop"
ROCSMT_PUNCT_DIFF = "punct_diff"
ROCSMT_SCRAMBLED = "scrambled"
ROCSMT_SLASH_TO_AND = "slash_to_and"
ROCSMT_SLASH_TO_OR = "slash_to_or"
ROCSMT_SOUND = "sound"
ROCSMT_SPACING = "spacing"

ROCSMT_SPELLING = "spelling_error"  # unify both variants

ROCSMT_SURROUNDING_EMPHASIS = "surrounding_emphasis"
ROCSMT_SYMBOL_ADD = "symbol_add"
ROCSMT_SYMBOL_DROP = "symbol_drop"
ROCSMT_SYMBOL_PLACEMENT = "symbol_placement"
ROCSMT_TRUNCATION = "truncation"
ROCSMT_WORD_ADD = "word_add"
ROCSMT_WORD_DROP = "word_drop"
ROCSMT_WORD_ORDER = "word_order"
ROCSMT_WORD_TO_SYMBOL = "word_to_symbol"
ROCSMT_WORDS_TO_DIGITS = "words_to_digits"

ROCSMT_TO_UGC_TAXONOMY = {

    # --- Grammar ---
    ROCSMT_GRAMMAR: [GRAMMAR],
    ROCSMT_INFLECTION: [GRAMMAR],
    ROCSMT_ARTICLE_ADD: [GRAMMAR],
    ROCSMT_ARTICLE_DROP: [GRAMMAR],
    ROCSMT_PRONOUN_DROP: [GRAMMAR],
    ROCSMT_WORD_ORDER: [GRAMMAR],
    ROCSMT_WORD_ADD: [GRAMMAR],
    ROCSMT_WORD_DROP: [GRAMMAR],

    # --- Spelling / orthography ---
    ROCSMT_SPELLING: [SPELLING],
    ROCSMT_SCRAMBLED: [SPELLING],
    ROCSMT_SPACING: [SPELLING],

    # --- Word elongation ---
    ROCSMT_ELONGATION: [WORD_ELONGATION],

    # --- Capitalisation ---
    ROCSMT_CAPITALISATION: [CAPITALISATION],
    ROCSMT_CAMELCASE: [CAPITALISATION],

    # --- Informal abbreviations ---
    ROCSMT_ABBREVIATION: [INFORMAL_ABBREVIATIONS],
    ROCSMT_CONTRACTION: [INFORMAL_ABBREVIATIONS],
    ROCSMT_DEVOWELLING: [INFORMAL_ABBREVIATIONS],
    ROCSMT_DIGIT_LETTER_SIM: [INFORMAL_ABBREVIATIONS],
    ROCSMT_DOUBLE_TO_SINGLE: [INFORMAL_ABBREVIATIONS],
    ROCSMT_PHONETIC_DISTANCE: [INFORMAL_ABBREVIATIONS],
    ROCSMT_TRUNCATION: [INFORMAL_ABBREVIATIONS],

    # --- Informal acronyms ---
    ROCSMT_ACRONYMISATION: [INFORMAL_ACRONYMS],

    # --- Emoticons ---
    ROCSMT_EMOTICON: [EMOTICONS_EMOJIS],

    # --- Punctuation ---
    ROCSMT_NORM_PUNCT: [ATYPICAL_PUNCTUATION],
    ROCSMT_PUNCT_DIFF: [ATYPICAL_PUNCTUATION],
    ROCSMT_ASTERISKS: [ATYPICAL_PUNCTUATION],
    ROCSMT_SURROUNDING_EMPHASIS: [ATYPICAL_PUNCTUATION],

    # --- Profanity ---
    ROCSMT_CENSURE: [SELF_CENSORED_PROFANITY],

    # --- Informal language / noise ---
    ROCSMT_INTERJECTION: [INFORMAL_ABBREVIATIONS],
    ROCSMT_MIMIC_SPOKEN: [INFORMAL_ABBREVIATIONS],
    ROCSMT_SOUND: [INFORMAL_ABBREVIATIONS],
    ROCSMT_DIALECTISM: [INFORMAL_ABBREVIATIONS],
    ROCSMT_CUTE: [INFORMAL_ABBREVIATIONS],
    ROCSMT_DIMINUTIVE: [INFORMAL_ABBREVIATIONS],

    # --- Symbol / numeric transformations ---
    ROCSMT_WORD_TO_SYMBOL: [INFORMAL_ABBREVIATIONS],
    ROCSMT_SYMBOL_ADD: [ATYPICAL_PUNCTUATION],
    ROCSMT_SYMBOL_DROP: [ATYPICAL_PUNCTUATION],
    ROCSMT_SYMBOL_PLACEMENT: [ATYPICAL_PUNCTUATION],
    ROCSMT_WORDS_TO_DIGITS: [INFORMAL_ABBREVIATIONS],
    ROCSMT_DIGITS_TO_WORDS: [INFORMAL_ABBREVIATIONS],

    # --- Slash constructions ---
    ROCSMT_SLASH_TO_AND: [INFORMAL_ABBREVIATIONS],
    ROCSMT_SLASH_TO_OR: [INFORMAL_ABBREVIATIONS],

    # --- Lexical ---
    ROCSMT_LEX_CHOICE: [GRAMMAR, INFORMAL_ABBREVIATIONS],

    # --- Catch-all ---
    ROCSMT_ERROR: [],
}