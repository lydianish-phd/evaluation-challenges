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