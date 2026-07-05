import random, re, os
import numpy as np
import pandas as pd
import joblib

CRISIS_KW = [
    "suicide", "kill myself", "end my life", "want to die", "self-harm",
    "hurt myself", "not worth living", "better off dead", "end it all",
    "take my own life", "don't want to live", "wish i was dead",
    "end my suffering", "never wake up", "harm myself", "cut myself", "suicidal",
]

SLANG = {
    "fr": "for real", "ngl": "not going to lie", "lowkey": "somewhat",
    "rn": "right now", "idk": "i don't know", "imo": "in my opinion",
    "nah": "no", "bet": "okay", "no cap": "no lie", "deadass": "seriously",
    "im cooked": "i'm in trouble", "ghosted": "stopped contacting",
    "left on read": "didn't reply", "spiraling": "losing emotional control",
    "bro": "friend", "bruh": "friend", "lol": "laughing", "lmao": "laughing",
}

SHORT_ANSWERS = {"yeah", "nah", "idk", "maybe", "sure", "kinda", "ok", "okay", "ye", "ig", "i guess", "not sure"}

AMBIGUOUS = [
    (r"\bkill(s|ed)?\s+(a\s+)?(someone|somebody|my|him|her|them|a\s+guy|a\s+man|a\s+woman)\b", "violence"),
    (r"\bmurder(s|ed|ing)?\s+(a\s+)?(someone|somebody|my|him|her|them)\b", "violence"),
    (r"\brob(s|bed|bing)?\s+(a\s+)?(bank|store|someone|somebody)\b", "crime"),
    (r"\bstole?\s+(a\s+)?(car|money|something)\b", "crime"),
    (r"\bkidnap(ped|ping)?\b", "violence"),
    (r"\bshoot(s|ing)?\s+(up|someone|a\s+place|a\s+school)\b", "violence"),
    (r"\b(hack(s|ed|ing)?)\s+(into|a\s+system|their\s+account)\b", "crime"),
    (r"\b(fight|riot|revolution)\b", "violence"),
    (r"\bburn(s|ed|ing)?\s+(down\s+)?(a\s+)?(house|building|car)\b", "violence"),
    (r"\b(rape|raped|raping)\b", "violence"),
    (r"\b(torture|tortured|torturing)\b", "violence"),
    (r"\b(bomb|bombed|bombing)\s+(a|the|this|their)\b", "violence"),
]

TOPICS = ["anxiety", "sadness", "anger", "stress", "sleep", "relationships", "work", "confusion", "positive", "general"]

_response_data = pd.DataFrame({
    "topic": TOPICS,
    # Shorter, more natural reflections
    "reflection": [
        ["That sounds like a lot to carry.", "I hear you. That kind of worry can be exhausting.", "That anxiety sounds really real.", "It makes sense you'd feel on edge."],
        ["That sounds heavy.", "I'm sorry you're feeling this way.", "Sadness like that doesn't just go away.", "That takes a lot of courage to say."],
        ["Something clearly got under your skin.", "That frustration sounds valid.", "I hear the anger in what you're saying.", "That's a real feeling."],
        ["You're juggling a lot right now.", "That pressure sounds overwhelming.", "It's okay to feel stretched thin.", "That's a lot on your plate."],
        ["Sleep struggles are the worst.", "That sounds really draining.", "I hear you. Restless nights take a toll.", "That's hard to deal with night after night."],
        ["That sounds painful.", "Relationship stuff cuts deep.", "I hear you. That tension is real.", "That kind of hurt lingers."],
        ["Work can really wear you down.", "That sounds frustrating.", "I hear you. That's a tough spot.", "That mismatch hurts."],
        ["It's okay to not have all the answers.", "That uncertainty is uncomfortable.", "I hear you. Being stuck is hard.", "That's a tough place to be."],
        ["That's really nice to hear.", "I'm glad you're feeling that way.", "That's wonderful.", "Good for you. You deserve that."],
        ["I hear you.", "Thanks for sharing that.", "I'm here.", "Tell me more."],
    ],
    # Shorter validations
    "validation": [
        ["That makes total sense.", "Anxiety can be really overwhelming."],
        ["Of course you'd feel that way.", "That kind of sadness is real."],
        ["Anyone would feel that way.", "That anger makes sense."],
        ["That's a lot for anyone.", "No wonder you feel stretched."],
        ["That would drain anyone.", "Of course you're tired of it."],
        ["That makes sense.", "Of course that hurts."],
        ["That would frustrate anyone.", "No wonder you feel that way."],
        ["Totally understandable.", "That kind of uncertainty is tough."],
        ["Love hearing that.", "You should feel good about that."],
        ["", ""],
    ],
    # Shorter, more conversational questions
    "question": [
        ["What does that feel like for you?", "When does it hit you the hardest?"],
        ["When did this start?", "What's weighing on you most?"],
        ["What set this off?", "What would help you feel heard?"],
        ["What's the hardest part?", "If you could drop one thing, what would it be?"],
        ["What's keeping you up?", "How long has this been going on?"],
        ["What do you need right now?", "How long has this been building?"],
        ["What's draining you most?", "What would change look like?"],
        ["What feels unclear?", "What's the core of it for you?"],
        ["What's making you feel good?", "What brought this on?"],
        ["What's on your mind?", "What made you reach out?"],
    ],
    # Shorter soft endings
    "soft_ending": [
        ["I'm here with you.", "Take your time."],
        ["I'm here with you.", "No rush."],
        ["I'm right here.", "Take your time."],
        ["No rush.", "I'm here."],
        ["Take your time.", "I'm listening."],
        ["I'm here.", "No pressure."],
        ["Take your time.", "I'm here."],
        ["No rush.", "I'm listening."],
        ["That's great.", "Glad to hear it."],
        ["I'm here.", "Take your time."],
    ],
})

EMOTION_WORDS = {
    "anxiety": ["anxious", "worried", "nervous", "panic", "scared", "afraid", "dread", "tense", "on edge"],
    "sadness": ["sad", "sorrow", "grief", "depressed", "lonely", "empty", "crying", "tears", "down", "blue", "low", "awful", "bad", "miserable"],
    "anger": ["angry", "mad", "furious", "annoyed", "irritated", "rage", "pissed"],
    "stress": ["stressed", "overwhelmed", "burned out", "pressure", "exhausted", "swamped"],
    "confusion": ["confused", "lost", "unsure", "torn"],
    "positive": ["great", "good", "happy", "amazing", "wonderful", "grateful", "thankful", "excited", "proud", "blessed", "hopeful", "optimistic", "content", "peaceful", "joyful"],
}

# Mood check-in phrases map to topics directly
MOOD_MAP = {
    "feeling great": "positive", "feeling good": "positive", "feeling happy": "positive",
    "feeling amazing": "positive", "feeling wonderful": "positive",
    "feeling okay": "general", "feeling alright": "general",
    "feeling low": "sadness", "feeling bad": "sadness", "feeling awful": "sadness",
    "feeling terrible": "sadness", "feeling down": "sadness", "feeling blue": "sadness",
}

NAME_PATTERNS = [
    r"my\s+name(?:'s|\s+is)\s+(.+)",
    r"i(?:'|'?\s*)\s*m\s+(.+)",
    r"i\s+am\s+(.+)",
    r"call\s+me\s+(.+)",
    r"you\s+can\s+call\s+me\s+(.+)",
    r"the\s+name(?:'s|\s+is)\s+(.+)",
    r"it(?:'s|\s+is)\s+(.+)",
    r"sup\s+im\s+(.+)",
    r"hey\s+im\s+(.+)",
    r"hi\s+im\s+(.+)",
    r"hello\s+im\s+(.+)",
    r"yo\s+im\s+(.+)",
    r"im\s+([a-z]{2,})\b",
]

EMOTION_SKIP = {
    "sad", "happy", "anxious", "angry", "stressed", "scared", "nervous",
    "worried", "depressed", "lonely", "tired", "exhausted", "confused",
    "lost", "empty", "hopeless", "hurt", "broken", "numb", "fine",
    "okay", "good", "bad", "great", "terrible", "feeling", "feel",
    "awesome", "amazing", "wonderful", "fantastic", "pretty", "really",
}

STOP_WORDS = {
    "i", "me", "my", "we", "you", "he", "she", "it", "a", "an", "the",
    "and", "or", "but", "is", "am", "are", "was", "have", "has", "do",
    "not", "no", "so", "if", "all", "just", "can", "will", "to", "in",
    "of", "for", "on", "at", "by", "from", "with", "about", "been",
    "being", "be", "that", "this", "these", "those", "what", "when",
    "where", "how", "why", "which", "who", "whom", "whose",
    "very", "too", "also", "just", "really", "quite", "pretty",
    "don't", "dont", "cant", "won't", "wont", "isn't", "isnt",
    "wasn't", "wasnt", "aren't", "arent", "didn't", "didnt",
    "i'm", "im", "you're", "youre", "he's", "hes", "she's", "shes",
    "it's", "its", "we're", "were", "they're", "theyre",
    "feeling", "feel", "felt", "like", "want", "need", "know",
    "think", "think", "going", "come", "make", "made",
}


def detect_crisis(text):
    return any(kw in text.lower() for kw in CRISIS_KW)


class Counselor:
    def __init__(self):
        self.history = []
        self.user_name = None
        self.awaiting = None
        self.model = None
        path = os.path.join(os.path.dirname(__file__), "counselor_model.joblib")
        if os.path.exists(path):
            self.model = joblib.load(path)
            self.classes_ = self.model.named_steps["clf"].classes_

    def greet(self):
        return ("Hi, I'm here to listen. I help people talk through what's on their mind. "
                "I'm not a replacement for a licensed therapist. What's your name?")

    def respond(self, text):
        text = text.strip()
        if not text:
            self.history.append(text)
            return "I'm here whenever you're ready."
        self.history.append(text)

        if detect_crisis(text):
            self.awaiting = None
            return ("I'm really concerned about what you're saying. Please reach out now:\n"
                    "- National Suicide Prevention Lifeline: 988\n"
                    "- Crisis Text Line: Text HOME to 741741\nThere are people ready to help you.")

        if self.awaiting is not None:
            orig = self.awaiting
            self.awaiting = None
            low = text.lower()
            if any(w in low for w in {"nah", "no", "fictional", "hypothetical", "story", "pretend", "made up", "joking", "joke", "not real", "in a game", "game", "roleplay", "rp"}):
                return f"Got it, no worries. {random.choice(['What else is on your mind?', 'What would you like to talk about?', 'What brings you here today?'])}"
            if any(w in low for w in {"real", "yes", "actually", "happened", "serious", "fr", "deadass"}) and len(low.split()) < 8:
                return f"Thank you for being honest with me.\n\n{self._generate(orig)}"
            return self._generate(text)

        for pat, cat in AMBIGUOUS:
            if re.search(pat, text.lower()):
                self.awaiting = text
                return random.choice([
                    "Can you help me understand what you mean by that?",
                    "I want to make sure I understand you correctly. Are you describing something real, or something else?",
                    "Before I respond — can you clarify what you're sharing?",
                ])

        if self.user_name is None:
            name = self._extract_name(text)
            if name:
                self.user_name = name
                return f"Hi, {name}! It's nice to meet you. I'm SAGE. What's been on your mind?"
            words = text.split()
            # If it's short and doesn't look like counseling, treat first word as name
            if len(words) <= 4 and not self._is_counseling(text):
                candidate = words[-1].strip(",.!?")
                if len(candidate) >= 2 and candidate.lower() not in EMOTION_SKIP:
                    self.user_name = candidate.title()
                    return f"Hi, {self.user_name}! It's nice to meet you. I'm SAGE. What would you like to talk about?"
            self.user_name = "Friend"

        return self._generate(text)

    def _extract_name(self, text):
        low = text.lower().strip()
        for pat in NAME_PATTERNS:
            m = re.match(pat, low)
            if m:
                name = self._clean_name(m.group(1))
                if name:
                    return name
        return None

    def _clean_name(self, name):
        name = name.strip().rstrip(".!?")
        words = [w for w in name.split() if w.lower() not in EMOTION_SKIP]
        if not words or (len(words) == 1 and len(words[0]) < 2) or all(w.isdigit() for w in words):
            return None
        return " ".join(words).title()

    def _is_counseling(self, text):
        words = text.lower().split()
        if not words:
            return False
        if words[0] in {"i", "me", "my", "we", "you", "he", "she", "it"}:
            return True
        if self.model is not None:
            _, conf = self._predict(text)
            return conf > 0.25
        return False

    def _predict(self, text):
        if self.model is None:
            return "general", 0.0
        df = pd.DataFrame({"text": [text]})
        probs = self.model.predict_proba(df["text"])
        idx = np.argmax(probs, axis=1)[0]
        return self.classes_[idx], float(np.max(probs, axis=1)[0])

    def _predict_with_mood_override(self, text):
        low = text.lower()
        for phrase, topic in MOOD_MAP.items():
            if phrase in low:
                return topic, 0.95
        return self._predict(text)

    def _translate_slang(self, text):
        out = text.lower()
        for phrase, meaning in sorted(SLANG.items(), key=lambda x: -len(x[0])):
            out = out.replace(phrase, meaning)
        return out

    def _is_short(self, text):
        low = text.lower().strip(",.!? ")
        return low in SHORT_ANSWERS or (len(low.split()) <= 2 and low not in {"i don't know", "not sure"})

    def _detect_emotions(self, text):
        low = text.lower()
        found = []
        for t, words in EMOTION_WORDS.items():
            if any(w in low for w in words):
                found.append(t)
        return found[:2]

    def _key_phrases(self, text):
        generic = {"game", "things", "thing", "stuff", "something", "everything",
                    "nothing", "anything", "someone", "everyone", "somewhere",
                    "person", "people", "place", "time", "day", "night", "today",
                    "yesterday", "tomorrow", "week", "month", "year", "life",
                    "world", "lot", "bit", "way", "kind", "sort", "type",
                    "happened", "happening", "going", "done", "said", "told",
                    "talk", "share", "tell", "speak", "say", "feel", "think",
                    "know", "want", "need", "like", "love", "hate"}
        words = [w.strip(",.!?;:'\"") for w in text.lower().split()
                 if w.strip(",.!?;:'\"") not in STOP_WORDS
                 and w.strip(",.!?;:'\"") not in generic
                 and len(w.strip(",.!?;:'\"")) > 2
                 and w.strip(",.!?;:'\"").lower() != (self.user_name or "").lower()]
        random.shuffle(words)
        return words[:2]

    def _maybe_add_name(self, sentence):
        """Optionally prepend the user's name to a sentence for a personal touch."""
        if not self.user_name or self.user_name == "Friend":
            return sentence
        if random.random() > 0.4:
            return sentence
        prefix = random.choice([f"{self.user_name}, ", f"Hey {self.user_name}, "])
        return prefix + sentence[0].lower() + sentence[1:]

    def _generate(self, text):
        translated = self._translate_slang(text)
        topic, _ = self._predict_with_mood_override(translated)
        row = _response_data[_response_data["topic"] == topic].iloc[0]

        reflection = self._maybe_add_name(random.choice(row["reflection"]))

        # Only add key phrase if it's meaningful (not a stop word)
        key_words = self._key_phrases(translated)
        if key_words and topic not in ("general", "positive"):
            kw = random.choice(key_words)
            templates = [
                f"Especially with {kw}.",
                f"Especially around {kw}.",
                f"When it comes to {kw}, that's real.",
            ]
            reflection += " " + random.choice(templates)

        validation = random.choice(row["validation"])
        question = random.choice(row["question"])

        # Short answer handling
        if self._is_short(text) and len(self.history) >= 3:
            for prev in reversed(self.history[:-1]):
                if len(prev.split()) > 3:
                    prev_topic, _ = self._predict_with_mood_override(prev)
                    prev_row = _response_data[_response_data["topic"] == prev_topic].iloc[0]
                    return f"{self._maybe_add_name(random.choice(prev_row['reflection']))} Take your time."
            return f"It's okay not to know what to say. {random.choice(row['soft_ending'])}"

        # Build response - sometimes shorter, sometimes fuller
        roll = random.random()
        if roll < 0.2:
            return f"{reflection} {random.choice(row['soft_ending'])}"
        elif roll < 0.5:
            return f"{reflection} {question}"
        elif roll < 0.7:
            parts = [reflection]
            if validation:
                parts.append(validation)
            parts.append(question)
            return " ".join(parts)
        else:
            parts = [reflection]
            if validation:
                parts.append(validation)
            secondary = self._detect_emotions(translated)
            if secondary and secondary[0] != topic and random.random() < 0.3:
                parts.append("There might be some of that mixed in too.")
            parts.append(question)
            return " ".join(parts)

    def goodbye(self):
        return (f"{self.user_name or 'Friend'}, thank you for talking with me. "
                "If things get difficult, please reach out to a counselor or crisis line. Take care.")


if __name__ == "__main__":
    import sys
    bot = Counselor()
    print(bot.greet(), "\n")
    while True:
        try:
            u = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if u.lower() in ("quit", "exit", "bye", "goodbye"):
            print(bot.goodbye())
            break
        print(f"Bot: {bot.respond(u)}\n")
    sys.exit(0)
