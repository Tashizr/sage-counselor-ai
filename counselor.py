import random, re, os, json
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
    (r"\bkilled\s+(a\s+)?(someone|my|him|her|them)\b", "violence"),
    (r"\bmurder\w*\s+(a\s+)?(someone|my|him|her|them)\b", "violence"),
    (r"\brobbed\s+(a\s+)?(bank|store|someone)\b", "crime"),
    (r"\bstole?\s+(a\s+)?(car|money|something)\b", "crime"),
    (r"\bkidnapped\b", "violence"),
    (r"\bshoot\w*\s+(up|someone|a\s+place)\b", "violence"),
    (r"\b(hacked|hack)\s+(into|a\s+system)\b", "crime"),
]

# Pandas-driven response templates
TOPICS = ["anxiety", "sadness", "anger", "stress", "sleep", "relationships", "work", "confusion", "positive", "general"]

_response_data = pd.DataFrame({
    "topic": TOPICS,
    "reflection": [
        ["It sounds like there's a lot of weight in what you're describing.", "I wonder if this has been sitting with you for a while.", "That worry sounds really present for you right now.", "It seems like your mind is working hard to keep you alert."],
        ["It sounds like there's a heaviness that's hard to put into words.", "I wonder if this has been weighing on you more than you let on.", "It seems like there's a deep ache in what you're sharing.", "That kind of sadness sounds like it runs deep."],
        ["It sounds like something really got to you.", "I wonder if there's more underneath that frustration.", "It seems like this touched something important.", "That kind of anger usually points to something that matters."],
        ["It sounds like you're carrying a lot right now.", "I wonder if this has been building for a while.", "It seems like there's a lot pulling at your attention.", "That kind of pressure sounds exhausting."],
        ["It sounds like your mind doesn't quiet down when you need it to.", "I wonder if there's something your mind is trying to process.", "It seems like rest has been hard to come by.", "That struggle to sleep sounds really draining."],
        ["It sounds like there's something shifting in a relationship that matters to you.", "I wonder if this has been on your mind for a while.", "It seems like this connection means a lot to you.", "That kind of tension is hard to carry."],
        ["It sounds like work has been taking a lot out of you.", "I wonder if this has been building up over time.", "It seems like there's a mismatch between what you give and what you get.", "That kind of work stress can affect everything else too."],
        ["It sounds like you're at a place where things aren't clear yet.", "I wonder if this uncertainty has been unsettling.", "It seems like you're weighing something important.", "That kind of not-knowing is genuinely uncomfortable."],
        ["It sounds like something good is happening for you.", "I wonder if this has been a long time coming.", "It seems like things are moving in a good direction.", "That kind of positive energy is really nice to hear."],
        ["It sounds like there's something on your mind.", "I wonder if this has been sitting with you for a while.", "It seems like this is important to you.", "I appreciate you sharing that with me."],
    ],
    "validation": [
        ["It makes sense you'd feel that way.", "Anxiety is your body's way of trying to protect you, even when it overdoes it."],
        ["It makes sense you'd feel that way.", "Sadness is a natural response when something matters to us."],
        ["Anyone might feel that way in your position.", "Anger often shows us when something we care about has been crossed."],
        ["Anyone would feel stretched under that load.", "It makes sense to feel overwhelmed with everything on your plate."],
        ["It makes sense that would leave you feeling drained.", "Sleep struggles often reflect how full our minds are during the day."],
        ["It makes sense that would hurt.", "Relationships touch such a deep part of us."],
        ["It makes sense that would weigh on you.", "So much of our identity gets tied up in work."],
        ["It makes sense to feel uncertain right now.", "Confusion often means something important is shifting."],
        ["That's really nice to hear.", "It's good you're noticing that."],
        [""],
    ],
    "question": [
        ["What does that anxiety feel like in your body right now?", "When does that worry tend to show up most?", "What's the first thing that comes to mind when you think about what's worrying you?"],
        ["When did that heavy feeling first start showing up?", "What does that sadness need you to understand about it?", "Is there something specific that triggered this, or has it been building?"],
        ["What happened that sparked this frustration?", "What does this anger want you to protect?", "What would need to happen for you to feel heard about this?"],
        ["What part of this feels the heaviest right now?", "If you could set one thing down today, what would it be?", "What's pulling at your attention most urgently?"],
        ["What's going through your mind when you're lying awake?", "How long has this pattern been going on?", "Is it hard to fall asleep, or hard to stay asleep?"],
        ["What do you find yourself needing that you're not getting right now?", "What kind of conversation have you been wanting to have?", "How long has this dynamic been building?"],
        ["What part of your work drains you the most?", "What would a better version of your work life look like?", "Is it the work itself, or the environment around it?"],
        ["What feels most unclear when you sit with it?", "What's the heart of what you're trying to figure out?", "What options are you weighing right now?"],
        ["What's contributing to that good feeling?", "What helped bring this about?", "How does it feel in your body when you experience this?"],
        ["What's coming up for you as you share that?", "What feels most important about this to you?", "What made you decide to bring this up today?"],
    ],
    "soft_ending": [
        ["I'm here with you.", "We can sit with this for a while.", "Take your time."],
        ["I'm here with you.", "We can stay with this for a bit.", "Take your time."],
        ["I'm here with you.", "Let that sit for a moment.", "Take your time."],
        ["We can slow down here.", "Take your time.", "I'm here with you."],
        ["Take your time.", "We can let that sit.", "I'm here with you."],
        ["I'm here with you.", "Take your time with this.", "We can sit with this."],
        ["Take your time.", "I'm here with you.", "We can slow down here."],
        ["Take your time.", "We can sit with the question.", "I'm here with you."],
        ["That's really nice.", "I'm glad for you.", "That's good to hear."],
        ["Take your time.", "I'm here with you.", "We can sit with this."],
    ],
})

EMOTION_WORDS = {
    "anxiety": ["anxious", "worried", "nervous", "panic", "scared", "afraid", "dread"],
    "sadness": ["sad", "sorrow", "grief", "depressed", "lonely", "empty"],
    "anger": ["angry", "mad", "furious", "annoyed", "irritated"],
    "stress": ["stressed", "overwhelmed", "burned out", "pressure"],
    "confusion": ["confused", "lost", "unsure", "torn"],
}

NAME_PATTERNS = [
    r"my\s+name(?:'s|\s+is)\s+(.+)",
    r"i(?:'|'?\s*|\s+)m\s+(.+)",
    r"i\s+am\s+(.+)",
    r"call\s+me\s+(.+)",
    r"you\s+can\s+call\s+me\s+(.+)",
    r"the\s+name(?:'s|\s+is)\s+(.+)",
    r"it(?:'s|\s+is)\s+(.+)",
]

EMOTION_SKIP = {
    "sad", "happy", "anxious", "angry", "stressed", "scared", "nervous",
    "worried", "depressed", "lonely", "tired", "exhausted", "confused",
    "lost", "empty", "hopeless", "hurt", "broken", "numb", "fine",
    "okay", "good", "bad", "great", "terrible", "feeling", "feel",
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
            return "I'm here. You can take your time."
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
            if any(w in low for w in {"nah", "no", "fictional", "hypothetical", "story", "pretend", "made up"}):
                return "I appreciate you clarifying. What's actually on your mind?"
            if any(w in low for w in {"real", "yes", "actually", "happened", "serious"}) and len(low.split()) < 8:
                return f"Thank you for being honest with me.\n\n{self._generate(orig)}"
            return self._generate(text)

        for pat, cat in AMBIGUOUS:
            if re.search(pat, text.lower()):
                self.awaiting = text
                start = random.choice([
                    "Can you help me understand what you mean by that?",
                    "I want to make sure I understand you correctly.",
                ])
                follow = "Are you describing something that actually happened, or something else?"
                return f"{start} {follow}"

        if self.user_name is None:
            name = self._extract_name(text)
            if name:
                self.user_name = name
                return f"Hi, {name}! It's nice to meet you. I'm SAGE. What's been on your mind?"
            words = text.split()
            if len(words) <= 3 and not self._is_counseling(text):
                self.user_name = words[0].strip(",.!?")
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
        return [t for t, words in EMOTION_WORDS.items() if any(w in low for w in words)][:2]

    def _key_phrases(self, text):
        stop = {"i", "me", "my", "we", "you", "he", "she", "it", "a", "an", "the", "and", "or", "but",
                "is", "am", "are", "was", "have", "has", "do", "not", "no", "so", "if", "all", "just",
                "can", "will", "to", "in", "of", "for", "on", "at", "by", "from", "with", "about"}
        words = [w.strip(",.!?;:'\"") for w in text.lower().split()
                 if w.strip(",.!?;:'\"") not in stop and len(w.strip(",.!?;:'\"")) > 2]
        random.shuffle(words)
        return words[:3]

    def _generate(self, text):
        translated = self._translate_slang(text)
        topic, _ = self._predict(translated)
        row = _response_data[_response_data["topic"] == topic].iloc[0]

        reflection = random.choice(row["reflection"])
        key_words = self._key_phrases(translated)
        if key_words and topic != "general":
            reflection += f" Especially with {random.choice(key_words)} on your mind."

        validation = random.choice(row["validation"])
        question = random.choice(row["question"])

        if self._is_short(text) and len(self.history) >= 3:
            for prev in reversed(self.history[:-1]):
                if len(prev.split()) > 3:
                    prev_topic, _ = self._predict(prev)
                    prev_row = _response_data[_response_data["topic"] == prev_topic].iloc[0]
                    ref = random.choice(prev_row["reflection"])
                    end = random.choice(prev_row["soft_ending"])
                    return f"{ref} You don't have to find the perfect words. Take your time."
            return f"It's okay not to know exactly what to say. {random.choice(row['soft_ending'])}"

        parts = [reflection]
        if validation:
            parts.append(validation)

        secondary = self._detect_emotions(translated)
        if secondary and secondary[0] != topic and random.random() < 0.3:
            parts.append("And it sounds like there's some of that too.")

        if random.random() < 0.25:
            parts.append(random.choice(row["soft_ending"]))
        else:
            parts.append(question)

        return " ".join(parts)

    def goodbye(self):
        return (f"{self.user_name or 'Friend'}, thank you for talking with me. "
                "If things get difficult, please reach out to a counselor or crisis line. Take care.")


if __name__ == "__main__":
    import sys, time
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
