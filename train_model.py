import os, json, gzip
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import joblib

SEED = 42
np.random.seed(SEED)

CATEGORIES = ["anxiety", "sadness", "anger", "stress", "sleep", "relationships", "work", "confusion", "positive", "general"]

# Core training phrases per category
PHRASES = {
    "anxiety": [
        "i feel anxious", "my heart is racing", "i keep worrying", "panic attacks",
        "i feel nervous", "i can't stop overthinking", "my chest feels tight",
        "i have this constant dread", "afraid something bad will happen", "social situations make me anxious",
        "i'm terrified of failing", "i can't breathe when anxious", "on high alert",
        "i'm afraid to leave my house", "i feel paralyzed by fear", "i keep catastrophizing",
        "my stomach is in knots", "i obsess over my health", "i'm scared of being judged",
        "i worry i'm not good enough", "i'm scared of making mistakes", "i avoid people because of anxiety",
        "my mind races at night", "i'm afraid of the future", "i overthink every conversation",
        "i feel panicked for no reason", "i can't relax", "dating gives me anxiety",
        "i feel like everyone is watching me", "i have a constant knot in my stomach",
        "i can't stop imagining worst case scenarios", "i feel trapped by anxious thoughts",
        "every little noise makes me jump", "i feel dizzy from anxiety", "i'm afraid of public speaking",
        "i worry about things that haven't happened", "i have trouble breathing when anxious",
        "i feel restless and on edge", "i'm scared of my own feelings", "i obsess over what people think",
    ],
    "sadness": [
        "i feel empty inside", "nothing makes me happy", "i've been crying a lot", "i feel so alone",
        "i don't see the point", "i feel hopeless", "i miss how things used to be",
        "i feel like a burden", "everything feels pointless", "i've lost interest in everything",
        "i feel like giving up", "i'm tired of feeling sad", "i feel broken beyond repair",
        "i feel disconnected from everyone", "nobody cares about me", "i've never felt this low",
        "i feel like i'm losing myself", "i'm not worthy of love", "i feel a deep sadness",
        "i don't know how to be happy", "i feel like i let everyone down", "i can't escape this sadness",
        "i'm grieving someone", "i feel guilty for being sad", "i feel numb",
        "i miss being happy so much", "i feel like crying but can't", "i'm tired of pretending",
        "i feel like my life has no meaning", "i feel like i'm drowning in sorrow",
        "i feel like i'll never be happy again", "i feel invisible", "i'm mourning the life i thought i'd have",
        "i feel like my soul is empty", "i've been isolating myself", "i feel like i don't belong",
    ],
    "anger": [
        "i'm so angry i could scream", "everything makes me furious", "i feel like punching something",
        "i can't control my temper", "i'm mad at the world", "rage building inside me",
        "i'm irritated by everyone", "i can't let go of anger", "i feel like i'll explode",
        "i'm so angry i can't think straight", "i feel resentful", "i keep getting into arguments",
        "i'm furious about how i've been treated", "i feel betrayed", "i'm tired of being taken for granted",
        "i'm angry all the time", "i feel treated unfairly", "i can't stop thinking about what made me angry",
        "i feel like punching a wall", "i'm so mad i can't sleep", "i feel like screaming",
        "my anger controls me", "i keep snapping at people", "i hate feeling this angry",
        "i'm so frustrated", "i can't stop feeling bitter", "i'm angry at myself",
        "i can't forgive what they did", "i'm tired of being walked all over",
        "i feel disrespected every day", "small things set me off", "i've been holding in anger for years",
        "i feel like my anger is eating me alive", "i'm furious they lied", "i feel enraged thinking about it",
    ],
    "stress": [
        "i'm so stressed i can't function", "too much to handle", "i'm overwhelmed",
        "i can't keep up with responsibilities", "drowning in work", "under so much pressure",
        "i can't catch a break", "exhausted from stress", "carrying the world on my shoulders",
        "i can't handle everything", "i'm going to crash", "stressed about money",
        "no time for myself", "so busy i can't breathe", "always behind",
        "i can't relax even with free time", "spreading myself too thin", "stressed about deadlines",
        "no control over my life", "i'm burning out", "tired from all this pressure",
        "overwhelmed by responsibilities", "stressed about family", "running on empty",
        "struggling to balance everything", "i'm about to break", "stressed about the future",
        "too many decisions", "constantly rushing", "too many people depending on me",
        "stretched too thin to function", "worrying about all my responsibilities",
        "stress making me physically sick", "overwhelmed by partner's expectations",
        "one step away from a breakdown", "can't keep up with expectations",
        "i feel like i'm failing at everything", "tired of being pulled in every direction",
    ],
    "sleep": [
        "i can't fall asleep", "waking up in the middle of the night", "haven't slept well in weeks",
        "exhausted but can't sleep", "terrible nightmares", "lie awake for hours",
        "tired from lack of sleep", "can't shut off my brain", "waking up more tired",
        "i've been having insomnia", "can't sleep because worrying", "restless nights",
        "tired no matter how much i sleep", "wake up multiple times", "trouble falling asleep every night",
        "haven't had a good night's rest", "wake up with anxiety", "stuck in a cycle of bad sleep",
        "dread going to bed", "groggy and tired all day", "need sleeping pills",
        "wake up at 3am", "toss and turn all night", "mind is racing at bedtime",
        "feel like i haven't slept", "body won't let me sleep", "wake up with headaches from poor sleep",
        "afraid to sleep because of nightmares", "sleepwalking through life",
        "sleep too much and still tired", "sleep schedule is broken", "grind teeth at night from stress",
        "chronic insomnia affecting everything", "sleep anxiety keeps me awake",
        "vivid nightmares", "can't nap even when exhausted", "wake up gasping for air",
        "stay up late even though i need rest",
    ],
    "relationships": [
        "problems with my partner", "relationship is falling apart", "don't feel loved",
        "we keep fighting", "can't talk to my partner", "partner losing interest",
        "we don't communicate", "only one trying", "partner doesn't understand me",
        "trapped in my relationship", "scared of losing partner", "we've grown apart",
        "relationship is one-sided", "partner won't listen", "same arguments over and over",
        "can't trust my partner", "friends don't care", "distant from family",
        "walking on eggshells", "partner doesn't appreciate me", "lonely in my relationship",
        "nothing in common anymore", "partner always criticizing", "partner hiding things",
        "can't be myself around partner", "relationship causing stress", "partner doesn't support me",
        "taken for granted", "don't feel heard", "partner and i want different things",
        "caught partner cheating", "going through a breakup", "getting divorced",
        "manipulated in relationship", "gaslit by partner", "family doesn't approve of partner",
        "trust issues from past", "rejected by someone i love", "heartbroken after breakup",
        "scared to open up after being hurt", "partner invalidates my feelings",
        "feeling controlled", "lonely even with partner", "friends drifting away",
    ],
    "work": [
        "i hate my job", "stuck in my career", "boss always criticizing",
        "not satisfied with my job", "no future at work", "considering quitting",
        "can't stand coworkers", "not valued at work", "bored with my job",
        "not progressing in career", "dread going to work", "job is making me miserable",
        "underpaid and overworked", "don't know what career path", "wasting my potential",
        "scared of losing my job", "can't handle work pressure", "work has no meaning",
        "problems with colleagues", "need a career change", "work-life balance is terrible",
        "not good enough at my job", "passed over for promotions", "can't concentrate at work",
        "in the wrong profession", "workplace is toxic", "not respected at work",
        "scared to look for new job", "hit a dead end in career", "job causing too much stress",
        "don't get along with manager", "can't grow in current role", "frustrated with career trajectory",
        "burned out from work", "imposter syndrome at work", "laid off and feel lost",
        "lost my job and don't know what to do", "no passion for work",
        "workplace full of bullying", "anxious every sunday night before work",
        "working overtime every day", "underqualified and terrified of failing",
        "career has no direction", "embarrassed about being unemployed",
        "never find a job i enjoy",
    ],
    "confusion": [
        "don't know what to do with my life", "feel lost and confused", "don't know who i am",
        "unsure about my decisions", "can't figure out what i want", "feel directionless",
        "don't know what path to take", "confused about my feelings", "don't understand myself",
        "at a crossroads", "unsure about everything", "can't make up my mind",
        "feel like i don't know anything", "confused about what to do", "don't know what's right",
        "feel like i'm in a fog", "can't think clearly", "torn between two choices",
        "don't know what i believe", "questioning everything", "confused about my identity",
        "don't know what matters", "no direction", "unsure about my future",
        "can't figure out what's important", "don't know myself", "confused about my purpose",
        "don't know what i'm supposed to do", "struggling to find my way",
        "going in circles", "can't decide what i want", "don't know how to move forward",
        "lost in my own life", "not sure what i believe", "can't figure out what makes me happy",
        "wandering without purpose", "confused about my goals", "don't know what success means",
        "starting over and don't know how", "questioning every decision",
        "confused about sexuality", "struggling with faith", "having an identity crisis",
        "unsure about my values", "don't know if i should stay or leave",
    ],
    "positive": [
        "feel really happy today", "grateful for everything", "feel hopeful about the future",
        "had a really good day", "proud of what i accomplished", "excited about what's coming",
        "feel at peace", "thankful for friends and family", "things are getting better",
        "feeling optimistic", "feel loved and supported", "happy with how things are going",
        "feel strong and capable", "proud of myself for getting through this",
        "joyful for no particular reason", "grateful to be alive", "growing as a person",
        "excited about my future", "content with my life", "happy about my progress",
        "more confident lately", "thankful for good things", "feel blessed",
        "proud of how far i've come", "feel peaceful and calm", "excited to start new chapter",
        "everything falling into place", "grateful for my health", "feel accomplished",
        "looking forward to tomorrow", "feel inspired and motivated", "happy with who i'm becoming",
        "connected to people around me", "grateful for second chances", "feel like i can handle anything",
        "proud of my personal growth", "surrounded by love", "excited about new opportunities",
        "exactly where i need to be", "thankful for beautiful day", "making a difference",
        "proud of the person i'm becoming", "hopeful and optimistic", "grateful for each new day",
        "inspired to pursue my dreams", "deeply connected to people i love",
        "sense of peace i haven't felt in years", "proud of myself for asking for help",
        "grateful for my support system", "excited about possibilities",
        "deep sense of gratitude", "proud that i'm healing", "joyful and lighthearted",
        "finally becoming myself",
    ],
    "general": [
        "don't know where to start", "just wanted to talk to someone", "been thinking about things",
        "don't really know what to say", "can we just talk", "not sure why i'm here",
        "been reflecting on my life", "need someone to talk to", "want to share something",
        "been meaning to talk about this", "don't understand my feelings right now",
        "just need to vent", "want to get something off my chest", "been thinking about the past",
        "trying to process some things", "feel like talking but don't know about what",
        "need some advice", "been doing a lot of thinking", "want to understand myself better",
        "trying to figure things out", "just need someone to listen", "something i've been wanting to say",
        "been going through a lot", "feel like sharing something personal",
        "working on understanding my emotions", "want to talk about what's been happening",
        "trying to be more open about feelings", "feel like i need perspective",
        "been noticing patterns in my behavior", "want to work on myself",
        "here because i want to change", "need help understanding something",
        "been journaling about my thoughts", "want to be more self aware",
        "trying to make sense of my life", "want to check in with myself",
        "here to reflect on my week", "want to share thoughts with someone safe",
        "need clarity on things", "feel like i need to talk through something",
        "trying to be honest about feelings", "want to learn how to cope better",
        "here to work on my mental health", "been wanting to reach out for help",
        "trying to process something that happened", "need help sorting out my thoughts",
        "want to be a better version of myself", "here to talk through a decision",
        "want to share what's been on my mind",
    ],
}

INTENT_MAP = {
    "greeting": "general", "small_talk": "general", "introduce_name": "general",
    "humor": "positive", "sarcasm": "general", "joke": "positive",
    "relationship_problem": "relationships", "breakup": "relationships",
    "loneliness": "sadness", "academic_stress": "stress", "exam_anxiety": "anxiety",
    "career_stress": "stress", "family_conflict": "relationships",
    "friendship_issue": "relationships", "financial_stress": "stress",
    "grief": "sadness", "loss": "sadness", "identity_question": "confusion",
    "depression_like": "sadness", "anxiety_like": "anxiety", "panic": "anxiety",
    "gratitude": "positive", "burnout": "stress", "anger_expression": "anger",
    "regret": "sadness", "guilt": "sadness", "excitement": "positive",
    "hope": "positive",
}

EMOTION_MAP = {
    "anxiety": "anxiety", "panic": "anxiety", "fear": "anxiety",
    "sadness": "sadness", "grief": "sadness", "loneliness": "sadness",
    "anger": "anger", "frustration": "anger", "rage": "anger",
    "stress": "stress", "overwhelm": "stress", "burnout": "stress",
    "sleep": "sleep", "insomnia": "sleep",
    "relationships": "relationships", "love": "relationships",
    "work": "work", "career": "work", "job": "work",
    "confusion": "confusion", "uncertainty": "confusion",
    "gratitude": "positive", "joy": "positive", "hope": "positive",
}


def build_synthetic_df():
    rows = [(text, cat) for cat, texts in PHRASES.items() for text in texts]
    df = pd.DataFrame(rows, columns=["text", "label"])
    print(f"  Synthetic: {len(df)} samples")
    return df


def build_jsonl_df():
    dataset_dir = os.path.join(os.path.dirname(__file__), "dataset")
    files = [
        os.path.join(dataset_dir, "sage_massive_training.jsonl.gz"),
        os.path.join(dataset_dir, "sage_training.jsonl"),
    ]
    rows = []
    for path in files:
        if not os.path.exists(path):
            continue
        opener = gzip.open if path.endswith(".gz") else open
        mode = "rt" if path.endswith(".gz") else "r"
        with opener(path, mode, encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line.strip()) if line.strip() else {}
                text = rec.get("user_message", "").strip()
                intent = rec.get("detected_intent", "") or rec.get("intent", "")
                if text and intent:
                    rows.append({"text": text, "label": INTENT_MAP.get(intent, "general")})
        print(f"  Loaded {len(rows):,} from {os.path.basename(path)}")
    if not rows:
        return None
    df = pd.DataFrame(rows)
    print(f"  JSONL: {len(df):,} samples")
    return df


def build_kb_df():
    kb_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    if not os.path.isdir(kb_dir):
        return None
    rows = []
    for fname in os.listdir(kb_dir):
        fpath = os.path.join(kb_dir, fname)
        if fname.endswith(".jsonl"):
            with open(fpath, encoding="utf-8") as f:
                for line in f:
                    e = json.loads(line.strip()) if line.strip() else {}
                    text, emos = e.get("user", ""), e.get("emotions", [])
                    for emo in emos:
                        rows.append({"text": text, "label": EMOTION_MAP.get(emo.lower(), "general")})
        elif fname.endswith(".json"):
            with open(fpath, encoding="utf-8") as f:
                data = json.load(f)
            for key in data:
                items = data[key] if isinstance(data[key], list) else []
                for item in items:
                    if isinstance(item, dict):
                        for subkey in ["example_phrases", "common_thoughts", "typical_thoughts",
                                       "warning_phrases", "user_message", "phrase"]:
                            for t in item.get(subkey, []):
                                if isinstance(t, str) and len(t) > 5:
                                    emo = item.get("emotion", item.get("likely_emotion", ""))
                                    rows.append({"text": t, "label": EMOTION_MAP.get(emo.lower() if emo else "", "general")})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    print(f"  KB: {len(df)} samples")
    return df


def balance(df, per_class=80):
    parts = []
    for cat in df["label"].unique():
        sub = df[df["label"] == cat]
        n = min(len(sub), per_class)
        parts.append(sub.sample(n=n, replace=len(sub) < per_class, random_state=SEED))
    balanced = pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"  Balanced: {len(balanced)} samples")
    return balanced


def train(X_train, y_train):
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 3), min_df=2,
                                  max_df=0.85, sublinear_tf=True, stop_words="english")),
        ("clf", LogisticRegression(solver="lbfgs", max_iter=1000, random_state=SEED,
                                   C=1.5, class_weight="balanced")),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def main():
    print("=" * 50, "\nCOUNSELOR MODEL TRAINING\n", "=" * 50)

    print("\n[1/4] Building synthetic dataset...")
    syn = build_synthetic_df()

    print("\n[2/4] Loading JSONL dataset...")
    jsonl = build_jsonl_df()

    print("\n[3/4] Loading knowledge base...")
    kb = build_kb_df()

    print("\n[4/4] Training...")
    dfs = [d for d in [syn, jsonl, kb] if d is not None]
    combined = pd.concat(dfs, ignore_index=True)
    balanced = balance(combined, per_class=max(int(len(combined) / 10 * 1.2), 80))

    X = balanced["text"].to_numpy(dtype=str)
    y = balanced["label"].to_numpy(dtype=str)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)

    model = train(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    print(f"\n  Accuracy: {acc:.3f}")
    print(classification_report(y_test, y_pred, zero_division=0))

    out = os.path.join(os.path.dirname(__file__), "counselor_model.joblib")
    joblib.dump(model, out)
    print(f"  Model saved: {out}")


if __name__ == "__main__":
    main()
