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

# Natural, conversational training phrases - shorter and more realistic
PHRASES = {
    "anxiety": [
        "i feel anxious", "my heart is racing", "i keep worrying", "panic attacks",
        "i feel nervous", "i can't stop overthinking", "my chest feels tight",
        "i have constant dread", "afraid something bad will happen", "social situations scare me",
        "terrified of failing", "can't breathe when anxious", "on high alert all the time",
        "afraid to leave my house", "feel paralyzed by fear", "stomach in knots from worry",
        "obsess over my health", "scared of being judged", "worry im not good enough",
        "scared of making mistakes", "avoid people because of anxiety", "mind races at night",
        "afraid of the future", "overthink every conversation", "panicked for no reason",
        "cant relax at all", "dating gives me anxiety", "everyone is watching me",
        "cant stop imagining worst case", "trapped by anxious thoughts", "every noise makes me jump",
        "dizzy from anxiety", "afraid of public speaking", "worry about things that havent happened",
        "trouble breathing when anxious", "restless and on edge", "obsess over what people think",
        "anxious all the time", "heart pounds when phone rings", "cant stop worrying",
        "knot in my stomach", "scared of my own feelings", "avoid checking my email",
        "feel like im losing my mind", "cant stop catastrophizing", "afraid to speak up",
        "constant feeling of dread", "terrified of being alone", "nervous about everything",
        "scared of having another panic attack", "cant think clearly when anxious",
    ],
    "sadness": [
        "i feel empty", "nothing makes me happy", "ive been crying", "i feel so alone",
        "i dont see the point", "i feel hopeless", "i miss how things used to be",
        "i feel like a burden", "everything feels pointless", "lost interest in everything",
        "i feel like giving up", "tired of feeling sad", "feel broken beyond repair",
        "disconnected from everyone", "nobody cares about me", "never felt this low",
        "losing myself", "not worthy of love", "deep sadness",
        "dont know how to be happy", "i let everyone down", "cant escape this sadness",
        "grieving someone", "guilty for being sad", "i feel numb",
        "miss being happy", "feel like crying but cant", "tired of pretending to be okay",
        "life has no meaning", "drowning in sorrow", "ill never be happy again",
        "i feel invisible", "mourning the life i thought id have", "soul is empty",
        "isolating myself", "dont belong anywhere", "feeling down",
        "feeling low", "feeling blue", "feeling awful", "feeling bad", "feeling terrible",
        "i feel so tired", "cant go on like this", "nobody would notice if i disappeared",
        "heartbroken", "feeling miserable", "so sad right now",
    ],
    "anger": [
        "im so angry", "everything makes me furious", "feel like punching something",
        "cant control my temper", "mad at the world", "rage building inside me",
        "irritated by everyone", "cant let go of anger", "feel like ill explode",
        "so angry i cant think", "feel resentful", "keep getting into arguments",
        "furious about how ive been treated", "feel betrayed", "tired of being taken for granted",
        "angry all the time", "treated unfairly", "cant stop thinking about what made me angry",
        "feel like punching a wall", "so mad i cant sleep", "feel like screaming",
        "anger controls me", "keep snapping at people", "hate feeling this angry",
        "so frustrated", "cant stop feeling bitter", "angry at myself",
        "cant forgive what they did", "tired of being walked all over",
        "disrespected every day", "small things set me off", "holding in anger for years",
        "anger eating me alive", "furious they lied", "enraged thinking about it",
        "pissed off", "fed up with everything", "sick of being treated like this",
    ],
    "stress": [
        "im so stressed", "too much to handle", "im overwhelmed",
        "cant keep up", "drowning in work", "under so much pressure",
        "cant catch a break", "exhausted from stress", "carrying the world",
        "cant handle everything", "im going to crash", "stressed about money",
        "no time for myself", "so busy i cant breathe", "always behind",
        "cant relax even with free time", "spreading myself too thin", "stressed about deadlines",
        "no control over my life", "im burning out", "tired from all this pressure",
        "overwhelmed by responsibilities", "stressed about family", "running on empty",
        "struggling to balance", "im about to break", "stressed about the future",
        "too many decisions", "constantly rushing", "too many people depending on me",
        "stretched too thin", "worrying about responsibilities", "stress making me sick",
        "overwhelmed by expectations", "one step from breakdown", "cant keep up with expectations",
        "failing at everything", "pulled in every direction", "feeling crushed",
    ],
    "sleep": [
        "cant fall asleep", "waking up in the middle of the night", "havent slept well",
        "exhausted but cant sleep", "terrible nightmares", "lie awake for hours",
        "tired from lack of sleep", "cant shut off my brain", "waking up more tired",
        "having insomnia", "cant sleep because worrying", "restless nights",
        "tired no matter how much i sleep", "wake up multiple times", "trouble falling asleep",
        "havent had a good nights rest", "wake up with anxiety", "stuck in bad sleep cycle",
        "dread going to bed", "groggy all day", "need sleeping pills",
        "wake up at 3am", "toss and turn all night", "mind racing at bedtime",
        "feel like i havent slept", "body wont let me sleep", "headaches from poor sleep",
        "afraid to sleep because of nightmares", "sleepwalking through life",
        "sleep too much and still tired", "sleep schedule is broken", "grind teeth at night",
        "chronic insomnia", "sleep anxiety", "vivid nightmares",
        "cant nap even when exhausted", "wake up gasping", "stay up late even though i need rest",
    ],
    "relationships": [
        "problems with my partner", "relationship is falling apart", "dont feel loved",
        "we keep fighting", "cant talk to my partner", "partner losing interest",
        "we dont communicate", "only one trying", "partner doesnt understand me",
        "trapped in my relationship", "scared of losing partner", "grown apart",
        "relationship is one-sided", "partner wont listen", "same arguments over and over",
        "cant trust my partner", "friends dont care", "distant from family",
        "walking on eggshells", "partner doesnt appreciate me", "lonely in relationship",
        "nothing in common anymore", "partner always criticizing", "partner hiding things",
        "cant be myself around partner", "relationship causing stress", "partner doesnt support me",
        "taken for granted", "dont feel heard", "want different things",
        "caught partner cheating", "going through a breakup", "getting divorced",
        "manipulated in relationship", "gaslit by partner", "family doesnt approve",
        "trust issues from past", "rejected by someone i love", "heartbroken after breakup",
        "scared to open up", "partner invalidates my feelings", "feeling controlled",
        "lonely even with partner", "friends drifting away", "miss my ex",
    ],
    "work": [
        "i hate my job", "stuck in my career", "boss always criticizing",
        "not satisfied with my job", "no future at work", "considering quitting",
        "cant stand coworkers", "not valued at work", "bored with my job",
        "not progressing", "dread going to work", "job is making me miserable",
        "underpaid and overworked", "dont know what career path", "wasting my potential",
        "scared of losing my job", "cant handle work pressure", "work has no meaning",
        "problems with colleagues", "need a career change", "work-life balance is terrible",
        "not good enough at my job", "passed over for promotions", "cant concentrate at work",
        "wrong profession", "workplace is toxic", "not respected at work",
        "scared to look for new job", "dead end in career", "job causing too much stress",
        "dont get along with manager", "cant grow in current role", "frustrated with career",
        "burned out from work", "imposter syndrome", "laid off and feel lost",
        "lost my job", "no passion for work", "workplace bullying",
        "anxious before work", "working overtime every day", "underqualified and terrified",
        "career has no direction", "embarrassed about being unemployed",
    ],
    "confusion": [
        "dont know what to do", "feel lost", "dont know who i am",
        "unsure about my decisions", "cant figure out what i want", "feel directionless",
        "dont know what path to take", "confused about my feelings", "dont understand myself",
        "at a crossroads", "unsure about everything", "cant make up my mind",
        "feel like i dont know anything", "confused about what to do", "dont know whats right",
        "feel like im in a fog", "cant think clearly", "torn between two choices",
        "dont know what i believe", "questioning everything", "confused about my identity",
        "dont know what matters", "no direction", "unsure about my future",
        "cant figure out whats important", "dont know myself", "confused about my purpose",
        "dont know what im supposed to do", "struggling to find my way",
        "going in circles", "cant decide", "dont know how to move forward",
        "lost in my own life", "not sure what i believe", "cant figure out what makes me happy",
        "wandering without purpose", "confused about my goals", "dont know what success means",
        "starting over and dont know how", "questioning every decision",
        "having an identity crisis", "unsure about my values", "dont know if i should stay or leave",
    ],
    "positive": [
        "i feel great", "i feel good", "im happy", "feeling happy today",
        "feel really happy", "grateful for everything", "feel hopeful",
        "had a really good day", "proud of what i accomplished", "excited about whats coming",
        "feel at peace", "thankful for friends", "things are getting better",
        "feeling optimistic", "feel loved", "happy with how things are going",
        "feel strong", "proud of myself", "joyful today",
        "grateful to be alive", "growing as a person", "excited about my future",
        "content with my life", "happy about my progress", "more confident lately",
        "thankful for good things", "feel blessed", "proud of how far ive come",
        "feel peaceful", "excited to start new chapter", "everything falling into place",
        "grateful for my health", "feel accomplished", "looking forward to tomorrow",
        "feel inspired", "happy with who im becoming", "connected to people",
        "grateful for second chances", "feel like i can handle anything", "proud of my growth",
        "surrounded by love", "excited about new opportunities", "exactly where i need to be",
        "thankful for today", "making a difference", "hopeful and optimistic",
        "grateful for each day", "inspired to pursue my dreams", "deeply connected to people i love",
        "proud of myself for asking for help", "grateful for support", "excited about possibilities",
        "feeling amazing", "feeling wonderful", "feeling fantastic", "life is good",
        "today was great", "im doing well", "feeling blessed", "feeling content",
    ],
    "general": [
        "dont know where to start", "just wanted to talk", "been thinking about things",
        "dont really know what to say", "can we just talk", "not sure why im here",
        "been reflecting on my life", "need someone to talk to", "want to share something",
        "been meaning to talk about this", "dont understand my feelings",
        "just need to vent", "want to get something off my chest", "been thinking about the past",
        "trying to process some things", "feel like talking", "need some advice",
        "been doing a lot of thinking", "want to understand myself", "trying to figure things out",
        "just need someone to listen", "something ive been wanting to say", "been going through a lot",
        "feel like sharing something", "working on understanding my emotions",
        "want to talk about whats been happening", "trying to be more open",
        "feel like i need perspective", "been noticing patterns", "want to work on myself",
        "here because i want to change", "need help understanding something",
        "been journaling", "want to be more self aware", "trying to make sense of my life",
        "want to check in with myself", "here to reflect", "want to share thoughts",
        "need clarity", "feel like i need to talk", "trying to be honest about feelings",
        "want to learn how to cope", "here to work on my mental health", "been wanting to reach out",
        "trying to process something", "need help sorting out thoughts",
        "want to be a better version of myself", "here to talk through a decision",
        "whats on my mind", "i need to talk about something",
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
    "hope": "positive", "self_harm_concern": "anxiety", "violence_disclosure": "anxiety",
    "abuse_disclosure": "anxiety", "trauma_discussion": "anxiety",
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
    "excitement": "positive", "pride": "positive", "contentment": "positive",
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
