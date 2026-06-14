import json
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── ANSI colors ───────────────────────────────────────────────
class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    PURPLE  = "\033[38;5;135m"
    CYAN    = "\033[38;5;81m"
    GREEN   = "\033[38;5;83m"
    YELLOW  = "\033[38;5;220m"
    RED     = "\033[38;5;203m"
    WHITE   = "\033[97m"
    BG_DARK = "\033[48;5;236m"


# ── Helpers ───────────────────────────────────────────────────
def banner():
    print(f"""
{C.PURPLE}{C.BOLD}
  ╔══════════════════════════════════════════╗
  ║       NLP Customer Service Chatbot       ║
  ║        TF-IDF · Cosine Similarity        ║
  ╚══════════════════════════════════════════╝
{C.RESET}""")


def divider():
    print(f"{C.DIM}  {'─' * 44}{C.RESET}")


def confidence_badge(score: float) -> str:
    if score >= 0.6:
        return f"{C.GREEN}[HIGH  {score:.0%}]{C.RESET}"
    elif score >= 0.3:
        return f"{C.YELLOW}[MED   {score:.0%}]{C.RESET}"
    else:
        return f"{C.RED}[LOW   {score:.0%}]{C.RESET}"


def print_user(text: str):
    print(f"\n  {C.CYAN}{C.BOLD}You ›{C.RESET}  {C.WHITE}{text}{C.RESET}")


def print_bot(text: str, score: float):
    badge = confidence_badge(score)
    print(f"  {C.PURPLE}{C.BOLD}Bot ›{C.RESET}  {text}")
    print(f"         {C.DIM}Confidence: {C.RESET}{badge}")


def print_error(text: str):
    print(f"  {C.RED}{C.BOLD}Bot ›{C.RESET}  {C.DIM}{text}{C.RESET}")


def typing_effect(text: str, delay: float = 0.018):
    """Simulate typing for the bot reply."""
    import sys
    print(f"  {C.PURPLE}{C.BOLD}Bot ›{C.RESET}  ", end="", flush=True)
    for ch in text:
        sys.stdout.write(ch)
        sys.stdout.flush()
        time.sleep(delay)
    print()


# ── Load data ─────────────────────────────────────────────────
def load_intents(path: str = "intents.json"):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        if "questions" not in data or "answers" not in data:
            raise KeyError("intents.json must have 'questions' and 'answers' keys.")
        if len(data["questions"]) != len(data["answers"]):
            raise ValueError("questions and answers must have the same length.")
        return data["questions"], data["answers"]
    except FileNotFoundError:
        print(f"{C.RED}Error:{C.RESET} intents.json not found.")
        raise SystemExit(1)
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"{C.RED}Error:{C.RESET} {e}")
        raise SystemExit(1)


# ── NLP engine ────────────────────────────────────────────────
class ChatBot:
    THRESHOLD = 0.30

    def __init__(self, questions: list, answers: list):
        self.answers   = answers
        self.questions = questions
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),   # unigrams + bigrams for better matching
            stop_words="english",
            min_df=1,
        )
        self.q_vectors = self.vectorizer.fit_transform(questions)

    def reply(self, user_input: str) -> tuple[str, float]:
        """Return (answer, confidence_score)."""
        vec        = self.vectorizer.transform([user_input])
        scores     = cosine_similarity(vec, self.q_vectors)[0]
        best_idx   = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        if best_score >= self.THRESHOLD:
            return self.answers[best_idx], best_score
        return None, best_score


# ── Main loop ─────────────────────────────────────────────────
def main():
    banner()

    questions, answers = load_intents()
    bot = ChatBot(questions, answers)

    print(f"  {C.DIM}Loaded {len(questions)} intents  ·  Type {C.RESET}"
          f"{C.BOLD}'exit'{C.RESET}{C.DIM} or {C.RESET}"
          f"{C.BOLD}'quit'{C.RESET}{C.DIM} to stop{C.RESET}\n")
    divider()

    while True:
        try:
            user_input = input(f"\n  {C.CYAN}{C.BOLD}You ›{C.RESET}  ").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n  {C.PURPLE}Bot › {C.RESET}Goodbye! 👋\n")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "bye"}:
            print(f"\n  {C.PURPLE}{C.BOLD}Bot ›{C.RESET}  Goodbye! 👋\n")
            break

        answer, score = bot.reply(user_input)

        print()
        if answer:
            typing_effect(answer)
            badge = confidence_badge(score)
            print(f"         {C.DIM}Confidence: {C.RESET}{badge}")
        else:
            typing_effect("Sorry, I don't understand. Could you rephrase?")
            badge = confidence_badge(score)
            print(f"         {C.DIM}Confidence: {C.RESET}{badge}")

        divider()


if __name__ == "__main__":
    main()