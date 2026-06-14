import json
import time
import threading
import numpy as np
import tkinter as tk
from tkinter import font as tkfont
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ── Load & validate intents ───────────────────────────────────
def load_intents(path="intents.json"):
    try:
        with open(path, "r") as f:
            data = json.load(f)
        assert len(data["questions"]) == len(data["answers"]), \
            "questions and answers length mismatch"
        return data["questions"], data["answers"]
    except FileNotFoundError:
        raise SystemExit(f"Error: '{path}' not found.")
    except (KeyError, AssertionError, json.JSONDecodeError) as e:
        raise SystemExit(f"Error in intents.json: {e}")


# ── NLP engine ────────────────────────────────────────────────
class ChatBot:
    THRESHOLD = 0.30

    def __init__(self, questions, answers):
        self.answers = answers
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
        self.q_vectors  = self.vectorizer.fit_transform(questions)

    def reply(self, text):
        vec   = self.vectorizer.transform([text])
        sims  = cosine_similarity(vec, self.q_vectors)[0]
        idx   = int(np.argmax(sims))
        score = float(sims[idx])
        if score >= self.THRESHOLD:
            return self.answers[idx], score, "high" if score >= 0.6 else "medium"
        return "Sorry, I didn't understand that. Could you rephrase?", score, "low"


# ── Theme ─────────────────────────────────────────────────────
BG        = "#1E1E2E"
SURFACE   = "#2A2A3E"
SURFACE2  = "#313147"
PURPLE    = "#7C6FF7"
PURPLE_DK = "#5A54C4"
CYAN      = "#56CFE1"
GREEN     = "#6EEB83"
YELLOW    = "#F7C948"
RED       = "#F46B6B"
TEXT_1    = "#EAEAF5"
TEXT_2    = "#9090B0"
TEXT_3    = "#5C5C7A"
RADIUS    = 12


# ── Main App ──────────────────────────────────────────────────
class ChatApp(tk.Tk):
    def __init__(self, bot: ChatBot):
        super().__init__()
        self.bot = bot
        self.title("AI Customer Support")
        self.geometry("520x680")
        self.minsize(420, 560)
        self.configure(bg=BG)
        self._build_fonts()
        self._build_ui()
        self._welcome()

    # ── Fonts ─────────────────────────────────────────────────
    def _build_fonts(self):
        self.f_title  = tkfont.Font(family="Segoe UI", size=13, weight="bold")
        self.f_sub    = tkfont.Font(family="Segoe UI", size=9)
        self.f_bubble = tkfont.Font(family="Segoe UI", size=10)
        self.f_meta   = tkfont.Font(family="Segoe UI", size=8)
        self.f_input  = tkfont.Font(family="Segoe UI", size=10)
        self.f_btn    = tkfont.Font(family="Segoe UI", size=10, weight="bold")

    # ── UI layout ─────────────────────────────────────────────
    def _build_ui(self):
        # Header
        hdr = tk.Frame(self, bg=SURFACE, pady=12)
        hdr.pack(fill=tk.X)

        dot_canvas = tk.Canvas(hdr, width=10, height=10, bg=SURFACE,
                               highlightthickness=0)
        dot_canvas.pack(side=tk.LEFT, padx=(16, 6))
        dot_canvas.create_oval(1, 1, 9, 9, fill=GREEN, outline="")

        tk.Label(hdr, text="Support Assistant", font=self.f_title,
                 bg=SURFACE, fg=TEXT_1).pack(side=tk.LEFT)
        tk.Label(hdr, text="NLP · TF-IDF", font=self.f_sub,
                 bg=SURFACE, fg=TEXT_2).pack(side=tk.LEFT, padx=8)

        # Separator
        tk.Frame(self, bg=TEXT_3, height=1).pack(fill=tk.X)

        # Chat canvas + scrollbar
        chat_frame = tk.Frame(self, bg=BG)
        chat_frame.pack(fill=tk.BOTH, expand=True, padx=0, pady=0)

        self.scrollbar = tk.Scrollbar(chat_frame, bg=SURFACE, troughcolor=BG,
                                      bd=0, width=6, relief=tk.FLAT)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y, padx=(0, 4), pady=8)

        self.chat_box = tk.Text(
            chat_frame,
            bg=BG, fg=TEXT_1,
            font=self.f_bubble,
            relief=tk.FLAT,
            bd=0,
            padx=16, pady=12,
            wrap=tk.WORD,
            cursor="arrow",
            state=tk.DISABLED,
            yscrollcommand=self.scrollbar.set,
            spacing3=4,
        )
        self.chat_box.pack(fill=tk.BOTH, expand=True)
        self.scrollbar.config(command=self.chat_box.yview)

        # Tags for styling
        self.chat_box.tag_config("user_bubble", foreground=TEXT_1,
                                 background=PURPLE, relief=tk.FLAT,
                                 lmargin1=90, lmargin2=90, rmargin=16,
                                 spacing1=6, spacing3=2)
        self.chat_box.tag_config("bot_bubble", foreground=TEXT_1,
                                 background=SURFACE2,
                                 lmargin1=16, lmargin2=16, rmargin=90,
                                 spacing1=6, spacing3=2)
        self.chat_box.tag_config("user_label", foreground=PURPLE,
                                 font=self.f_meta,
                                 lmargin1=90, rmargin=16, spacing1=10)
        self.chat_box.tag_config("bot_label", foreground=TEXT_2,
                                 font=self.f_meta,
                                 lmargin1=16, spacing1=10)
        self.chat_box.tag_config("conf_high",   foreground=GREEN,  font=self.f_meta, lmargin1=16)
        self.chat_box.tag_config("conf_medium", foreground=YELLOW, font=self.f_meta, lmargin1=16)
        self.chat_box.tag_config("conf_low",    foreground=RED,    font=self.f_meta, lmargin1=16)
        self.chat_box.tag_config("divider", foreground=TEXT_3,
                                 font=self.f_meta, justify=tk.CENTER, spacing1=4)

        # Suggestion chips
        chips_frame = tk.Frame(self, bg=BG)
        chips_frame.pack(fill=tk.X, padx=16, pady=(4, 0))
        tk.Label(chips_frame, text="Quick questions:", font=self.f_meta,
                 bg=BG, fg=TEXT_2).pack(anchor=tk.W)

        chips_row = tk.Frame(chips_frame, bg=BG)
        chips_row.pack(fill=tk.X, pady=4)
        chips = ["Working hours", "Refund policy", "Track order",
                 "Payment methods", "Return product"]
        for c in chips:
            tk.Button(
                chips_row, text=c, font=self.f_meta,
                bg=SURFACE2, fg=TEXT_2, relief=tk.FLAT,
                padx=10, pady=4, cursor="hand2",
                activebackground=PURPLE, activeforeground=TEXT_1,
                command=lambda t=c: self._chip_click(t)
            ).pack(side=tk.LEFT, padx=(0, 6))

        # Separator
        tk.Frame(self, bg=TEXT_3, height=1).pack(fill=tk.X, pady=(8, 0))

        # Input area
        input_frame = tk.Frame(self, bg=SURFACE, pady=10, padx=12)
        input_frame.pack(fill=tk.X)

        self.entry = tk.Entry(
            input_frame,
            font=self.f_input,
            bg=SURFACE2, fg=TEXT_1,
            insertbackground=CYAN,
            relief=tk.FLAT, bd=0,
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True,
                        ipady=8, padx=(0, 10))
        self.entry.bind("<Return>", lambda e: self._send())
        self.entry.bind("<FocusIn>",  lambda e: self.entry.config(bg="#3A3A55"))
        self.entry.bind("<FocusOut>", lambda e: self.entry.config(bg=SURFACE2))

        self.send_btn = tk.Button(
            input_frame,
            text="Send  ›",
            font=self.f_btn,
            bg=PURPLE, fg=TEXT_1,
            activebackground=PURPLE_DK,
            activeforeground=TEXT_1,
            relief=tk.FLAT, bd=0,
            padx=16, pady=6,
            cursor="hand2",
            command=self._send,
        )
        self.send_btn.pack(side=tk.RIGHT)

        self.entry.focus()

    # ── Insert text helpers ───────────────────────────────────
    def _insert(self, text, tag=None):
        self.chat_box.config(state=tk.NORMAL)
        if tag:
            self.chat_box.insert(tk.END, text, tag)
        else:
            self.chat_box.insert(tk.END, text)
        self.chat_box.config(state=tk.DISABLED)
        self.chat_box.see(tk.END)

    def _welcome(self):
        self._insert("  Support Assistant  ", "bot_label")
        self._insert("\n")
        self._insert("  👋  Welcome! How can I help you today?  \n", "bot_bubble")
        self._insert("\n")

    # ── Send logic ────────────────────────────────────────────
    def _send(self):
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        self.send_btn.config(state=tk.DISABLED)

        # User bubble
        self._insert("  You  ", "user_label")
        self._insert("\n")
        self._insert(f"  {text}  \n", "user_bubble")
        self._insert("\n")

        # Show typing in a thread to keep UI responsive
        threading.Thread(target=self._bot_reply, args=(text,), daemon=True).start()

    def _bot_reply(self, text):
        # Simulate thinking delay
        time.sleep(0.5)

        answer, score, level = self.bot.reply(text)
        pct = f"{score:.0%}"
        icons = {"high": "●", "medium": "◑", "low": "○"}
        conf_labels = {"high": f"  {icons['high']} High confidence · {pct}  ",
                       "medium": f"  {icons['medium']} Medium confidence · {pct}  ",
                       "low": f"  {icons['low']} Low confidence · {pct}  "}

        self.after(0, self._insert, "  Bot  ", "bot_label")
        self.after(0, self._insert, "\n")
        self.after(0, self._insert, f"  {answer}  \n", "bot_bubble")
        self.after(0, self._insert, conf_labels[level] + "\n", f"conf_{level}")
        self.after(0, self._insert, "\n")
        self.after(0, self.send_btn.config, {"state": tk.NORMAL})
        self.after(0, self.entry.focus)

    def _chip_click(self, text):
        self.entry.delete(0, tk.END)
        self.entry.insert(0, text)
        self._send()


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    questions, answers = load_intents("intents.json")
    bot = ChatBot(questions, answers)
    app = ChatApp(bot)
    app.mainloop()