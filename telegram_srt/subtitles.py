"""Subtitle layout using recognition timestamps, never estimated reading times."""
from dataclasses import dataclass
import math
import re
import textwrap


LAYOUTS = {"1": (1,), "2": (2,), "3": (3,), "5": (3, 2), "7": (3, 2, 2)}


@dataclass(frozen=True)
class Word:
    start: float
    end: float
    text: str


def timestamp(milliseconds: int) -> str:
    hours, rest = divmod(milliseconds, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    seconds, millis = divmod(rest, 1000)
    return f"{hours:02}:{minutes:02}:{seconds:02},{millis:03}"


APOSTROPHES = "'\u2018\u2019\u02bb\u02bc`\u00b4"
_APOSTROPHE_SPACES = re.compile(r"(?<=[^\W\d_])\s*([" + re.escape(APOSTROPHES) + r"])\s*(?=[^\W\d_])")


def joined_words(words, pause=0.8):
    """Join apostrophe fragments before counting words or assigning cue edges."""
    pending = None
    for word in words:
        if (not math.isfinite(word.start) or not math.isfinite(word.end)
                or word.end < word.start or word.end < 0 or not word.text.strip()):
            continue
        text = _APOSTROPHE_SPACES.sub(r"\1", word.text.replace("\r", " ").replace("\n", " "))
        current = Word(max(0, word.start), word.end, text)
        if pending:
            left, right = pending.text.rstrip(), text.lstrip()
            # A complete quoted phrase is punctuation, not a word suffix.
            quoted = (len(right) > 2 and right[0] in APOSTROPHES
                      and right.rstrip(".,!?;:")[-1] in APOSTROPHES)
            starts_apostrophe = (right[0] in APOSTROPHES and not quoted
                                 and (not right[1:].strip() or right[1:].lstrip()[0].isalpha()))
            ends_apostrophe = (len(left) > 1 and left[-1] in APOSTROPHES
                               and left[-2].isalpha() and left.lstrip()[0] not in APOSTROPHES)
            attached_letters = (not text[0].isspace() and left[-1].isascii()
                                and left[-1].isalpha() and right[0].isascii() and right[0].isalpha())
            if (current.start - pending.end < pause and
                    ((left[-1].isalpha() and starts_apostrophe)
                     or (ends_apostrophe and right[0].isalpha()) or attached_letters)):
                pending = Word(pending.start, max(pending.end, current.end),
                               _APOSTROPHE_SPACES.sub(r"\1", left + right))
                continue
            yield pending
        pending = current
    if pending:
        yield pending


def render_srt(words, width=42, max_duration=6.0, pause=0.8, layout_mode="auto") -> str:
    """Group words into at most two lines, splitting on pauses and sentences.

    Whisper word text includes language-appropriate leading spaces. Preserve
    those spaces (and lack of spaces in languages such as Chinese).
    """
    if layout_mode != "auto" and layout_mode not in LAYOUTS:
        raise ValueError("Unsupported subtitle layout")
    pattern = LAYOUTS.get(layout_mode)
    cues = []
    group = []

    def layout(items):
        if pattern:
            lines, offset = [], 0
            for count in pattern:
                line = "".join(w.text for w in items[offset:offset + count]).strip()
                if line:
                    lines.append(line)
                offset += count
            return lines
        text = "".join(w.text for w in items).strip()
        return textwrap.wrap(text, width=width, break_long_words=True,
                             break_on_hyphens=False)

    def flush():
        if group:
            cues.append((group[0].start, group[-1].end, layout(group)))
            group.clear()

    for word in joined_words(words, pause=pause):
        if not (math.isfinite(word.start) and math.isfinite(word.end)):
            continue
        if not word.text.strip() or word.end <= word.start or word.end <= 0:
            continue
        clean = Word(max(0, word.start), word.end,
                     word.text.replace("\r", " ").replace("\n", " "))
        if group and (clean.start - group[-1].end >= pause
                      or clean.end - group[0].start > max_duration
                      or (pattern and len(group) >= sum(pattern))
                      or (not pattern and len(layout(group + [clean])) > 2)):
            flush()
        group.append(clean)
        if not pattern and clean.text.rstrip().endswith((".", "?", "!", "。", "？", "！")):
            flush()
    flush()
    output = []
    previous_end = 0
    for start, end, lines in cues:
        start_ms = max(previous_end, round(start * 1000))
        end_ms = round(end * 1000)
        if end_ms <= start_ms:
            continue
        # Escape markup so spoken text cannot become subtitle formatting.
        body = "\n".join(lines).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        output.append(f"{len(output) + 1}\n{timestamp(start_ms)} --> {timestamp(end_ms)}\n{body}\n")
        previous_end = end_ms
    return "\n".join(output) + ("\n" if output else "")


def plain_text_from_srt(srt):
    """Flatten our generated SRT cues and decode escaped spoken text once."""
    from html import unescape
    bodies = []
    for block in srt.strip().split("\n\n"):
        lines = block.splitlines()
        if len(lines) >= 3 and " --> " in lines[1]:
            bodies.extend(lines[2:])
    return unescape(" ".join(bodies)).strip()
