import asyncio
from dataclasses import dataclass
from io import BytesIO
import logging
import os
from pathlib import Path
import re
import secrets
import time
import tempfile

from dotenv import load_dotenv
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.error import TelegramError
from telegram.ext import Application, CallbackQueryHandler, CommandHandler, ContextTypes, MessageHandler, filters

from .media import MediaError
from .remote import RemoteTranscriber
from .subtitles import LAYOUTS, plain_text_from_srt

LOG = logging.getLogger(__name__)
EXTENSIONS = set("mp3 wav m4a aac ogg oga opus flac wma aiff aif amr mp4 mov mkv webm avi m4v mpeg mpg 3gp ts mts m2ts flv wmv".split())


@dataclass(frozen=True)
class Settings:
    token: str
    max_mb: int = 20
    max_seconds: int = 3600
    max_jobs: int = 3
    api_url: str = ""
    allowed_users: frozenset[int] = frozenset()

    @classmethod
    def from_env(cls):
        load_dotenv()
        token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
        if not token or token == "replace-with-your-bot-token":
            raise ValueError("Set TELEGRAM_BOT_TOKEN in .env first (get it from @BotFather).")
        settings = cls(
            token=token,
            max_mb=int(os.getenv("MAX_FILE_MB", "20")),
            max_seconds=int(os.getenv("MAX_DURATION_SECONDS", "3600")),
            max_jobs=int(os.getenv("MAX_JOBS", "3")),
            api_url=os.getenv("TELEGRAM_API_URL", "").rstrip("/"),
            allowed_users=frozenset(int(v.strip()) for v in os.getenv("ALLOWED_USER_IDS", "").split(",") if v.strip()),
        )
        if min(settings.max_mb, settings.max_seconds, settings.max_jobs) <= 0:
            raise ValueError("Size, duration, and job limits must be positive.")
        if not settings.api_url and settings.max_mb > 20:
            raise ValueError("MAX_FILE_MB > 20 requires a local TELEGRAM_API_URL.")
        return settings


class BotService:
    def __init__(self, settings, transcriber=None):
        self.settings = settings
        self.transcriber = transcriber or RemoteTranscriber()
        self.active_users = set()
        self.pending = {}
        self.transcripts = {}
        self.worker_lock = asyncio.Lock()

    def authorized(self, update):
        return (update.effective_user is not None and
                (not self.settings.allowed_users or update.effective_user.id in self.settings.allowed_users))

    async def help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.authorized(update):
            return
        await update.effective_message.reply_text(
            "Send a voice note, audio file, video, or video note, then choose Text only or a subtitle layout. Text arrives as a message or a .txt file for long recordings. SRT subtitles come "
            "with timestamps relative to the beginning of your recording.\n\n"
            "Common formats: MP3, WAV, M4A, OGG/Opus, FLAC, MP4, MOV, MKV, AVI and WebM. "
            "Send videos as files to preserve the original upload.\n\n"
            f"Limits: {self.settings.max_mb} MB, {self.settings.max_seconds // 60} minutes.\n"
            "/language uz - Uzbek (default, Rubai STT v2)\n"
            "/language auto - detect language\n"
            "/language en - set another language code\n"
            "/status — check your job\n\n"
            "Import the SRT into your editor alongside the same untrimmed recording. "
            "Automatic transcription and timing may need review, especially for noise or overlapping voices."
        )

    async def language(self, update, context):
        if not self.authorized(update):
            return
        from faster_whisper.tokenizer import _LANGUAGE_CODES
        code = context.args[0].lower() if len(context.args) == 1 else ""
        if code != "auto" and code not in _LANGUAGE_CODES:
            await update.effective_message.reply_text("Use /language auto or a supported code such as en, ru, uz, es, ar, zh.")
            return
        context.user_data["language"] = None if code == "auto" else code
        await update.effective_message.reply_text(f"Language: {code}. This applies to your next uploads.")

    async def status(self, update, context):
        if self.authorized(update):
            busy = update.effective_user.id in self.active_users
            if update.effective_user.id in self.pending and not busy:
                await update.effective_message.reply_text("Choose an output using the buttons below your upload.")
                return
            await update.effective_message.reply_text("Your recording is queued or processing." if busy else "No active job. Send an audio or video file.")

    async def receive(self, update, context):
        if not self.authorized(update):
            return
        message = update.effective_message
        media = message.voice or message.audio or message.video or message.video_note or message.document
        if media is None:
            await message.reply_text("Send an audio or video file, or use /help.")
            return
        name = getattr(media, "file_name", None) or "recording"
        suffix = Path(name).suffix.lower().lstrip(".")
        mime = getattr(media, "mime_type", "") or ""
        if message.document and suffix not in EXTENSIONS and not mime.startswith(("audio/", "video/")):
            await message.reply_text("Please send a supported audio or video file. See /help for formats.")
            return
        limit = self.settings.max_mb * 1024 * 1024
        if media.file_size and media.file_size > limit:
            await message.reply_text(f"This file exceeds my {self.settings.max_mb} MB limit. Please send a smaller clip.")
            return
        uid = update.effective_user.id
        if uid in self.active_users:
            await message.reply_text("You already have a job running. Please wait for its result before sending another.")
            return
        if len(self.active_users) >= self.settings.max_jobs:
            await message.reply_text("The queue is full. Please try again after a recording finishes.")
            return
        # Keep one pending upload per user; replacing it invalidates older buttons.
        now = time.monotonic()
        self.pending = {u: p for u, p in self.pending.items() if p["expires"] > now}
        token = secrets.token_hex(6)
        labels = [("text", "\U0001f4dd Text only"), ("1", "1 word per line"), ("2", "2 words per line"),
                  ("3", "3 words per line"), ("5", "5 words per 2 lines"),
                  ("7", "7 words per 3 lines")]
        keyboard = InlineKeyboardMarkup([
            [InlineKeyboardButton(label, callback_data=f"layout:{token}:{key}")]
            for key, label in labels
        ])
        self.pending[uid] = dict(token=token, message=message, media=media, name=name,
                                 language=context.user_data.get("language", "uz"),
                                 expires=now + 900, chat_id=update.effective_chat.id)
        try:
            await message.reply_text("Choose your output:", reply_markup=keyboard)
        except BaseException:
            self.pending.pop(uid, None)
            raise

    async def choose_layout(self, update, context):
        query = update.callback_query
        if not self.authorized(update):
            await query.answer("This selection is not available to you.", show_alert=True)
            return
        uid = update.effective_user.id
        parts = (query.data or "").split(":")
        pending = self.pending.get(uid)
        if (len(parts) != 3 or parts[2] not in {*LAYOUTS, "text"} or not pending
                or pending["token"] != parts[1]
                or pending["chat_id"] != update.effective_chat.id
                or pending["expires"] <= time.monotonic()):
            await query.answer("This selection expired or belongs to another upload. Send your recording again.", show_alert=True)
            return
        if uid in self.active_users or len(self.active_users) >= self.settings.max_jobs:
            await query.answer("The queue is full. Please try this button again shortly.", show_alert=True)
            return
        self.pending.pop(uid)
        self.active_users.add(uid)
        try:
            await query.answer()
            await query.edit_message_text("Recording accepted. Queued for transcription.")
            context.application.create_task(
                self.process(pending["message"], pending["media"], pending["name"], uid,
                             pending["language"], query.message, parts[2]), update=update,
            )
        except BaseException:
            self.active_users.discard(uid)
            self.pending[uid] = pending
            raise

    async def process(self, message, media, name, uid, language, notice, layout="auto"):
        try:
            async with self.worker_lock:
                with tempfile.TemporaryDirectory(prefix="telegram-srt-") as directory:
                    folder = Path(directory)
                    # Never construct local paths from a user's filename.
                    source = folder / "upload.media"
                    await notice.edit_text("Downloading your recording…")
                    remote = await media.get_file(read_timeout=3600 if self.settings.api_url else 180)
                    if remote.file_size and remote.file_size > self.settings.max_mb * 1024 * 1024:
                        raise MediaError("This file exceeds the configured download limit.")
                    if self.settings.api_url and getattr(remote, "file_path", None) and Path(remote.file_path).is_absolute():
                        shared_root = Path(os.getenv("TELEGRAM_FILES_DIR", "/var/lib/telegram-bot-api")).resolve()
                        local_source = Path(remote.file_path).resolve()
                        if not local_source.is_relative_to(shared_root) or not local_source.is_file():
                            raise MediaError("The local Telegram file storage is not accessible. Contact the bot operator.")
                        # Read shared files directly; never copy gigabytes on the event loop.
                        source = local_source
                    else:
                        await remote.download_to_drive(custom_path=source, read_timeout=180)
                    if source.stat().st_size > self.settings.max_mb * 1024 * 1024:
                        raise MediaError("This file exceeds the configured download limit.")
                    await notice.edit_text("Transcribing and aligning subtitles… Use /status to check your job.")
                    # Shield the thread and wait on cancellation before deleting its files.
                    job = asyncio.create_task(asyncio.to_thread(self.transcriber.transcribe, source, folder, language, "auto" if layout == "text" else layout))
                    try:
                        srt, detected = await asyncio.shield(job)
                    except asyncio.CancelledError:
                        await job
                        raise
                    if not srt:
                        raise MediaError("No speech was detected. Try a recording with clearer speech.")
                    stem = re.sub(r"[^\w .-]", "_", Path(name).stem, flags=re.UNICODE).strip(" .")[:100] or "transcript"
                    text = plain_text_from_srt(srt)
                    if layout == "text":
                        await self.send_text(message, text, stem)
                        await notice.edit_text("Done. Your transcript is above.")
                    else:
                        token = self.cache_transcript(uid, message.chat_id, text, stem)
                        await message.reply_document(
                            document=BytesIO(srt.encode("utf-8")), filename=f"{stem}.srt",
                            caption=f"Subtitles ready. Language: {detected}. Import with the original recording at the same timeline start.\nGet plain text below (available for 30 minutes).",
                            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton(
                                "\U0001f4dd Get plain text", callback_data=f"text:{token}")]]),
                            write_timeout=180,
                        )
                        await notice.edit_text("Done. Your SRT is attached above.")
        except MediaError as exc:
            await self.report(notice, str(exc))
        except Exception as exc:
            # Exception text can include a Telegram download URL containing the token.
            LOG.error("Transcription job failed (%s)", type(exc).__name__)
            await self.report(notice, "I couldn't finish this recording. Please try again or contact the bot operator.")
        finally:
            self.active_users.discard(uid)

    def cache_transcript(self, uid, chat_id, text, stem):
        # Bounded, temporary memory only; timers release text even while idle.
        while len(self.transcripts) >= 100:
            oldest = self.transcripts.pop(next(iter(self.transcripts)))
            oldest["timer"].cancel()
        token = secrets.token_hex(12)
        timer = asyncio.get_running_loop().call_later(1800, self.transcripts.pop, token, None)
        self.transcripts[token] = dict(uid=uid, chat_id=chat_id, text=text, stem=stem,
                                       expires=time.monotonic() + 1800, timer=timer)
        return token

    async def send_text(self, message, text, stem):
        # Count UTF-16 units conservatively for Telegram, including emoji.
        if len(text.encode("utf-16-le")) // 2 <= 4000:
            await message.reply_text(text, parse_mode=None, disable_web_page_preview=True)
        else:
            await message.reply_document(document=BytesIO(text.encode("utf-8")),
                                         filename=f"{stem}.txt", caption="Your full transcript.",
                                         write_timeout=180)

    async def get_text(self, update, context):
        query = update.callback_query
        token = (query.data or "").removeprefix("text:")
        item = self.transcripts.get(token)
        if (not self.authorized(update) or not item
                or item["uid"] != update.effective_user.id
                or item["chat_id"] != update.effective_chat.id
                or item["expires"] <= time.monotonic()):
            await query.answer("This transcript expired or belongs to another upload. Send the recording again and choose Text only.", show_alert=True)
            return
        await query.answer()
        try:
            await self.send_text(query.message, item["text"], item["stem"])
        except TelegramError:
            LOG.warning("Could not deliver cached transcript; button remains available for retry")

    async def report(self, notice, text):
        try:
            await notice.edit_text(text)
        except TelegramError:
            LOG.warning("Could not deliver job status")


async def error_handler(update, context):
    LOG.error("Telegram update failed (%s)", type(context.error).__name__)


def build_application(settings):
    service = BotService(settings)
    builder = Application.builder().token(settings.token).read_timeout(180).write_timeout(180)
    if settings.api_url:
        builder = (builder.base_url(f"{settings.api_url}/bot")
                   .base_file_url(f"{settings.api_url}/file/bot").local_mode(True))
    app = builder.build()
    app.add_handler(CommandHandler(["start", "help"], service.help))
    app.add_handler(CommandHandler("language", service.language))
    app.add_handler(CommandHandler("status", service.status))
    app.add_handler(CallbackQueryHandler(service.get_text, pattern=r"^text:"))
    app.add_handler(CallbackQueryHandler(service.choose_layout, pattern=r"^layout:"))
    app.add_handler(MessageHandler(filters.ALL & ~filters.COMMAND, service.receive))
    app.add_error_handler(error_handler)
    return app


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    try:
        settings = Settings.from_env()
    except ValueError as exc:
        raise SystemExit(str(exc)) from None
    build_application(settings).run_polling(allowed_updates=["message", "callback_query"])


if __name__ == "__main__":
    main()
