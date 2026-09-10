from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import AsyncMock, Mock, patch

from telegram_srt.bot import BotService, Settings, build_application
from telegram_srt.media import MediaError


class FakeTranscriber:
    def __init__(self, fail=False):
        self.fail = fail
        self.folder = None

    def transcribe(self, source, folder, language, layout="auto"):
        self.folder = folder
        assert source.read_bytes() == b"test"
        if self.fail:
            raise MediaError("No speech was detected.")
        return "1\n00:00:01,000 --> 00:00:02,000\nHello!\n\n", "en"


class BotTests(unittest.IsolatedAsyncioTestCase):
    async def test_success_and_failure_cleanup(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                engine = FakeTranscriber(fail)
                service = BotService(Settings("123:fake"), engine)
                service.active_users.add(1)
                async def download(custom_path, **kwargs):
                    Path(custom_path).write_bytes(b"test")
                remote = SimpleNamespace(file_size=4, download_to_drive=AsyncMock(side_effect=download))
                media = SimpleNamespace(get_file=AsyncMock(return_value=remote))
                message = SimpleNamespace(reply_document=AsyncMock(), chat_id=10)
                notice = SimpleNamespace(edit_text=AsyncMock())
                await service.process(message, media, "../../sample.mp4", 1, None, notice)
                self.assertNotIn(1, service.active_users)
                self.assertFalse(engine.folder.exists())
                if fail:
                    message.reply_document.assert_not_awaited()
                    self.assertIn("No speech", notice.edit_text.call_args.args[0])
                else:
                    self.assertEqual(message.reply_document.call_args.kwargs["filename"], "sample.srt")
                    self.assertIn(b"00:00:01,000", message.reply_document.call_args.kwargs["document"].getvalue())

    async def test_reject_oversized_file_before_queue(self):
        service = BotService(Settings("123:fake"))
        media = SimpleNamespace(file_name="large.mp4", mime_type="video/mp4", file_size=21 * 1024 * 1024)
        message = SimpleNamespace(voice=None, audio=None, video=media, video_note=None, document=None, reply_text=AsyncMock())
        update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=1))
        await service.receive(update, SimpleNamespace())
        self.assertFalse(service.active_users)
        self.assertIn("exceeds", message.reply_text.call_args.args[0])

    async def test_duplicate_and_queue_limit(self):
        service = BotService(Settings("123:fake", max_jobs=1))
        service.active_users.add(1)
        media = SimpleNamespace(file_name="clip.mp4", mime_type="video/mp4", file_size=1)
        message = SimpleNamespace(voice=None, audio=None, video=media, video_note=None, document=None, reply_text=AsyncMock())
        for uid, expected in [(1, "already"), (2, "queue is full")]:
            update = SimpleNamespace(effective_message=message, effective_user=SimpleNamespace(id=uid))
            await service.receive(update, SimpleNamespace())
            self.assertIn(expected, message.reply_text.call_args.args[0])

    async def test_inline_selection_ownership_stale_and_duplicate(self):
        service = BotService(Settings("123:fake"))
        media = SimpleNamespace(file_name="clip.mp4", mime_type="video/mp4", file_size=1)
        message = SimpleNamespace(voice=None, audio=None, video=media, video_note=None,
                                  document=None, reply_text=AsyncMock())
        user, chat = SimpleNamespace(id=1), SimpleNamespace(id=10)
        update = SimpleNamespace(effective_message=message, effective_user=user, effective_chat=chat)
        tasks = []
        def schedule(coro, **kwargs):
            tasks.append(coro)
            coro.close()
        context = SimpleNamespace(user_data={"language": "en"}, application=SimpleNamespace(create_task=Mock(side_effect=schedule)))
        await service.receive(update, context)
        self.assertFalse(service.active_users)
        context.application.create_task.assert_not_called()
        keyboard = message.reply_text.call_args.kwargs["reply_markup"].inline_keyboard
        self.assertEqual([r[0].text for r in keyboard], ["\U0001f4dd Text only", "1 word per line", "2 words per line", "3 words per line", "5 words per 2 lines", "7 words per 3 lines"])
        old_data = keyboard[3][0].callback_data
        await service.receive(update, context)
        data = message.reply_text.call_args.kwargs["reply_markup"].inline_keyboard[3][0].callback_data
        query = SimpleNamespace(data=old_data, answer=AsyncMock(), edit_message_text=AsyncMock(), message=SimpleNamespace())
        callback = SimpleNamespace(callback_query=query, effective_user=user, effective_chat=chat)
        await service.choose_layout(callback, context)
        self.assertEqual(len(tasks), 0)
        query.data = data
        callback.effective_user = SimpleNamespace(id=2)
        await service.choose_layout(callback, context)
        self.assertEqual(len(tasks), 0)
        callback.effective_user = user
        await service.choose_layout(callback, context)
        self.assertEqual(len(tasks), 1)
        self.assertIn(1, service.active_users)
        self.assertNotIn(1, service.pending)
        query.edit_message_text.assert_awaited_once()
        await service.choose_layout(callback, context)
        self.assertEqual(len(tasks), 1)

    async def test_text_only_and_cached_output_do_not_retranscribe(self):
        engine = FakeTranscriber()
        engine.transcribe = Mock(wraps=engine.transcribe)
        service = BotService(Settings("123:fake"), engine)
        async def download(custom_path, **kwargs):
            Path(custom_path).write_bytes(b"test")
        remote = SimpleNamespace(file_size=4, download_to_drive=AsyncMock(side_effect=download))
        media = SimpleNamespace(get_file=AsyncMock(return_value=remote))
        message = SimpleNamespace(reply_document=AsyncMock(), reply_text=AsyncMock(), chat_id=10)
        notice = SimpleNamespace(edit_text=AsyncMock())
        await service.process(message, media, "clip.mp4", 1, "en", notice, "text")
        message.reply_text.assert_awaited_once_with("Hello!", parse_mode=None, disable_web_page_preview=True)
        message.reply_document.assert_not_awaited()
        self.assertFalse(service.transcripts)
        self.assertFalse(engine.folder.exists())
        await service.process(message, media, "clip.mp4", 1, "en", notice, "5")
        data = message.reply_document.call_args.kwargs["reply_markup"].inline_keyboard[0][0].callback_data
        query = SimpleNamespace(data=data, answer=AsyncMock(), message=message)
        update = SimpleNamespace(callback_query=query, effective_user=SimpleNamespace(id=1), effective_chat=SimpleNamespace(id=10))
        await service.get_text(update, None)
        self.assertEqual(message.reply_text.await_count, 2)
        self.assertEqual(engine.transcribe.call_count, 2)
        for uid, chat in [(2, 10), (1, 11)]:
            update.effective_user.id, update.effective_chat.id = uid, chat
            await service.get_text(update, None)
            self.assertEqual(message.reply_text.await_count, 2)
        update.effective_user.id, update.effective_chat.id = 1, 10
        item = next(iter(service.transcripts.values()))
        item["expires"] = 0
        await service.get_text(update, None)
        self.assertEqual(message.reply_text.await_count, 2)
        self.assertTrue(query.answer.call_args.kwargs["show_alert"])
        item["timer"].cancel()

    async def test_long_text_file_and_literal_message(self):
        service = BotService(Settings("123:fake"))
        message = SimpleNamespace(reply_document=AsyncMock(), reply_text=AsyncMock())
        literal = "<b>Salom</b> & [hello]"
        await service.send_text(message, literal, "clip")
        self.assertEqual(message.reply_text.call_args.args[0], literal)
        self.assertIsNone(message.reply_text.call_args.kwargs["parse_mode"])
        long_text = "\U0001f600" * 2100
        await service.send_text(message, long_text, "clip")
        self.assertEqual(message.reply_document.call_args.kwargs["filename"], "clip.txt")
        self.assertEqual(message.reply_document.call_args.kwargs["document"].getvalue().decode("utf-8"), long_text)

    async def test_local_file_is_read_without_copy_or_deletion(self):
        with tempfile.TemporaryDirectory() as directory:
            shared = Path(directory)
            original = shared / "recording.mp4"
            original.write_bytes(b"test")
            remote = SimpleNamespace(file_size=4, file_path=str(original), download_to_drive=AsyncMock())
            media = SimpleNamespace(get_file=AsyncMock(return_value=remote))
            message = SimpleNamespace(reply_document=AsyncMock(), reply_text=AsyncMock(), chat_id=10)
            notice = SimpleNamespace(edit_text=AsyncMock())
            service = BotService(Settings("123:fake", max_mb=4096, api_url="http://telegram-api:8081"), FakeTranscriber())
            with patch.dict("os.environ", {"TELEGRAM_FILES_DIR": str(shared)}):
                await service.process(message, media, "recording.mp4", 1, "en", notice, "text")
            remote.download_to_drive.assert_not_awaited()
            self.assertEqual(original.read_bytes(), b"test")
            message.reply_text.assert_awaited_once()
            self.assertFalse(service.transcriber.folder.exists())
            with patch.dict("os.environ", {"TELEGRAM_FILES_DIR": str(shared / "different-root")}):
                await service.process(message, media, "recording.mp4", 1, "en", notice, "text")
            self.assertIn("not accessible", notice.edit_text.call_args.args[0])
            self.assertEqual(message.reply_text.await_count, 1)

    async def test_application_builds(self):
        app = build_application(Settings("123:fake"))
        self.assertEqual(len(app.handlers[0]), 6)
