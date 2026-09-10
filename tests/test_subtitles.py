import unittest
from telegram_srt.subtitles import Word, render_srt, timestamp, plain_text_from_srt


class SubtitleTests(unittest.TestCase):
    def test_selected_layout_counts_and_word_timing(self):
        patterns = {"1": [1], "2": [2], "3": [3], "5": [3, 2], "7": [3, 2, 2]}
        words = [Word(2 + i * .3, 2.2 + i * .3, f" word{i}") for i in range(15)]
        for mode, pattern in patterns.items():
            with self.subTest(mode=mode):
                blocks = render_srt(words, layout_mode=mode).strip().split("\n\n")
                self.assertEqual([len(line.split()) for line in blocks[0].splitlines()[2:]], pattern)
                self.assertIn(f"00:00:02,000 --> {timestamp(round(words[sum(pattern)-1].end*1000))}", blocks[0])
                actual = [word for block in blocks for line in block.splitlines()[2:] for word in line.split()]
                self.assertEqual(actual, [w.text.strip() for w in words])
                for block in blocks:
                    self.assertLessEqual(len(block.splitlines()[2:]), len(pattern))
        with self.assertRaises(ValueError):
            render_srt(words, layout_mode="invalid")

    def test_plain_text_removes_cues_and_decodes_markup_once(self):
        srt = "1\n00:00:01,000 --> 00:00:02,000\nSalom &amp; &lt;b&gt;\ndunyo!\n\n2\n00:00:03,000 --> 00:00:04,000\nLiteral &amp;lt; 123.\n\n"
        self.assertEqual(plain_text_from_srt(srt), "Salom & <b> dunyo! Literal &lt; 123.")
        self.assertEqual(plain_text_from_srt(""), "")

    def test_uzbek_apostrophe_fragments_count_as_single_words(self):
        for apostrophe in "'\u2018\u2019\u02bb\u02bc`\u00b4":
            with self.subTest(apostrophe=apostrophe):
                words = [Word(1, 1.2, " so"), Word(1.2, 1.5, apostrophe + "z"),
                         Word(2, 2.1, " o"), Word(2.1, 2.1, " " + apostrophe + "   "),
                         Word(2.1, 2.4, " n"), Word(2.5, 2.8, " yil"),
                         Word(3, 3.2, " bo" + apostrophe), Word(3.2, 3.7, " ldi.")]
                srt = render_srt(words, layout_mode="1")
                blocks = srt.strip().split("\n\n")
                self.assertEqual([b.splitlines()[2] for b in blocks],
                                 ["so"+apostrophe+"z", "o"+apostrophe+"n", "yil", "bo"+apostrophe+"ldi."])
                self.assertIn("00:00:01,000 --> 00:00:01,500", blocks[0])
                self.assertIn("00:00:02,000 --> 00:00:02,400", blocks[1])
                self.assertIn("00:00:03,000 --> 00:00:03,700", blocks[3])
                self.assertEqual(plain_text_from_srt(srt), " ".join(b.splitlines()[2] for b in blocks))

    def test_apostrophe_words_at_every_layout_boundary(self):
        fragments = []
        for i in range(14):
            fragments.extend([Word(i*.3, i*.3+.1, " bo"), Word(i*.3+.1, i*.3+.25, " 'ldi")])
        for mode, pattern in {"1":[1], "2":[2], "3":[3], "5":[3,2], "7":[3,2,2]}.items():
            blocks=render_srt(fragments,layout_mode=mode).strip().split("\n\n")
            self.assertEqual([len(line.split()) for line in blocks[0].splitlines()[2:]],pattern)
            self.assertEqual(plain_text_from_srt("\n\n".join(blocks)), " ".join(["bo'ldi"]*14))

    def test_internal_apostrophe_spaces_and_quoted_text(self):
        self.assertIn("bo'ldi", render_srt([Word(0,1," bo   'ldi")],layout_mode="1"))
        blocks=render_srt([Word(0,1," dedi"),Word(1,2," 'salom'.")],layout_mode="1").strip().split("\n\n")
        self.assertEqual(len(blocks),2)
        self.assertEqual(blocks[1].splitlines()[2],"'salom'.")
        result=render_srt([Word(0,1," so"),Word(4,5," 'z")],layout_mode="1")
        self.assertEqual(len(result.strip().split("\n\n")),2)

    def test_millisecond_carry_and_hours(self):
        self.assertEqual(timestamp(round(59.9996 * 1000)), "00:01:00,000")
        self.assertEqual(timestamp(3_661_001), "01:01:01,001")

    def test_original_silence_and_gaps_preserved(self):
        srt = render_srt([Word(2.5, 3.1, " Hello"), Word(3.2, 3.8, " world."), Word(10, 11, " Again!")])
        self.assertEqual(srt, "1\n00:00:02,500 --> 00:00:03,800\nHello world.\n\n2\n00:00:10,000 --> 00:00:11,000\nAgain!\n\n")

    def test_two_line_layout(self):
        words = [Word(i / 5, (i + 1) / 5, " something") for i in range(30)]
        for block in render_srt(words).strip().split("\n\n"):
            lines = block.splitlines()[2:]
            self.assertLessEqual(len(lines), 2)
            self.assertTrue(all(len(line) <= 42 for line in lines))

    def test_unicode_and_language_spacing(self):
        result = render_srt([Word(0, 1, "你好"), Word(1, 2, "世界。")])
        self.assertIn("你好世界。", result)
        self.assertIn("Привет", render_srt([Word(0, 1, " Привет!")]))

    def test_invalid_words_and_overlap(self):
        result = render_srt([Word(float("nan"), 1, "invalid"), Word(0, 0, "zero"),
                             Word(0, 2, " First."), Word(1.8, 3, " Second.")])
        self.assertIn("00:00:02,000 --> 00:00:03,000", result)
        self.assertNotIn("invalid", result)
        self.assertEqual(render_srt([]), "")


if __name__ == "__main__":
    unittest.main()
