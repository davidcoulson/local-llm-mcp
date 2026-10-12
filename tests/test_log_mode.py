"""Tests for local_summarize's log mode: parsing, normalisation, the digest, and routing.

Run: uv run --with pytest --with "mcp>=1.20,<2" --with httpx --with numpy pytest
"""

from __future__ import annotations

import asyncio
import inspect
import json
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import server


def syslog_fixture() -> str:
    """A small Unraid-style syslog with the kinds of events the digest must handle."""
    lines = []
    # sshd noise: many sessions from changing ports.
    for i in range(300):
        minute = 10 + i % 50
        lines.append(
            f"Oct 10 11:{minute:02d}:{i % 60:02d} tower sshd-session[{4000 + i}]: "
            f"Accepted publickey for root from 192.0.2.{20 + i % 3} port {50000 + i} ssh2"
        )
    # UPS glitch, three times in ten minutes.
    for t, ago in (
        ("11:05:38", "382.375"),
        ("11:13:38", "416.413"),
        ("11:15:46", "64.0645"),
    ):
        lines.append(
            f"Oct 10 {t} tower usbhid-ups[15329]: Device reports that it is charging, but the "
            f"current battery charge 99 is less than 100 reported {ago} seconds ago"
        )
    # Hourly mover run with two folder-not-dataset warnings and routine lines.
    for hour in range(12, 22):
        for share, size in (("isos", 214 + hour), ("media", 3 + hour)):
            lines += [
                f"Oct 10 {hour}:00:03 tower move: Share Information: Name: {share} - Path: /mnt/exos24tb/{share}",
                f"Oct 10 {hour}:00:03 tower move: cannot open 'exos24tb/{share}': dataset does not exist",
                f"Oct 10 {hour}:00:03 tower move: Warning: Share is currently a folder. Convert it to a dataset for better performance.",
                f"Oct 10 {hour}:00:03 tower move: exos24tb/{share} used: {size}GiB",
            ]
        lines.append(f"Oct 10 {hour}:00:02 tower move: Secondary storage: none")
        # Disk spin-downs and docker veth churn: routine noise.
        lines.append(
            f"Oct 10 {hour}:14:06 tower emhttpd: spinning down /dev/sd{'abc'[hour % 3]}"
        )
        lines.append(
            f"Oct 10 {hour}:22:21 tower kernel: veth{hour:02d}f5298e1: renamed from eth0"
        )
    # One-off events.
    lines += [
        "Oct 10 12:43:05 tower smbd[506365]:   fruit_get_bandsize: Didn't find band-size key in [/mnt/ssd8tb/timemachine/Mac.sparsebundle/Info.plist]",
        "Oct 10 12:43:05 tower smbd[506365]:   fruit_tmsize_do_dirent: Processing sparsebundle [Mac.sparsebundle] failed",
        "Oct 10 12:57:51 tower tpm-guid-check: GUID mismatch persists after an automatic reboot; not rebooting again (manual intervention needed)",
        "Oct 10 12:57:53 tower tpm-guid-check: licence OK or not the TPM-GUID case; nothing to do",
        "Oct 10 18:01:56 tower kernel: usb 3-6.1: USB disconnect, device number 3",
        "Oct 10 18:01:59 tower kernel: usb 3-6.1: new full-speed USB device number 6 using xhci_hcd",
        "Oct 10 19:30:00 tower lldpd[18007]: an unremarkable one-off message",
    ]
    lines.sort(key=lambda ln: ln[7:15])  # chronological, as a real syslog is
    return "\n".join(lines) + "\n"


# --- parsing --------------------------------------------------------------------------------


def test_parse_bsd_line_drops_host_and_pid():
    e = server.parse_log_line(
        "Oct 10 12:43:05 tower smbd[506365]:   Processing sparsebundle failed"
    )
    assert (e.stamp, e.program, e.message) == (
        "Oct 10 12:43:05",
        "smbd",
        "Processing sparsebundle failed",
    )
    assert e.when is not None and e.when.hour == 12 and e.when.minute == 43


def test_parse_program_with_spaces_and_kernel_timestamp():
    e = server.parse_log_line(
        "Oct 10 11:14:06 tower SAS Assist v2024.11.25: Spinning down device /dev/sdah"
    )
    assert e.program == "SAS Assist v2024.11.25"
    assert e.message == "Spinning down device /dev/sdah"
    k = server.parse_log_line(
        "Oct  9 01:02:03 tower kernel: [ 1234.567890] ata5: SATA link down"
    )
    assert (k.program, k.message) == ("kernel", "ata5: SATA link down")


def test_parse_iso_timestamps_and_level_lines():
    e = server.parse_log_line(
        "2026-10-10T11:00:03.123+01:00 tower dockerd[99]: container started"
    )
    assert (e.program, e.message) == ("dockerd", "container started")
    assert e.when is not None and e.when.year == 2026
    app = server.parse_log_line("2026-10-10 11:00:03,412 ERROR db: connection lost")
    assert app.program == "" and app.message == "ERROR db: connection lost"


def test_continuation_line_inherits_previous_entry():
    prev = server.parse_log_line("Oct 10 12:00:00 tower kernel: Call Trace:")
    cont = server.parse_log_line("  dump_stack+0x5c/0x80", prev)
    assert cont.when == prev.when and cont.program == "kernel"
    assert cont.message == "dump_stack+0x5c/0x80"


def test_looks_like_log():
    assert server.looks_like_log(syslog_fixture())
    prose = "\n".join(f"This is paragraph {i} of a design document." for i in range(50))
    assert not server.looks_like_log(prose)
    assert server.looks_like_log(
        "\n".join(f"2026-10-10 11:00:{i:02d} INFO worker: tick {i}" for i in range(20))
    )


# --- normalisation --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("a", "b"),
    [
        (
            "Accepted publickey for root from 192.0.2.20 port 51234 ssh2",
            "Accepted publickey for root from 192.168.1.9 port 2201 ssh2",
        ),
        ("exos24tb/isos used: 214GiB", "exos24tb/isos used: 3.8TiB"),
        (
            "usb 3-6.1: USB disconnect, device number 3",
            "usb 1-2: USB disconnect, device number 12",
        ),
        ("vethf5298e1: renamed from eth0", "veth1ecb4e7: renamed from eth0"),
        ("mapped at 0xffff8a3b0000 len 4096", "mapped at 0x1000 len 8"),
        (
            "job 3f2a9c1e-77b0-4c1e-9d2a-0123456789ab done",
            "job 00000000-1111-2222-3333-444444444444 done",
        ),
        ("marker from 2026-10-10 12:57:51", "marker from 2025-01-02 03:04:05"),
        (
            "removal request for fe80::c436:b7ff:fe1b:504c%199",
            "removal request for fe80::1%7",
        ),
        (
            "charge 99 is less than 100 reported 382.375 seconds ago",
            "charge 98 is less than 100 reported 64.0645 seconds ago",
        ),
        ("image sha d11c9ad pulled", "image sha 741556d pulled"),
    ],
)
def test_repeats_share_a_template(a, b):
    assert server.normalize_message(a) == server.normalize_message(b)


@pytest.mark.parametrize(
    ("a", "b"),
    [
        # Names that contain digits are identifiers, not counters.
        (
            "cannot open 'exos24tb/isos': dataset does not exist",
            "cannot open 'exos24tb/media': dataset does not exist",
        ),
        ("spinning down /dev/sdah", "spinning down /dev/sdm"),
        (
            "Consider removing /boot/config/shares/Games.cfg",
            "Consider removing /boot/config/shares/poste.cfg",
        ),
    ],
)
def test_different_events_keep_separate_templates(a, b):
    assert server.normalize_message(a) != server.normalize_message(b)


def test_normalize_keeps_words_and_digits_inside_names():
    out = server.normalize_message(
        "nvme8tb/zvols used: 35GiB on sdq1 via ssh2 from 10.0.0.1:22"
    )
    assert out == "nvme8tb/zvols used: <N> on sdq1 via ssh2 from <IP>"


# --- classification -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("program", "message", "flag"),
    [
        ("smbd", "Processing sparsebundle [Mac.sparsebundle] failed", "fail"),
        ("tpm-guid-check", "GUID mismatch persists after an automatic reboot", "warn"),
        ("move", "Warning: Share is currently a folder.", "warn"),
        ("kernel", "usb 3-6.1: USB disconnect, device number 3", "hw"),
        ("kernel", "ata5.00: failed command: READ FPDMA QUEUED", "hw"),
        ("kernel", "nvme nvme0: I/O 12 QID 3 timeout, aborting", "hw"),
        ("kernel", "mce: [Hardware Error]: Machine check events logged", "hw"),
        ("kernel", "Buffer I/O error on dev sdq1, logical block 0", "error"),
        ("kernel", "bond0: link down", "hw"),
        ("kernel", "php-fpm[1234]: segfault at 0 ip 000055 sp 00007f error 4", "hw"),
        ("kernel", "Out of memory: Killed process 4242 (qemu)", "hw"),
        ("kernel", "Call Trace:", "hw"),
        (
            "usbhid-ups",
            "Device reports that it is charging, but the current battery charge 99",
            "hw",
        ),
        ("kernel", "BTRFS critical (device sdb1): corrupt leaf", "crit"),
    ],
)
def test_classify_flags(program, message, flag):
    assert flag in server.classify(program, message)


@pytest.mark.parametrize(
    ("program", "message"),
    [
        ("move", "Primary storage: nvme8tb (zfs) - size: 14TiB - used: 26 %"),
        ("move", "Notify: errorsOnly"),
        ("emhttpd", "read SMART /dev/sdah"),
        ("sshd-session", "Accepted publickey for root from 192.0.2.20 port 51234 ssh2"),
        ("move", "Secondary storage: none"),
    ],
)
def test_classify_leaves_routine_lines_unflagged(program, message):
    assert server.classify(program, message) == ()


# --- digest ---------------------------------------------------------------------------------


def test_digest_counts_repeats_exactly_and_spots_the_period():
    d = server.build_digest(syslog_fixture(), name="syslog")
    line = next(ln for ln in d.text.splitlines() if "exos24tb/isos': dataset" in ln)
    assert line.startswith("10x 12:00:03-21:00:03, every ~60 min [warn] move:")
    folder = next(ln for ln in d.text.splitlines() if "currently a folder" in ln)
    assert folder.startswith("20x ")
    ups = next(
        ln for ln in d.text.splitlines() if "usbhid-ups" in ln and "charging" in ln
    )
    assert ups.startswith("3x 11:05:38-11:15:46 [hw]")


def test_digest_keeps_rare_events_with_times_and_context():
    text = server.build_digest(syslog_fixture()).text
    once = text.split("== Problems seen only 1-3 times")[1].split("\n== ")[0]
    assert (
        "1x 12:43:05 [fail] smbd: fruit_tmsize_do_dirent: Processing sparsebundle"
        in once
    )
    assert "1x 12:57:51 [warn] tpm-guid-check: GUID mismatch persists" in once
    assert "1x 18:01:56 [hw] kernel: usb 3-6.1: USB disconnect" in once
    # Lines logged seconds around a one-off problem travel with it.
    assert "near: 18:01:59 kernel: usb 3-6.1: new full-speed USB device" in once
    assert "near: 12:57:53 tpm-guid-check: licence OK" in once
    assert "near: 12:43:05 smbd: fruit_get_bandsize" in once
    # An unflagged one-off still appears, in its own section.
    rare = text.split("== Other rare lines")[1]
    assert "19:30:00 lldpd: an unremarkable one-off message" in rare


def test_digest_collapses_noise_into_counts():
    text = server.build_digest(syslog_fixture()).text
    assert "Accepted publickey" not in text.split("== Routine noise")[0]
    noise = text.split("== Routine noise, counted only ==")[1]
    assert "sshd logins and sessions: 300 lines (1 kinds)" in noise
    assert "disk spin-down/spin-up and SMART reads: 10 lines" in noise
    assert "docker network interfaces (veth/bridge): 10 lines" in noise
    routine = text.split("== Routine repeating messages")[1]
    assert "move:" in routine and "in 10 runs every ~60 min" in routine


def test_focus_brings_noise_back():
    text = server.build_digest(syslog_fixture(), focus="ssh logins").text
    assert "sshd logins and sessions" not in text
    focus = text.split("== Other lines matching the focus")[1].split("\n== ")[0]
    assert "300x" in focus and "Accepted publickey for root" in focus


def test_focus_words_match_at_word_starts():
    lines = [
        "Oct 10 11:00:00 tower move: backups used: 3.6TiB",
        "Oct 10 12:00:00 tower move: backups used: 3.7TiB",
        "Oct 10 13:00:00 tower move: backups used: 3.8TiB",
        "Oct 10 13:30:00 tower emhttpd: disk sdb temperature 41 C",
        "Oct 10 13:40:00 tower emhttpd: diskette mode off",
        "Oct 10 14:00:00 tower webgui: ups status page opened",
    ]
    text = server.build_digest("\n".join(lines), focus="ups and disks").text
    focus = text.split("== Other lines matching the focus (ups, disk) ==")[1].split(
        "\n== "
    )[0]
    assert "backups" not in focus  # "ups" must start a word
    assert "webgui: ups status page opened" in focus
    assert "disk sdb temperature" in focus  # "disks" -> "disk", matched at word start
    assert "diskette mode off" in focus


def test_digest_is_small_and_has_a_header():
    d = server.build_digest(syslog_fixture(), name="syslog")
    assert d.lines == len(syslog_fixture().splitlines())
    assert d.text.startswith(f"LOG DIGEST of syslog: {d.lines:,} lines")
    assert "Oct 10, 11:05:38 to 21:22:21" in d.text
    assert len(d.text) < 5_000
    assert "18:01:56 [hw] kernel: usb 3-6.1: USB disconnect" in d.problems


def test_digest_stays_bounded_when_every_line_is_a_distinct_error():
    words = ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel"]
    lines = [
        f"Oct 10 12:{i // 60 % 60:02d}:{i % 60:02d} tower app: error in module {words[i % 8]}{words[i // 8 % 8]}{words[i // 64 % 8]}"
        for i in range(2000)
    ]
    d = server.build_digest("\n".join(lines))
    limit = (
        server.DIGEST_FLAGGED_CHARS
        + server.DIGEST_RARE_CHARS
        + server.DIGEST_ROUTINE_CHARS
        + 2_000
    )
    assert len(d.text) < limit
    assert "more flagged messages omitted for space" in d.text
    assert len(d.problems) < server.PROBLEMS_CHARS + 200


def test_digest_without_timestamps_still_counts():
    d = server.build_digest("disk error 5 on sdb\ndisk error 7 on sdb\nall good\n")
    assert d.lines == 3
    assert "no timestamps found" in d.text
    assert "2x ? [error] disk error 5 on sdb" in d.text
    assert d.problems == "2x ? [error] disk error 5 on sdb"


# --- the tool -------------------------------------------------------------------------------


@pytest.fixture
def fake_chat(monkeypatch):
    calls = []

    async def chat(client, system, user, max_tokens=1024):
        calls.append({"system": system, "user": user, "max_tokens": max_tokens})
        return f"summary {len(calls)}"

    monkeypatch.setattr(server, "_chat", chat)
    return calls


def test_signature_is_backward_compatible():
    params = inspect.signature(server.local_summarize).parameters
    assert list(params)[:2] == ["path", "focus"]
    assert params["focus"].default == ""
    assert all(
        p.default is not inspect.Parameter.empty for p in list(params.values())[1:]
    )


def test_log_file_goes_to_the_model_as_one_digest(tmp_path, fake_chat):
    f = tmp_path / "syslog.txt"
    f.write_text(syslog_fixture())
    out = asyncio.run(server.local_summarize(str(f), focus="hardware"))
    assert len(fake_chat) == 1
    sent = fake_chat[0]["user"]
    assert sent.startswith("Focus: hardware\n\nLOG DIGEST of syslog.txt")
    assert "192.0.2.20" not in sent  # noise is counted, not sent
    assert len(sent) < 8_000
    assert out.startswith("summary 1\n\nFlagged messages, exact")
    assert "1x 12:43:05 [fail] smbd" in out
    assert "include_digest=True shows the digest" in out


def test_include_digest_appends_it(tmp_path, fake_chat):
    f = tmp_path / "syslog.txt"
    f.write_text(syslog_fixture())
    out = asyncio.run(server.local_summarize(str(f), include_digest=True))
    assert "LOG DIGEST of syslog.txt" in out


def test_text_mode_keeps_the_chunked_path(tmp_path, fake_chat, monkeypatch):
    monkeypatch.setattr(server, "CHUNK_CHARS", 1_000)
    f = tmp_path / "syslog.txt"
    f.write_text(syslog_fixture())
    out = asyncio.run(server.local_summarize(str(f), mode="text"))
    pieces = len(server._chunks(syslog_fixture()))
    assert pieces > 1 and len(fake_chat) == pieces + 1
    assert all(c["max_tokens"] == 400 for c in fake_chat[:-1])
    assert out.startswith(f"summary {pieces + 1}")


def test_auto_mode_leaves_prose_alone(tmp_path, fake_chat):
    f = tmp_path / "notes.md"
    f.write_text("# Notes\n\nSome prose about the design.\n" * 5)
    asyncio.run(server.local_summarize(str(f)))
    assert len(fake_chat) == 1 and "LOG DIGEST" not in fake_chat[0]["user"]


def test_log_mode_can_be_forced(tmp_path, fake_chat):
    f = tmp_path / "odd.txt"
    f.write_text("disk error on sdb\nok\n")
    asyncio.run(server.local_summarize(str(f), mode="log"))
    assert "LOG DIGEST of odd.txt" in fake_chat[0]["user"]


def test_bad_mode_is_rejected(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x\n")
    with pytest.raises(ValueError, match="mode must be"):
        asyncio.run(server.local_summarize(str(f), mode="logs"))


# --- the model call -------------------------------------------------------------------------


def test_chat_caps_output_and_retries_without_think():
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "think" in body:
            return httpx.Response(
                400, json={"error": '"granite4" does not support thinking'}
            )
        return httpx.Response(200, json={"message": {"content": "  ok  "}})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await server._chat(client, "sys", "user", max_tokens=123)

    assert asyncio.run(run()) == "ok"
    assert len(bodies) == 2 and bodies[0]["think"] is False and "think" not in bodies[1]
    for b in bodies:
        assert b["options"]["num_predict"] == 123
        assert b["options"]["repeat_penalty"] > 1


def test_chat_raises_on_other_errors():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="boom")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await server._chat(client, "sys", "user")

    with pytest.raises(httpx.HTTPStatusError):
        asyncio.run(run())
