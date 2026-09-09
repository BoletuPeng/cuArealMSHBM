"""test_profile_io_writers.py — the two .b2nd profile writers.

  * **Codec** — the one-shot writer's LZ4-5 + BITSHUFFLE pipeline, and
    that the readers open a frame written with any other codec (a
    blosc2 frame self-describes its pipeline, which is why a codec
    change needs no ``format_version`` bump).
  * **Streaming writer equivalence** — ``SubjectProfileStreamWriter``
    (one chunk per session, so a producer can overlap the compression
    with the next session's compute) produces a file the readers cannot
    tell from the one-shot writer's: same payload, same chunk/block
    grid, same cparams, same vlmeta. Plus its own failure modes — a
    session never written, an aborted subject, a bad slab, and a file
    whose ``close()`` never ran.

The one-shot writer takes unpacked binary ``(T, N, D)`` only; its
pre-packed input mode went with its last caller (the stage pipeline's
retired GPU branch), and ``test_removed_symbols.py`` pins that. The
unpacked path's own contract is exercised by ``step2_io`` through the
public writer entry.

Written by Boletu Peng <zesheng.peng.21@ucl.ac.uk>
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from arealmshbm.data_io.profile_io import (
    SubjectProfileStreamWriter,
    open_subject_profile_packed_tnd,
    read_subject_profile_packed_tnd,
    write_subject_profile_tnd,
)


def _make_binary_tnd(T: int, N: int, D: int, seed: int = 0) -> np.ndarray:
    """Random binary (T, N, D) fp32 in {0, 1}."""
    rng = np.random.default_rng(seed)
    return (rng.random((T, N, D)) < 0.3).astype(np.float32)


def _packed(T: int, N: int, D: int, seed: int = 0) -> np.ndarray:
    arr = _make_binary_tnd(T, N, D, seed=seed).astype(np.uint8)
    return np.ascontiguousarray(np.packbits(arr, axis=-1, bitorder="little"))


def _unpacked(pk: np.ndarray, D: int) -> np.ndarray:
    """The ``(T, N, D)`` binary fp32 array behind packed ``pk`` -- what
    the one-shot writer takes."""
    return np.ascontiguousarray(
        np.unpackbits(pk, axis=-1, bitorder="little")[..., :D]
        .astype(np.float32))


# ─────────────────────────────────────────────────────────────────────
# Codec: LZ4-5 / BITSHUFFLE, and the reader's codec-agnosticism
# ─────────────────────────────────────────────────────────────────────
def test_writer_uses_lz4_level5_bitshuffle(tmp_path: Path) -> None:
    """Pin the codec pipeline (see ``profile_io._cparams``)."""
    import blosc2

    from arealmshbm.data_io.profile_io import _cparams

    cp_ = _cparams()
    assert cp_.codec == blosc2.Codec.LZ4
    assert cp_.clevel == 5
    assert list(cp_.filters)[0] == blosc2.Filter.BITSHUFFLE
    assert all(f == blosc2.Filter.NOFILTER for f in list(cp_.filters)[1:])

    pk = _packed(3, 40, 19, seed=5)
    p = tmp_path / "lz4.b2nd"
    write_subject_profile_tnd(p, _unpacked(pk, 19))
    on_disk = blosc2.open(str(p), mode="r")
    try:
        assert on_disk.schunk.cparams.codec == blosc2.Codec.LZ4
    finally:
        del on_disk


def test_reader_opens_a_profile_written_with_another_codec(
        tmp_path: Path) -> None:
    """A blosc2 frame self-describes its codec and filter pipeline, so
    the readers key on ``format_version`` alone — a codec change is not
    a format change."""
    import blosc2

    from arealmshbm.data_io.profile_io import (
        _VLMETA_BITORDER_KEY, _VLMETA_BITORDER_VAL,
        _VLMETA_D_UNPACKED_KEY, _VLMETA_FORMAT_KEY, _VLMETA_FORMAT_VAL,
    )

    pk = _packed(2, 33, 21, seed=6)
    T, N, Db = pk.shape
    p = tmp_path / "other_codec.b2nd"
    other = blosc2.CParams(
        codec=blosc2.Codec.ZSTD, clevel=1, nthreads=1,
        filters=[blosc2.Filter.NOFILTER] * 6)
    a = blosc2.asarray(pk, chunks=(1, N, Db), urlpath=str(p), mode="w",
                       cparams=other)
    a.schunk.vlmeta[_VLMETA_FORMAT_KEY] = _VLMETA_FORMAT_VAL
    a.schunk.vlmeta[_VLMETA_D_UNPACKED_KEY] = 21
    a.schunk.vlmeta[_VLMETA_BITORDER_KEY] = _VLMETA_BITORDER_VAL
    del a

    got, D = read_subject_profile_packed_tnd(p)
    assert D == 21
    assert np.array_equal(got, pk)


# ─────────────────────────────────────────────────────────────────────
# Streaming (per-session) writer
# ─────────────────────────────────────────────────────────────────────
def test_stream_writer_matches_the_one_shot_writer(tmp_path: Path) -> None:
    """The whole point: a file assembled a session at a time must be
    indistinguishable to the readers from one written in a single call
    — same payload, same chunk/block grid, same cparams, same vlmeta."""
    import blosc2

    T, N, D = 5, 41, 19          # D % 8 != 0 to carry padding bits
    pk = _packed(T, N, D, seed=11)

    p_one = tmp_path / "one_shot.b2nd"
    p_str = tmp_path / "nested" / "streamed.b2nd"
    write_subject_profile_tnd(p_one, _unpacked(pk, D))
    w = SubjectProfileStreamWriter(p_str, T=T, N=N, D_unpacked=D)
    assert w.written == set()
    for t in range(T):
        w.write_session(t, pk[t])
    assert w.written == set(range(T))
    assert w.close() == p_str
    w.close()                     # idempotent

    a, Da = read_subject_profile_packed_tnd(p_one)
    b, Db = read_subject_profile_packed_tnd(p_str)
    assert Da == Db == D
    assert a.shape == b.shape == (T, N, (D + 7) // 8)
    assert np.array_equal(a, b)
    assert np.array_equal(b, pk)

    h_one = blosc2.open(str(p_one), mode="r")
    h_str = blosc2.open(str(p_str), mode="r")
    try:
        assert h_str.chunks == h_one.chunks == (1, N, (D + 7) // 8)
        assert h_str.blocks == h_one.blocks
        assert h_str.schunk.cparams.codec == h_one.schunk.cparams.codec
        assert h_str.schunk.cparams.clevel == h_one.schunk.cparams.clevel
        assert (h_str.schunk.vlmeta.getall()
                == h_one.schunk.vlmeta.getall())
    finally:
        del h_one, h_str

    # ... and the lazy reader still decodes one chunk at a time.
    lazy = open_subject_profile_packed_tnd(p_str)
    try:
        assert lazy.D_unpacked == D
        assert np.array_equal(lazy[T - 1], pk[T - 1])
    finally:
        del lazy


def test_stream_writer_accepts_sessions_out_of_order(tmp_path: Path) -> None:
    """The b2nd chunk index IS the session index, not an append
    counter, so a producer that finishes out of order is still fine."""
    T, N, D = 6, 23, 17
    pk = _packed(T, N, D, seed=12)
    p = tmp_path / "shuffled.b2nd"
    with SubjectProfileStreamWriter(p, T=T, N=N, D_unpacked=D) as w:
        for t in (3, 0, 5, 1, 4, 2):
            w.write_session(t, pk[t])
    got, D_read = read_subject_profile_packed_tnd(p)
    assert D_read == D
    assert np.array_equal(got, pk)


def test_stream_writer_close_refuses_a_gap_and_removes_the_file(
        tmp_path: Path) -> None:
    """A chunk never written reads back as zeros — a silently wrong
    profile. close() must refuse, and not leave the file behind."""
    T, N, D = 4, 9, 13
    pk = _packed(T, N, D, seed=13)
    p = tmp_path / "gap.b2nd"
    w = SubjectProfileStreamWriter(p, T=T, N=N, D_unpacked=D)
    w.write_session(0, pk[0])
    w.write_session(2, pk[2])
    with pytest.raises(ValueError, match=r"never written: \[1, 3\]"):
        w.close()
    assert not p.exists()
    assert w.written == {0, 2}


def test_a_file_whose_close_never_ran_is_rejected(tmp_path: Path) -> None:
    """``format_version`` is stamped LAST, by close(). A process that
    died after the final ``write_session`` leaves a complete-looking
    frame on disk; without the stamp the readers must refuse it rather
    than serve a subject that may be missing a session."""
    T, N, D = 3, 17, 19
    pk = _packed(T, N, D, seed=16)
    p = tmp_path / "died.b2nd"
    w = SubjectProfileStreamWriter(p, T=T, N=N, D_unpacked=D)
    for t in range(T):
        w.write_session(t, pk[t])
    del w                              # no close() — the process "died"
    assert p.exists()
    with pytest.raises(ValueError, match="format_version=None"):
        read_subject_profile_packed_tnd(p)
    with pytest.raises(ValueError, match="format_version=None"):
        open_subject_profile_packed_tnd(p)


def test_stream_writer_context_manager_aborts_on_an_exception(
        tmp_path: Path) -> None:
    """A producer that dies mid-subject must leave no half-profile —
    the one-shot writer never gets as far as creating a file, so this
    one has to clean up after itself."""
    T, N, D = 3, 9, 13
    pk = _packed(T, N, D, seed=14)
    p = tmp_path / "aborted.b2nd"
    with pytest.raises(RuntimeError, match="boom"):
        with SubjectProfileStreamWriter(p, T=T, N=N, D_unpacked=D) as w:
            w.write_session(0, pk[0])
            assert p.exists()          # the frame is live at this point
            raise RuntimeError("boom")
    assert not p.exists()
    w.abort()                          # idempotent, never raises


def test_stream_writer_rejects_bad_slabs(tmp_path: Path) -> None:
    T, N, D = 3, 9, 13
    Db = (D + 7) // 8
    pk = _packed(T, N, D, seed=15)
    w = SubjectProfileStreamWriter(tmp_path / "bad.b2nd", T=T, N=N,
                                   D_unpacked=D)
    try:
        with pytest.raises(ValueError, match=r"outside \[0, 3\)"):
            w.write_session(3, pk[0])
        with pytest.raises(ValueError, match="must be uint8"):
            w.write_session(0, np.zeros((N, Db), dtype=np.float32))
        with pytest.raises(ValueError, match="slab shape"):
            w.write_session(0, np.zeros((N, Db + 1), dtype=np.uint8))
        w.write_session(0, pk[0])
        with pytest.raises(ValueError, match="written twice"):
            w.write_session(0, pk[0])
    finally:
        w.abort()
    with pytest.raises(ValueError, match="write_session after close"):
        w.write_session(1, pk[1])


@pytest.mark.parametrize("bad", [dict(T=0, N=4, D_unpacked=8),
                                 dict(T=2, N=0, D_unpacked=8),
                                 dict(T=2, N=4, D_unpacked=0),
                                 dict(T=2, N=4, D_unpacked=-8)])
def test_stream_writer_rejects_bad_geometry(tmp_path: Path, bad) -> None:
    with pytest.raises(ValueError):
        SubjectProfileStreamWriter(tmp_path / "nope.b2nd", **bad)
    assert not (tmp_path / "nope.b2nd").exists()
