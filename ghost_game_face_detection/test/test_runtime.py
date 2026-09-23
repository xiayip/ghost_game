import threading
import time

from ghost_game_face_detection.core import FaceBox
from ghost_game_face_detection.runtime import FaceTracker, LatestSlot, LatestWorker, source_age_seconds


def test_latest_slot_replaces_pending_input():
    slot = LatestSlot()
    for i in range(100):
        slot.put(i)
    assert slot.take() == 99
    assert slot.take() is None
    assert slot.overwritten == 99


def test_busy_worker_processes_newest_not_fifo():
    started,release,done = threading.Event(),threading.Event(),threading.Event()
    seen = []
    def process(job):
        seen.append(job)
        if job == 0:
            started.set()
            assert release.wait(2)
        if job == 99:
            done.set()
        return job
    worker = LatestWorker(process,lambda job,error: (_ for _ in ()).throw(error),1000)
    try:
        worker.pending.put(0)
        assert started.wait(2)
        for i in range(1,100):
            worker.pending.put(i)
        release.set()
        assert done.wait(2)
        assert seen == [0,99]
    finally:
        release.set()
        worker.close()
    assert not worker.thread.is_alive()


def test_source_age_rejects_missing_future_and_stale():
    assert source_age_seconds(0,1_000_000_000,.15) is None
    assert source_age_seconds(1_000_000_000,1_151_000_000,.15) is None
    assert source_age_seconds(1_100_000_000,1_000_000_000,.15) is None
    assert source_age_seconds(1_000_000_000,1_050_000_000,.15) == .05


def test_target_stays_with_person_when_other_person_becomes_larger():
    tracker = FaceTracker(.4)
    a = FaceBox(20,20,120,120,.95)
    b = FaceBox(400,30,100,100,.95)
    selected,track_id,_ = tracker.update([a,b],1.)
    assert selected == a
    larger_b = FaceBox(350,10,220,220,.95)
    selected,new_id,_ = tracker.update([larger_b,a],1.04)
    assert selected == a and new_id == track_id


def test_lost_track_never_republishes_old_coordinates_and_reacquires_new_id():
    tracker = FaceTracker(.4)
    a,b = FaceBox(20,20,100,100,.9),FaceBox(400,20,100,100,.9)
    _,old_id,_ = tracker.update([a],1.)
    selected,_,_ = tracker.update([b],1.1)
    assert selected is None
    selected,new_id,_ = tracker.update([b],1.5)
    assert selected == b and new_id != old_id


def test_ambiguous_crossing_and_duplicate_stamps_are_rejected():
    tracker = FaceTracker()
    a = FaceBox(100,100,100,100,.9)
    tracker.update([a],1.)
    left,right = FaceBox(90,100,100,100,.9),FaceBox(110,100,100,100,.9)
    assert tracker.update([left,right],1.04)[2] == 'ambiguous_target'
    assert tracker.update([a],1.)[2] == 'non_increasing_stamp'
