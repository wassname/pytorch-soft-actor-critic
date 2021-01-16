



# %%
from gym_recording_modified.playback import get_recordings
from tqdm.auto import tqdm
from replay_memory import ReplayMemory
from pathlib import Path
from loguru import logger

def load_demonstrations(mem: ReplayMemory, recordings: Path):
    records = get_recordings(str(recordings))
    logger.info('picks in recordings', sum(records['reward']>10))
    ends=records["episodes_end_point"]
    for i in tqdm(range(len(ends)-1), desc='loading demonstrations'):
        a = ends[i]
        b = ends[i+1]
        for s in range(a+1, b):
            r = records['reward'][s]
            o = records['observation'][s-1]
            a = records['action'][s]
            no = records['observation'][s]
            t = s == b
            mem.push(o, a, r, no, t)

# %%

if __name__ == "__main__":
    # TEST
    from replay_memory import ReplayMemory
    from pathlib import Path

    mem = ReplayMemory(10000, 42)
    load_demonstrations(mem, Path("/media/wassname/Storage5/projects2/3ST/diy_bullet_conveyor/apple_gym/data/demonstrations"))


