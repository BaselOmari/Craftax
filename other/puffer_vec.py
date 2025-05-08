from pufferlib.environments import atari
env_creator = atari.env_creator('breakout')

import pufferlib.vector
vecenv = pufferlib.vector.make(
    env_creator, # A callable (class or function) that returns an env
    env_args: None, # A list of arguments to pass to each environment
    env_kwargs: None, # A list of dictionary keyword arguments to pass to each environment
    backend: Serial, # pufferlib.vector.[Serial|Multiprocessing|Native|Ray]
    num_envs: 1, # The total number of environments to create
    **kwargs # extra backend-specific options
)

# Make 4 copies of Breakout on the current process
vecenv = pufferlib.vector.make(env_creator, num_envs=4,
    backend=pufferlib.vector.Serial)

# Make 4 copies of Breakout, each on a separate process
vecenv = pufferlib.vector.make(env_creator, num_envs=4,
    backend=pufferlib.vector.Multiprocessing)

# Make 4 copies of Breakout, 2 on each of 2 processes
vecenv = pufferlib.vector.make(env_creator, num_envs=4,
    backend=pufferlib.vector.Multiprocessing, num_workers=2)

# Make 4 copies of Breakout, 2 on each of 2 processes,
# but only get two observations per step
vecenv = pufferlib.vector.make(env_creator, num_envs=4,
    backend=pufferlib.vector.Multiprocessing, num_workers=2,
    batch_size=2)

# Make 1024 instances of Ocean breakout on the current process
from pufferlib.ocean import Breakout
vecenv = pufferlib.vector.make(Breakout,
    backend=pufferlib.vector.Native,
    env_kwargs={'num_envs': 1024},
)

# Notice that Native envs handle multiple instances internally.
# You can still multiprocess/async, but don't make multiple external
# copies per process.
vecenv = pufferlib.vector.make(Breakout, num_envs=2,
    backend=pufferlib.vector.Multiprocessing, batch_size=1)

# Synchronous API - reset/step
import time
vecenv = pufferlib.vector.make(Breakout, num_envs=2,
    backend=pufferlib.vector.Multiprocessing)
vecenv.reset()
start, steps, TIMEOUT = time.time(), 0, 3
while time.time() - start < TIMEOUT:
    vecenv.step(vecenv.action_space.sample())
    steps += 1

vecenv.close()
print('Puffer FPS:', steps*vecenv.num_envs/TIMEOUT)

# Async API - async_reset, send/recv
# Call your model between recv() and send()
vecenv = pufferlib.vector.make(Breakout, num_envs=2,
    backend=pufferlib.vector.Multiprocessing, batch_size=1)
vecenv.async_reset()
start, steps, TIMEOUT = time.time(), 0, 3
while time.time() - start < TIMEOUT:
    vecenv.recv()
    vecenv.send(vecenv.action_space.sample())
    steps += 1

vecenv.close()
print('Puffer Async FPS:', steps*vecenv.num_envs/TIMEOUT)
